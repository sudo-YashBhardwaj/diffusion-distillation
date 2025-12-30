#!/usr/bin/env python3
"""
Flash Diffusion LoRA Training - Proper K-Step Distillation

This implements the actual Flash Diffusion / LCM training method:
- Teacher: Takes K DDIM steps from t -> t-k with CFG
- Student: Learns to match that K-step jump in ONE step (no CFG)
- Result: Student can generate in 2-8 steps

Key insight: The student doesn't just match teacher's noise prediction,
it learns to SKIP multiple denoising steps at once.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm.auto import tqdm


# =============================================================================
# Dataset
# =============================================================================

class ImageCaptionDataset(Dataset):
    """Simple dataset for image-caption pairs."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        resolution: int = 512,
    ):
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        if dataset_name:
            print(f"Loading dataset: {dataset_name}")
            self.dataset = load_dataset(dataset_name, split="train")
            self.use_hf = True
        elif data_dir:
            print(f"Loading dataset from local directory: {data_dir}")
            data_path = Path(data_dir)
            if not data_path.exists():
                raise ValueError(f"Data directory does not exist: {data_dir}")

            # Expect structure: data_dir/images/*.png and data_dir/captions/*.txt
            # Or a single folder with paired image/text files
            images_dir = data_path / "images"
            captions_dir = data_path / "captions"

            if images_dir.exists() and captions_dir.exists():
                self.image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
                self.caption_files = sorted(list(captions_dir.glob("*.txt")))
                if len(self.image_files) != len(self.caption_files):
                    raise ValueError(f"Mismatch: {len(self.image_files)} images vs {len(self.caption_files)} captions")
            else:
                # Try to find image files and matching text files
                image_files = sorted(
                    list(data_path.glob("*.png")) + list(data_path.glob("*.jpg")) + list(data_path.glob("*.jpeg"))
                )
                self.image_files = []
                self.caption_files = []
                for img_file in image_files:
                    txt_file = img_file.with_suffix(".txt")
                    if txt_file.exists():
                        self.image_files.append(img_file)
                        self.caption_files.append(txt_file)

            if len(self.image_files) == 0:
                raise ValueError(f"No image files found in {data_dir}")
            print(f"Found {len(self.image_files)} image-caption pairs")
            self.use_hf = False
        else:
            raise ValueError("Provide dataset_name or data_dir")

    def __len__(self):
        return len(self.dataset) if self.use_hf else len(self.image_files)

    def __getitem__(self, idx):
        if self.use_hf:
            item = self.dataset[idx]
            image = item["image"].convert("RGB") if hasattr(item["image"], "convert") else Image.open(item["image"]).convert("RGB")
            caption = item.get("text") or item.get("caption") or item.get("prompt") or ""
        else:
            image = Image.open(self.image_files[idx]).convert("RGB")
            caption = self.caption_files[idx].read_text().strip()
        
        return {"pixel_values": self.transform(image), "text": caption}


# =============================================================================
# Core Training Functions
# =============================================================================

def get_lcm_timesteps(num_train_timesteps: int = 1000, num_inference_steps: int = 4) -> List[int]:
    """
    Get timesteps that match LCM/Flash inference schedule.
    
    For 4 steps: [999, 749, 499, 249] (trailing spacing)
    """
    step_ratio = num_train_timesteps // num_inference_steps
    # Trailing timestep spacing (what LCMScheduler uses)
    timesteps = [num_train_timesteps - 1 - i * step_ratio for i in range(num_inference_steps)]
    return timesteps


def ddim_step(
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    scheduler: DDIMScheduler,
    prev_timestep: Optional[int] = None,
) -> torch.Tensor:
    """
    Single DDIM step: predict x_{t-1} from x_t.
    
    Args:
        model_output: Predicted noise from UNet
        timestep: Current timestep t
        sample: Current noisy latent x_t
        scheduler: DDIM scheduler (for alpha values)
        prev_timestep: Target timestep (if None, uses scheduler's default)
    
    Returns:
        Denoised sample x_{t-1}
    """
    # Get alpha values
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    
    if prev_timestep is None:
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_timestep = max(prev_timestep, 0)
    
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else torch.tensor(1.0)
    
    # Ensure tensors are on same device
    alpha_prod_t = alpha_prod_t.to(sample.device)
    alpha_prod_t_prev = alpha_prod_t_prev.to(sample.device)
    
    beta_prod_t = 1 - alpha_prod_t
    
    # Predict x0 from noise prediction
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    
    # Clip for stability
    pred_original_sample = pred_original_sample.clamp(-1, 1)
    
    # Compute x_{t-1}
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    
    return prev_sample


def teacher_multistep(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    latents: torch.Tensor,
    start_timestep: int,
    num_steps: int,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    guidance_scale: float = 7.5,
) -> torch.Tensor:
    """
    Teacher performs multiple DDIM steps with CFG.
    
    This is what the student learns to match in ONE step.
    
    Args:
        unet: Teacher UNet (frozen)
        scheduler: DDIM scheduler
        latents: Starting noisy latents at start_timestep
        start_timestep: Where to start denoising
        num_steps: How many DDIM steps to take
        prompt_embeds: Text embeddings
        negative_prompt_embeds: Unconditional embeddings
        guidance_scale: CFG scale
    
    Returns:
        Latents after num_steps of denoising
    """
    current_latents = latents.clone()
    
    # Create timestep sequence
    step_size = start_timestep // num_steps if num_steps > 0 else start_timestep
    timesteps = []
    t = start_timestep
    for _ in range(num_steps):
        timesteps.append(t)
        t = max(t - step_size, 0)
        if t == 0:
            break
    
    for t in timesteps:
        t_tensor = torch.tensor([t], device=latents.device, dtype=torch.long)
        
        # CFG: concatenate for batched inference
        latent_input = torch.cat([current_latents, current_latents])
        t_input = torch.cat([t_tensor, t_tensor])
        embed_input = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # Get noise prediction
        with torch.no_grad():
            noise_pred = unet(latent_input, t_input, encoder_hidden_states=embed_input).sample
        
        # Apply CFG
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # DDIM step
        prev_t = max(t - step_size, 0)
        current_latents = ddim_step(noise_pred, t, current_latents, scheduler, prev_t)
    
    return current_latents


def compute_loss(
    vae: AutoencoderKL,
    tokenizer,
    text_encoder: CLIPTextModel,
    teacher_unet: UNet2DConditionModel,
    student_unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    ddim_scheduler: DDIMScheduler,
    batch: Dict[str, torch.Tensor],
    num_ddim_steps: int = 10,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute Flash Diffusion / LCM style distillation loss.
    
    Key idea:
    1. Sample a timestep t from the inference schedule
    2. Teacher: run K DDIM steps from t with CFG -> get target latents
    3. Student: single prediction at t (no CFG) -> predict target latents
    4. Loss: MSE between student output and teacher target
    
    The student learns to skip K steps in one forward pass.
    """
    images = batch["pixel_values"].to(device, dtype=dtype)
    texts = batch["text"]
    batch_size = images.shape[0]

    # Encode images to latents
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Encode text
    text_inputs = tokenizer(
        texts, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    uncond_inputs = tokenizer(
        [""] * batch_size, padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        all_ids = torch.cat([uncond_inputs.input_ids, text_inputs.input_ids]).to(device)
        all_embeds = text_encoder(all_ids)[0]
        uncond_embeds, text_embeds = all_embeds.chunk(2)

    # Get discrete timesteps matching inference schedule
    inference_timesteps = get_lcm_timesteps(
        noise_scheduler.config.num_train_timesteps, 
        num_inference_steps
    )
    
    # Sample timesteps for this batch (from inference schedule)
    timestep_indices = torch.randint(0, len(inference_timesteps), (batch_size,))
    timesteps = torch.tensor(
        [inference_timesteps[i] for i in timestep_indices],
        device=device, dtype=torch.long
    )

    # Add noise at sampled timesteps
    noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # For each sample in batch:
    # - Teacher: K DDIM steps from t with CFG
    # - Student: single step prediction (no CFG)
    
    teacher_targets = []
    student_outputs = []
    
    for b in range(batch_size):
        t = int(timesteps[b].item())
        z_noisy = noisy_latents[b:b+1]
        
        # Teacher: K-step denoising with CFG
        with torch.no_grad():
            teacher_target = teacher_multistep(
                unet=teacher_unet,
                scheduler=ddim_scheduler,
                latents=z_noisy,
                start_timestep=t,
                num_steps=num_ddim_steps,
                prompt_embeds=text_embeds[b:b+1],
                negative_prompt_embeds=uncond_embeds[b:b+1],
                guidance_scale=guidance_scale,
            )
        teacher_targets.append(teacher_target)
        
        # Student: single forward pass (no CFG)
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        student_pred = student_unet(
            z_noisy, t_tensor, encoder_hidden_states=text_embeds[b:b+1]
        ).sample
        
        # Convert student noise prediction to denoised output
        # Using same formula as DDIM but skipping all K steps at once
        alpha_t = noise_scheduler.alphas_cumprod[t].to(device)
        
        # Compute what timestep teacher ended at
        step_size = t // num_ddim_steps if num_ddim_steps > 0 else t
        target_t = max(t - step_size * num_ddim_steps, 0)
        alpha_target = noise_scheduler.alphas_cumprod[target_t].to(device) if target_t > 0 else torch.tensor(1.0, device=device)
        
        # Student's predicted x0
        pred_x0 = (z_noisy - (1 - alpha_t) ** 0.5 * student_pred) / alpha_t ** 0.5
        pred_x0 = pred_x0.clamp(-1, 1)
        
        # Student's output at target timestep
        student_output = alpha_target ** 0.5 * pred_x0 + (1 - alpha_target) ** 0.5 * student_pred
        student_outputs.append(student_output)
    
    # Stack and compute loss
    teacher_targets = torch.cat(teacher_targets, dim=0)
    student_outputs = torch.cat(student_outputs, dim=0)
    
    loss = F.mse_loss(student_outputs.float(), teacher_targets.float())
    
    metrics = {
        "loss": loss.item(),
        "timestep_mean": timesteps.float().mean().item(),
        "student_std": student_outputs.std().item(),
        "teacher_std": teacher_targets.std().item(),
    }
    
    return loss, metrics


# =============================================================================
# Model Setup
# =============================================================================

def prepare_models(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_rank: int = 4,
    lora_alpha: int = 32,
    dtype: torch.dtype = torch.float16,
):
    """Load and prepare teacher/student models."""
    print(f"Loading base model: {base_model_id}")

    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=dtype)
    
    # Teacher (frozen)
    teacher_unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=dtype)
    teacher_unet.requires_grad_(False)
    teacher_unet.eval()

    # Student (with LoRA)
    student_unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=dtype)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    student_unet = get_peft_model(student_unet, lora_config)
    student_unet.print_trainable_parameters()

    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    ddim_scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

    return vae, tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler, ddim_scheduler


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Flash Diffusion LoRA Training")
    parser.add_argument("--dataset_name", type=str, default="lambdalabs/pokemon-blip-captions")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default="flash_lora")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale for teacher")
    parser.add_argument("--num_ddim_steps", type=int, default=10, help="Teacher DDIM steps per training step")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Target inference steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )

    # Dataset
    dataset = ImageCaptionDataset(
        dataset_name=args.dataset_name if not args.data_dir else None,
        data_dir=args.data_dir,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    # Models
    vae, tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler, ddim_scheduler = prepare_models(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in student_unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    # Prepare
    student_unet, optimizer, dataloader = accelerator.prepare(student_unet, optimizer, dataloader)
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    teacher_unet = teacher_unet.to(accelerator.device)

    # Output
    output_dir = Path(args.output_dir) / args.run_name / "lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training info
    inference_timesteps = get_lcm_timesteps(noise_scheduler.config.num_train_timesteps, args.num_inference_steps)
    
    print(f"\n{'='*60}")
    print("FLASH DIFFUSION TRAINING")
    print(f"{'='*60}")
    print(f"Method: K-step teacher rollout distillation")
    print(f"Teacher DDIM steps: {args.num_ddim_steps}")
    print(f"Target inference steps: {args.num_inference_steps}")
    print(f"Training timesteps: {inference_timesteps}")
    print(f"Teacher CFG scale: {args.guidance_scale}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Training loop
    student_unet.train()
    global_step = 0
    loss_history = []
    start_time = time.time()
    
    progress_bar = tqdm(range(args.max_steps), desc="Training")

    for epoch in range(1000):
        for batch in dataloader:
            with accelerator.accumulate(student_unet):
                loss, metrics = compute_loss(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    teacher_unet=teacher_unet,
                    student_unet=student_unet,
                    noise_scheduler=noise_scheduler,
                    ddim_scheduler=ddim_scheduler,
                    batch=batch,
                    num_ddim_steps=args.num_ddim_steps,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    device=accelerator.device,
                    dtype=torch.float16,
                )

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_unet.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    loss_history.append(metrics["loss"])
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": f"{metrics['loss']:.4f}"})

                    # Log every 50 steps
                    if global_step % 50 == 0:
                        avg_loss = sum(loss_history[-50:]) / min(50, len(loss_history))
                        elapsed = time.time() - start_time
                        print(f"\n[Step {global_step}] Loss: {metrics['loss']:.4f} (avg: {avg_loss:.4f})")
                        print(f"  Timestep mean: {metrics['timestep_mean']:.0f}")
                        print(f"  Student/Teacher std: {metrics['student_std']:.4f} / {metrics['teacher_std']:.4f}")
                        print(f"  Speed: {global_step/elapsed:.2f} steps/sec")

                    # Checkpoint every 500 steps
                    if global_step % 500 == 0:
                        ckpt_dir = output_dir / f"checkpoint-{global_step}"
                        accelerator.unwrap_model(student_unet).save_pretrained(ckpt_dir)
                        print(f"  Saved checkpoint: {ckpt_dir}")

                    if global_step >= args.max_steps:
                        break

        if global_step >= args.max_steps:
            break

    # Final save
    total_time = time.time() - start_time
    accelerator.unwrap_model(student_unet).save_pretrained(output_dir / "final")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total steps: {global_step}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Initial loss: {loss_history[0]:.4f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Loss change: {loss_history[0] - loss_history[-1]:.4f}")
    print(f"Saved to: {output_dir / 'final'}")
    print(f"{'='*60}\n")
    
    # Save training summary
    summary = {
        "total_steps": global_step,
        "total_time_minutes": total_time / 60,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "config": vars(args),
    }
    with open(output_dir / "final" / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()