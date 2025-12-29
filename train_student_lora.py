#!/usr/bin/env python3
"""
Train a LoRA student model via distillation from a teacher SD1.5 model.

This script implements few-step distillation aligned with Flash Diffusion / LCM-LoRA:
- Teacher: Frozen SD1.5 UNet that solves from timestep t -> 0 using multiple steps with CFG
- Student: SD1.5 UNet with trainable LoRA adapters that predicts at discrete few-step timesteps
- Training: Student learns to match teacher's final z0 (denoised latent) from a single prediction
- Loss: MSE between z0_student and z0_teacher
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm.auto import tqdm


class ImageCaptionDataset(Dataset):
    """Dataset for image-caption pairs."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        resolution: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            dataset_name: Hugging Face dataset name (e.g., 'diffusers/tuxemon')
            data_dir: Local directory with images and captions (alternative to dataset_name)
            resolution: Image resolution for training
        """
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # [-1, 1]
            ]
        )

        if dataset_name:
            print(f"Loading dataset from Hugging Face: {dataset_name}")
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
            raise ValueError("Either dataset_name or data_dir must be provided")

    def __len__(self) -> int:
        """Return dataset size."""
        if self.use_hf:
            return len(self.dataset)
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item."""
        if self.use_hf:
            item = self.dataset[idx]
            image = item["image"]
            if isinstance(image, str):
                image = Image.open(image)
            # Ensure image is RGB before transforms
            image = image.convert("RGB")
            # Handle different caption field names, including COCO format
            caption = item.get("text") or item.get("caption") or item.get("prompt") or ""
            # COCO datasets often have nested caption structures
            if not caption and "sentences" in item:
                sentences = item["sentences"]
                if isinstance(sentences, list) and len(sentences) > 0:
                    if isinstance(sentences[0], dict) and "raw" in sentences[0]:
                        caption = sentences[0]["raw"]
                    elif isinstance(sentences[0], str):
                        caption = sentences[0]
            if not caption and "annotations" in item:
                anns = item["annotations"]
                if isinstance(anns, list) and len(anns) > 0:
                    if isinstance(anns[0], dict) and "caption" in anns[0]:
                        caption = anns[0]["caption"]
            if not caption:
                caption = ""
        else:
            image = Image.open(self.image_files[idx]).convert("RGB")
            caption = self.caption_files[idx].read_text().strip()

        # Transform image
        image_tensor = self.transform(image)

        return {"pixel_values": image_tensor, "text": caption}


def prepare_models(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_rank: int = 4,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Prepare teacher and student models.

    Args:
        base_model_id: Base model identifier
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        device: Device to use
        dtype: Data type

    Returns:
        Tuple of (vae, tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler, teacher_scheduler)
    """
    print(f"Loading base model: {base_model_id}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        base_model_id,
        subfolder="vae",
        torch_dtype=dtype,
    )

    # Load teacher UNet (frozen)
    teacher_unet = UNet2DConditionModel.from_pretrained(
        base_model_id,
        subfolder="unet",
        torch_dtype=dtype,
    )
    teacher_unet.requires_grad_(False)
    teacher_unet.eval()
    
    # Compile teacher for faster inference (PyTorch 2.0+)
    try:
        teacher_unet = torch.compile(teacher_unet, mode="reduce-overhead")
        print("✓ Teacher UNet compiled with torch.compile")
    except Exception as e:
        print(f"Note: torch.compile not available ({e}), using standard mode")

    # Load student UNet with LoRA
    student_unet = UNet2DConditionModel.from_pretrained(
        base_model_id,
        subfolder="unet",
        torch_dtype=dtype,
    )

    # Add LoRA adapters
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=lora_dropout,
    )
    student_unet = get_peft_model(student_unet, lora_config)
    student_unet.print_trainable_parameters()
    
    # Note: torch.compile disabled for student due to LoRA compatibility issues
    # Teacher model can still use compile for faster inference

    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_id, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # Noise scheduler: DDPMScheduler for adding noise and alphas
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    
    # Teacher scheduler: DDIMScheduler for teacher denoising (handles arbitrary timesteps)
    teacher_scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

    return vae, tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler, teacher_scheduler


def compute_loss(
    vae: AutoencoderKL,
    tokenizer,
    text_encoder,
    teacher_unet: UNet2DConditionModel,
    student_unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    teacher_scheduler: DDIMScheduler,
    batch: Dict[str, torch.Tensor],
    teacher_steps: int = 50,
    student_steps: int = 4,
    guidance_scale: float = 7.5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Compute distillation loss using few-step distillation aligned with Flash Diffusion / LCM-LoRA.
    
    The teacher solves from timestep t to 0 using multiple steps with CFG.
    The student predicts at timestep t and is trained to match the teacher's final z0.

    Args:
        vae: VAE encoder/decoder
        tokenizer: Text tokenizer
        text_encoder: Text encoder
        teacher_unet: Frozen teacher UNet
        student_unet: Trainable student UNet
        noise_scheduler: DDPMScheduler for adding noise and alphas
        teacher_scheduler: DDIMScheduler for teacher denoising
        batch: Batch of images and captions
        teacher_steps: Number of steps for teacher denoising
        student_steps: Unused (kept for compatibility)
        guidance_scale: Guidance scale for teacher CFG
        device: Device
        dtype: Data type

    Returns:
        Loss tensor (MSE between z0_student and z0_teacher)
    """
    images = batch["pixel_values"].to(device, dtype=dtype)
    texts = batch["text"]

    # Encode images to latents
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Encode text (batch all text encoding together for efficiency)
    batch_size = latents.shape[0]
    text_inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # Prepare unconditional inputs for CFG (batch together)
    uncond_inputs = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        # Encode both conditional and unconditional embeddings in one pass
        all_input_ids = torch.cat([uncond_inputs.input_ids, text_inputs.input_ids], dim=0).to(device)
        all_embeddings = text_encoder(all_input_ids)[0]
        uncond_embeddings, text_embeddings = all_embeddings.chunk(2, dim=0)

    # Sample training timesteps uniformly from [0, num_train_timesteps)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=device,
    )

    # Add noise to latents using noise_scheduler (DDPMScheduler)
    noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Teacher: Solve from t -> 0 using multiple steps with CFG
    with torch.no_grad():
        
        # Calculate stride for consistent schedule
        stride = noise_scheduler.config.num_train_timesteps // teacher_steps  # e.g., 1000//50=20
        
        z0_teacher_list = []
        for b_idx in range(batch_size):
            t0 = int(timesteps[b_idx].item())
            z = noisy_latents[b_idx:b_idx+1]
            
            # Make a consistent schedule: t0, t0-stride, ..., 0 (clamped)
            t_seq = list(range(t0, -1, -stride))
            if t_seq[-1] != 0:
                t_seq.append(0)
            
            # Set timesteps for DDIMScheduler once per batch (optimization)
            if b_idx == 0:
                teacher_scheduler.set_timesteps(teacher_steps, device=device)
            
            for step_idx, t_cur in enumerate(t_seq):
                # Clamp timestep to valid range
                t_cur = max(0, min(t_cur, teacher_scheduler.config.num_train_timesteps - 1))
                t_tensor = torch.tensor([t_cur], device=device, dtype=torch.long)
                
                # CFG
                z_cfg = torch.cat([z, z], dim=0)
                t_cfg = torch.cat([t_tensor, t_tensor], dim=0)
                emb_cfg = torch.cat([uncond_embeddings[b_idx:b_idx+1], text_embeddings[b_idx:b_idx+1]], dim=0)
                
                eps = teacher_unet(z_cfg, t_cfg, encoder_hidden_states=emb_cfg).sample
                eps_u, eps_c = eps.chunk(2)
                eps = eps_u + guidance_scale * (eps_c - eps_u)
                
                # DDIMScheduler.step() can handle arbitrary timesteps
                z = teacher_scheduler.step(eps, t_cur, z).prev_sample
            
            z0_teacher_list.append(z)
        
        z0_teacher = torch.cat(z0_teacher_list, dim=0)

    # Student: Single prediction at timestep t
    student_pred = student_unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_embeddings,
    ).sample
    
    # Student z0 reconstruction using the SAME alphas as noise_scheduler (vectorized)
    # Get alphas for all timesteps in batch at once
    alphas = noise_scheduler.alphas_cumprod[timesteps].to(device=device, dtype=dtype)
    alphas = alphas.view(-1, 1, 1, 1)  # Broadcast shape
    z0_student = (noisy_latents - (1 - alphas).sqrt() * student_pred) / alphas.sqrt()

    # Distillation loss: MSE between z0_student and z0_teacher
    loss = F.mse_loss(z0_student.float(), z0_teacher.float(), reduction="mean")
    
    # Verify loss is valid (not NaN or Inf)
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: Invalid loss detected! Loss = {loss.item()}")
        print(f"  z0_student stats: min={z0_student.min().item():.4f}, max={z0_student.max().item():.4f}, mean={z0_student.mean().item():.4f}, std={z0_student.std().item():.4f}")
        print(f"  z0_teacher stats: min={z0_teacher.min().item():.4f}, max={z0_teacher.max().item():.4f}, mean={z0_teacher.mean().item():.4f}, std={z0_teacher.std().item():.4f}")
        print(f"  Timesteps used: {timesteps.cpu().tolist()}")
        print(f"  Batch size: {batch_size}")

    return loss


def main() -> None:
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train LoRA student via distillation")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset name (e.g., 'diffusers/tuxemon')",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Local directory with images and captions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="lora_student",
        help="Run name for checkpoint directory",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Maximum training steps (reduced for faster training with large datasets)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--teacher_steps",
        type=int,
        default=20,
        help="Number of steps for teacher denoising (fewer = faster training)",
    )
    parser.add_argument(
        "--student_steps",
        type=int,
        default=4,
        help="Number of steps for student inference",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for teacher classifier-free guidance",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Validate that dataset is provided
    if args.dataset_name is None and args.data_dir is None:
        # Default to tuxemon dataset if nothing specified
        args.dataset_name = "diffusers/tuxemon"
        print(f"No dataset specified, using default: {args.dataset_name}")

    # Set seed
    set_seed(args.seed)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )

    # Prepare dataset
    dataset = ImageCaptionDataset(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        resolution=args.resolution,
    )

    # Create dataloader with optimized settings for large datasets
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Increased for faster data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
    )

    # Prepare models
    vae, tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler, teacher_scheduler = prepare_models(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=accelerator.device,
        dtype=torch.float16,
    )

    # Setup optimizer (only trainable LoRA parameters)
    trainable_params = [p for p in student_unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
    )

    # Prepare with accelerator
    student_unet, optimizer, train_dataloader = accelerator.prepare(
        student_unet, optimizer, train_dataloader
    )
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    teacher_unet = teacher_unet.to(accelerator.device)

    # Create output directory
    output_dir = Path(args.output_dir) / args.run_name / "lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-training verification and test forward pass
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"PRE-TRAINING VERIFICATION")
        print(f"{'='*60}")
        
        # Model verification
        trainable_params = sum(p.numel() for p in student_unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in student_unet.parameters())
        print(f"\n[Model Status]")
        print(f"  Student UNet in training mode: {student_unet.training}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}% of total)")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Optimizer learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"  Number of parameter groups: {len(optimizer.param_groups)}")
        
        # Verify teacher is frozen
        teacher_trainable = sum(p.numel() for p in teacher_unet.parameters() if p.requires_grad)
        print(f"\n[Teacher Status]")
        print(f"  Teacher UNet trainable params: {teacher_trainable} (should be 0)")
        print(f"  Teacher UNet in eval mode: {teacher_unet.training == False}")
        
        # Verify text encoder is frozen
        text_encoder_trainable = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
        print(f"\n[Text Encoder Status]")
        print(f"  Text encoder trainable params: {text_encoder_trainable} (should be 0)")
        print(f"  Text encoder in eval mode: {text_encoder.training == False}")
        
        # Scheduler verification
        print(f"\n[Scheduler Configuration]")
        print(f"  Noise scheduler (DDPMScheduler):")
        print(f"    num_train_timesteps: {noise_scheduler.config.num_train_timesteps}")
        print(f"    beta_start: {noise_scheduler.config.beta_start}")
        print(f"    beta_end: {noise_scheduler.config.beta_end}")
        print(f"  Teacher scheduler (DDIMScheduler):")
        print(f"    num_train_timesteps: {teacher_scheduler.config.num_train_timesteps}")
        print(f"    num_inference_steps: {getattr(teacher_scheduler.config, 'num_inference_steps', 'N/A')}")
        
        # Warnings
        warnings = []
        if teacher_trainable > 0:
            warnings.append(f"  ⚠️  Teacher UNet has {teacher_trainable} trainable parameters (should be frozen)!")
        if text_encoder_trainable > 0:
            warnings.append(f"  ⚠️  Text encoder has {text_encoder_trainable} trainable parameters (should be frozen)!")
        if trainable_params == 0:
            warnings.append(f"  ❌ ERROR: No trainable parameters found! Training will not work!")
        if warnings:
            print(f"\n[Warnings]")
            for w in warnings:
                print(w)
        else:
            print(f"\n  ✅ All checks passed!")
        
        # Test forward pass
        print(f"\n[Test Forward Pass]")
        print(f"  Running test forward pass to verify training setup...")
        try:
            # Get a sample batch
            sample_batch = next(iter(train_dataloader))
            
            # Test loss computation
            with torch.no_grad():
                test_loss = compute_loss(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    teacher_unet=teacher_unet,
                    student_unet=student_unet,
                    noise_scheduler=noise_scheduler,
                    teacher_scheduler=teacher_scheduler,
                    batch=sample_batch,
                    teacher_steps=args.teacher_steps,
                    student_steps=args.student_steps,
                    guidance_scale=args.guidance_scale,
                    device=accelerator.device,
                    dtype=torch.float16,
                )
            
            print(f"  ✅ Test forward pass successful!")
            print(f"  Test loss (no grad): {test_loss.item():.6f}")
            
            # Test backward pass
            print(f"  Testing backward pass...")
            student_unet.train()
            optimizer.zero_grad()
            test_loss_grad = compute_loss(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                teacher_unet=teacher_unet,
                student_unet=student_unet,
                noise_scheduler=noise_scheduler,
                teacher_scheduler=teacher_scheduler,
                batch=sample_batch,
                teacher_steps=args.teacher_steps,
                student_steps=args.student_steps,
                guidance_scale=args.guidance_scale,
                device=accelerator.device,
                dtype=torch.float16,
            )
            
            accelerator.backward(test_loss_grad)
            
            # Check gradients
            grad_count = 0
            grad_norm_sum = 0.0
            for param in student_unet.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_count += 1
                    grad_norm_sum += param.grad.data.norm(2).item() ** 2
            
            if grad_count > 0:
                grad_norm = grad_norm_sum ** 0.5
                print(f"  ✅ Backward pass successful!")
                print(f"  Test loss (with grad): {test_loss_grad.item():.6f}")
                print(f"  Parameters with gradients: {grad_count}")
                print(f"  Gradient norm: {grad_norm:.6f}")
            else:
                print(f"  ❌ ERROR: No gradients computed! Training will not work!")
            
            optimizer.zero_grad()
            student_unet.train()
            
        except Exception as e:
            print(f"  ❌ ERROR in test forward/backward pass: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"{'='*60}\n")
    
    # Training loop
    student_unet.train()
    global_step = 0
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_local_main_process)

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Starting training for {args.max_steps} steps...")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}, LoRA alpha: {args.lora_alpha}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Device: {accelerator.device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(accelerator.device)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(accelerator.device) / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(accelerator.device) / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Training metrics tracking
    loss_history = []
    gradient_norm_history = []
    training_start_time = time.time()
    step_times = []

    for epoch in range(1000):  # Large number, break on max_steps
        epoch_start_time = time.time()
        print(f"\n[Epoch {epoch + 1}] Starting epoch...")
        
        for batch_idx, batch in enumerate(train_dataloader):
            step_start_time = time.time()
            
            with accelerator.accumulate(student_unet):
                # Compute loss
                loss = compute_loss(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    teacher_unet=teacher_unet,
                    student_unet=student_unet,
                    noise_scheduler=noise_scheduler,
                    teacher_scheduler=teacher_scheduler,
                    batch=batch,
                    teacher_steps=args.teacher_steps,
                    student_steps=args.student_steps,
                    guidance_scale=args.guidance_scale,
                    device=accelerator.device,
                    dtype=torch.float16,
                )

                # Backward pass
                accelerator.backward(loss)
                
                # Calculate gradient norm before optimizer step
                grad_norm = 0.0
                if accelerator.sync_gradients:
                    # Get gradient norm for trainable parameters
                    for param in student_unet.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    gradient_norm_history.append(grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()

                # Only update step counter and progress on actual optimizer updates
                if accelerator.sync_gradients:
                    step_time = time.time() - step_start_time
                    step_times.append(step_time)
                    global_step += 1
                    loss_value = loss.item()
                    loss_history.append(loss_value)
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": f"{loss_value:.6f}",
                        "grad_norm": f"{grad_norm:.4f}" if grad_norm > 0 else "N/A",
                    })
                    
                    # Detailed logging every N steps
                    log_interval = 10
                    if global_step % log_interval == 0 and accelerator.is_main_process:
                        avg_loss = sum(loss_history[-log_interval:]) / min(log_interval, len(loss_history))
                        avg_grad_norm = sum(gradient_norm_history[-log_interval:]) / min(log_interval, len(gradient_norm_history)) if gradient_norm_history else 0.0
                        avg_step_time = sum(step_times[-log_interval:]) / min(log_interval, len(step_times))
                        images_per_sec = args.batch_size * args.gradient_accumulation_steps / avg_step_time if avg_step_time > 0 else 0.0
                        
                        elapsed_time = time.time() - training_start_time
                        steps_per_sec = global_step / elapsed_time if elapsed_time > 0 else 0.0
                        remaining_steps = args.max_steps - global_step
                        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0
                        eta_minutes = eta_seconds / 60.0
                        
                        print(f"\n[Step {global_step}/{args.max_steps}]")
                        print(f"  Loss: {loss_value:.6f} (avg over last {log_interval}: {avg_loss:.6f})")
                        print(f"  Gradient norm: {grad_norm:.4f} (avg: {avg_grad_norm:.4f})")
                        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
                        print(f"  Speed: {avg_step_time:.3f}s/step, {images_per_sec:.2f} images/sec, {steps_per_sec:.2f} steps/sec")
                        print(f"  ETA: {eta_minutes:.1f} minutes ({eta_seconds:.0f} seconds)")
                        
                        if torch.cuda.is_available():
                            mem_allocated = torch.cuda.memory_allocated(accelerator.device) / 1e9
                            mem_reserved = torch.cuda.memory_reserved(accelerator.device) / 1e9
                            print(f"  GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
                        
                        # Check if model parameters are updating
                        if global_step == log_interval:
                            # Get a sample parameter to check if it's changing
                            sample_param = None
                            for param in student_unet.parameters():
                                if param.requires_grad and param.numel() > 0:
                                    sample_param = param.data.clone()
                                    break
                            if sample_param is not None:
                                print(f"  Model parameters are trainable (sample param shape: {sample_param.shape})")

                    # Save checkpoint
                    if global_step % 500 == 0 and accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(student_unet)
                        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                        unwrapped_model.save_pretrained(checkpoint_dir)
                        
                        # Save training metrics
                        metrics = {
                            "global_step": global_step,
                            "loss": loss_value,
                            "avg_loss": sum(loss_history[-100:]) / min(100, len(loss_history)) if loss_history else 0.0,
                            "grad_norm": grad_norm,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "epoch": epoch + 1,
                        }
                        import json
                        with open(checkpoint_dir / "training_metrics.json", "w") as f:
                            json.dump(metrics, f, indent=2)
                        
                        print(f"\n[Step {global_step}] Saved checkpoint to {checkpoint_dir}")
                        print(f"  Current loss: {loss_value:.6f}")
                        print(f"  Average loss (last 100 steps): {metrics['avg_loss']:.6f}")
                        if torch.cuda.is_available():
                            print(f"  GPU Memory: {torch.cuda.memory_allocated(accelerator.device) / 1e9:.2f} GB")

                    if global_step >= args.max_steps:
                        break
        
        epoch_time = time.time() - epoch_start_time
        print(f"[Epoch {epoch + 1}] Completed in {epoch_time:.2f} seconds")
        
        if global_step >= args.max_steps:
            break

    # Final save
    if accelerator.is_main_process:
        total_training_time = time.time() - training_start_time
        unwrapped_model = accelerator.unwrap_model(student_unet)
        unwrapped_model.save_pretrained(output_dir / "final")
        
        # Final training summary
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total steps: {global_step}")
        print(f"Total training time: {total_training_time / 60:.2f} minutes ({total_training_time:.2f} seconds)")
        print(f"Average time per step: {total_training_time / global_step:.3f} seconds")
        print(f"Average steps per second: {global_step / total_training_time:.2f}")
        
        if loss_history:
            final_loss = loss_history[-1]
            initial_loss = loss_history[0] if len(loss_history) > 0 else final_loss
            avg_loss = sum(loss_history) / len(loss_history)
            min_loss = min(loss_history)
            max_loss = max(loss_history)
            
            print(f"\nLoss Statistics:")
            print(f"  Initial loss: {initial_loss:.6f}")
            print(f"  Final loss: {final_loss:.6f}")
            print(f"  Average loss: {avg_loss:.6f}")
            print(f"  Min loss: {min_loss:.6f}")
            print(f"  Max loss: {max_loss:.6f}")
            print(f"  Loss reduction: {initial_loss - final_loss:.6f} ({((initial_loss - final_loss) / initial_loss * 100):.2f}%)")
        
        if gradient_norm_history:
            avg_grad_norm = sum(gradient_norm_history) / len(gradient_norm_history)
            print(f"\nGradient Statistics:")
            print(f"  Average gradient norm: {avg_grad_norm:.4f}")
            print(f"  Min gradient norm: {min(gradient_norm_history):.4f}")
            print(f"  Max gradient norm: {max(gradient_norm_history):.4f}")
        
        print(f"\nFinal checkpoint saved to: {output_dir / 'final'}")
        
        # Save final metrics
        final_metrics = {
            "total_steps": global_step,
            "total_training_time_seconds": total_training_time,
            "total_training_time_minutes": total_training_time / 60,
            "avg_time_per_step": total_training_time / global_step if global_step > 0 else 0,
            "avg_steps_per_second": global_step / total_training_time if total_training_time > 0 else 0,
            "loss_history": loss_history[-100:] if len(loss_history) > 100 else loss_history,  # Last 100 steps
            "gradient_norm_history": gradient_norm_history[-100:] if len(gradient_norm_history) > 100 else gradient_norm_history,
        }
        if loss_history:
            final_metrics.update({
                "initial_loss": loss_history[0],
                "final_loss": loss_history[-1],
                "average_loss": sum(loss_history) / len(loss_history),
                "min_loss": min(loss_history),
                "max_loss": max(loss_history),
            })
        
        import json
        with open(output_dir / "final" / "training_summary.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"Training summary saved to: {output_dir / 'final' / 'training_summary.json'}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
