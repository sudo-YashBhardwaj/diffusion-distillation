#!/usr/bin/env python3
"""
Train a LoRA student model via distillation from a teacher SD1.5 model.

This script implements few-step distillation aligned with Flash Diffusion / LCM-LoRA:
- Teacher: Frozen SD1.5 UNet that provides CFG-guided predictions
- Student: SD1.5 UNet with trainable LoRA adapters
- Training: Student learns to match teacher's CFG-guided output (noise matching)
- Loss: MSE between student prediction and teacher CFG prediction with Min-SNR weighting
"""

import argparse
import os
import random
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
    LCMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model, PeftModel
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


def compute_snr(noise_scheduler: DDPMScheduler, timesteps: torch.Tensor) -> torch.Tensor:
    """
    Compute Signal-to-Noise Ratio for given timesteps.
    
    SNR = alpha^2 / sigma^2 = alpha_cumprod / (1 - alpha_cumprod)
    
    Args:
        noise_scheduler: The noise scheduler containing alphas_cumprod
        timesteps: Tensor of timesteps
    
    Returns:
        SNR values for each timestep
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    
    alpha = sqrt_alphas_cumprod[timesteps]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    
    snr = (alpha / sigma) ** 2
    return snr


def get_discrete_timesteps(num_train_timesteps: int = 1000, num_inference_steps: int = 4) -> List[int]:
    """
    Get discrete timesteps that match LCM inference schedule.
    
    For 4-step inference with 1000 timesteps: [875, 625, 375, 125]
    
    Args:
        num_train_timesteps: Total number of training timesteps (usually 1000)
        num_inference_steps: Number of inference steps (e.g., 4)
    
    Returns:
        List of discrete timesteps
    """
    step_ratio = num_train_timesteps // num_inference_steps
    timesteps = [(num_inference_steps - 1 - i) * step_ratio + step_ratio // 2 for i in range(num_inference_steps)]
    return timesteps


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
    snr_gamma: float = 5.0,
    use_discrete_timesteps: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute distillation loss using noise-matching with Min-SNR weighting.
    
    This approach is more numerically stable than z0 reconstruction:
    - Teacher: Produces CFG-guided noise prediction
    - Student: Learns to match teacher's CFG output in a single forward pass
    - Loss: MSE with Min-SNR weighting for stability at high timesteps

    Args:
        vae: VAE encoder/decoder
        tokenizer: Text tokenizer
        text_encoder: Text encoder
        teacher_unet: Frozen teacher UNet
        student_unet: Trainable student UNet
        noise_scheduler: DDPMScheduler for adding noise and alphas
        teacher_scheduler: DDIMScheduler (unused in noise-matching, kept for compatibility)
        batch: Batch of images and captions
        teacher_steps: Unused (kept for compatibility)
        student_steps: Number of inference steps (for discrete timestep sampling)
        guidance_scale: Guidance scale for teacher CFG
        snr_gamma: Min-SNR gamma for loss weighting (higher = more uniform weighting)
        use_discrete_timesteps: If True, sample from discrete timesteps matching inference
        device: Device
        dtype: Data type

    Returns:
        Tuple of (loss, metrics_dict)
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

    # Sample timesteps
    if use_discrete_timesteps:
        # Sample from discrete timesteps matching LCM inference schedule
        discrete_timesteps = get_discrete_timesteps(
            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
            num_inference_steps=student_steps,
        )
        timestep_indices = torch.randint(0, len(discrete_timesteps), (batch_size,))
        timesteps = torch.tensor([discrete_timesteps[i] for i in timestep_indices], device=device, dtype=torch.long)
    else:
        # Uniform sampling (original behavior)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        )

    # Add noise to latents using noise_scheduler (DDPMScheduler)
    noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Teacher: Single forward pass with CFG
    with torch.no_grad():
        # Prepare inputs for CFG (concatenate unconditional and conditional)
        latent_model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
        timestep_input = torch.cat([timesteps, timesteps], dim=0)
        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0)
        
        # Teacher prediction
        teacher_noise_pred = teacher_unet(
            latent_model_input,
            timestep_input,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        # Apply CFG
        teacher_noise_pred_uncond, teacher_noise_pred_cond = teacher_noise_pred.chunk(2)
        teacher_noise_pred_cfg = teacher_noise_pred_uncond + guidance_scale * (
            teacher_noise_pred_cond - teacher_noise_pred_uncond
        )

    # Student: Single forward pass (no CFG - learns CFG-distilled behavior)
    student_noise_pred = student_unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_embeddings,
    ).sample

    # Compute Min-SNR weighted loss
    # This prevents high-timestep samples from dominating the loss
    snr = compute_snr(noise_scheduler, timesteps)
    
    # Min-SNR weighting: min(SNR, gamma) / SNR
    # At high timesteps (low SNR): weight ≈ 1
    # At low timesteps (high SNR): weight = gamma / SNR (downweighted)
    mse_loss_weights = torch.clamp(snr, max=snr_gamma) / snr
    mse_loss_weights = mse_loss_weights.to(device=device, dtype=dtype)
    
    # Compute per-sample MSE loss
    loss_per_sample = F.mse_loss(
        student_noise_pred.float(), 
        teacher_noise_pred_cfg.float(), 
        reduction="none"
    )
    loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3])  # Mean over spatial dims
    
    # Apply Min-SNR weighting
    loss = (loss_per_sample * mse_loss_weights).mean()
    
    # Compute metrics for logging
    metrics = {
        "loss": loss.item(),
        "loss_unweighted": loss_per_sample.mean().item(),
        "snr_mean": snr.mean().item(),
        "snr_min": snr.min().item(),
        "snr_max": snr.max().item(),
        "timestep_mean": timesteps.float().mean().item(),
        "student_pred_std": student_noise_pred.std().item(),
        "teacher_pred_std": teacher_noise_pred_cfg.std().item(),
    }
    
    # Verify loss is valid (not NaN or Inf)
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: Invalid loss detected! Loss = {loss.item()}")
        print(f"  student_noise_pred stats: min={student_noise_pred.min().item():.4f}, max={student_noise_pred.max().item():.4f}")
        print(f"  teacher_noise_pred_cfg stats: min={teacher_noise_pred_cfg.min().item():.4f}, max={teacher_noise_pred_cfg.max().item():.4f}")
        print(f"  Timesteps used: {timesteps.cpu().tolist()}")
        print(f"  SNR values: {snr.cpu().tolist()}")

    return loss, metrics


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
        help="Number of steps for teacher denoising (unused in noise-matching mode)",
    )
    parser.add_argument(
        "--student_steps",
        type=int,
        default=4,
        help="Number of steps for student inference (used for discrete timestep sampling)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for teacher classifier-free guidance",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="Min-SNR gamma for loss weighting (higher = more uniform, 5.0 recommended)",
    )
    parser.add_argument(
        "--use_discrete_timesteps",
        action="store_true",
        default=True,
        help="Use discrete timesteps matching inference schedule (recommended)",
    )
    parser.add_argument(
        "--use_uniform_timesteps",
        action="store_true",
        help="Use uniform timestep sampling instead of discrete (not recommended)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 to disable)",
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
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (e.g., './checkpoints/coco_lora_fast/lora/checkpoint-1000')",
    )

    args = parser.parse_args()
    
    # Handle timestep sampling flags
    if args.use_uniform_timesteps:
        args.use_discrete_timesteps = False

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

    # Resume from checkpoint if specified - need to handle this before preparing models
    resume_step = 0
    resume_checkpoint_path = None
    if args.resume_from_checkpoint:
        resume_checkpoint_path = Path(args.resume_from_checkpoint)
        if not resume_checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {resume_checkpoint_path}")
        
        # Load training metrics to get the step number
        metrics_path = resume_checkpoint_path / "training_metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            resume_step = metrics.get("global_step", 0)
        else:
            # Try to extract step number from checkpoint directory name
            checkpoint_name = resume_checkpoint_path.name
            if checkpoint_name.startswith("checkpoint-"):
                try:
                    resume_step = int(checkpoint_name.split("-")[1])
                except ValueError:
                    pass

    # Prepare models
    # If resuming, we'll load the PEFT model from checkpoint instead of creating a new one
    if args.resume_from_checkpoint:
        # Load base model components first
        from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTokenizer, CLIPTextModel
        
        base_model_id = "runwayml/stable-diffusion-v1-5"
        print(f"Loading base model: {base_model_id}")
        
        vae = AutoencoderKL.from_pretrained(
            base_model_id,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
        
        teacher_unet = UNet2DConditionModel.from_pretrained(
            base_model_id,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        teacher_unet.requires_grad_(False)
        teacher_unet.eval()
        
        try:
            teacher_unet = torch.compile(teacher_unet, mode="reduce-overhead")
            print("✓ Teacher UNet compiled with torch.compile")
        except Exception as e:
            print(f"Note: torch.compile not available ({e}), using standard mode")
        
        # Load base UNet (without PEFT)
        base_unet = UNet2DConditionModel.from_pretrained(
            base_model_id,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        
        # Load PEFT model directly from checkpoint
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")
        print(f"Loading checkpoint from: {resume_checkpoint_path}")
        
        # Load PEFT model from checkpoint
        student_unet = PeftModel.from_pretrained(base_unet, resume_checkpoint_path)
        
        # Enable training mode
        student_unet.train()
        
        # Disable inference mode in PEFT config to enable training
        if hasattr(student_unet, 'peft_config') and student_unet.peft_config:
            for adapter_name, config in student_unet.peft_config.items():
                config.inference_mode = False
        
        # CRITICAL: Manually enable requires_grad for all LoRA parameters
        # This ensures they are trainable even if loaded in inference mode
        trainable_count = 0
        for name, param in student_unet.named_parameters():
            # LoRA parameters typically have 'lora' in their name
            if 'lora' in name.lower():
                param.requires_grad = True
                trainable_count += 1
        
        student_unet.print_trainable_parameters()
        print(f"✓ Loaded LoRA adapter weights from checkpoint")
        print(f"✓ Manually enabled training for {trainable_count} LoRA parameters")
        print(f"✓ Resuming from step {resume_step}")
        
        metrics_path = resume_checkpoint_path / "training_metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            print(f"  Last checkpoint loss: {metrics.get('loss', 'N/A'):.6f}")
            print(f"  Last checkpoint avg loss: {metrics.get('avg_loss', 'N/A'):.6f}")
        
        tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_id, subfolder="text_encoder", torch_dtype=torch.float16
        )
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        
        noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        teacher_scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        
        print(f"{'='*60}\n")
    else:
        # Normal path: prepare models with new PEFT
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
        trainable_params_count = sum(p.numel() for p in student_unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in student_unet.parameters())
        print(f"\n[Model Status]")
        print(f"  Student UNet in training mode: {student_unet.training}")
        print(f"  Trainable parameters: {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}% of total)")
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
        
        # Training method info
        print(f"\n[Training Method]")
        print(f"  Loss type: Noise-matching with Min-SNR weighting")
        print(f"  SNR gamma: {args.snr_gamma}")
        print(f"  Guidance scale: {args.guidance_scale}")
        print(f"  Discrete timesteps: {args.use_discrete_timesteps}")
        if args.use_discrete_timesteps:
            discrete_ts = get_discrete_timesteps(
                noise_scheduler.config.num_train_timesteps, 
                args.student_steps
            )
            print(f"  Timesteps for {args.student_steps}-step inference: {discrete_ts}")
        print(f"  Max gradient norm: {args.max_grad_norm}")
        
        # Warnings
        warnings = []
        if teacher_trainable > 0:
            warnings.append(f"  ⚠️  Teacher UNet has {teacher_trainable} trainable parameters (should be frozen)!")
        if text_encoder_trainable > 0:
            warnings.append(f"  ⚠️  Text encoder has {text_encoder_trainable} trainable parameters (should be frozen)!")
        if trainable_params_count == 0:
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
                test_loss, test_metrics = compute_loss(
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
                    snr_gamma=args.snr_gamma,
                    use_discrete_timesteps=args.use_discrete_timesteps,
                    device=accelerator.device,
                    dtype=torch.float16,
                )
            
            print(f"  ✅ Test forward pass successful!")
            print(f"  Test loss (no grad): {test_loss.item():.6f}")
            print(f"  Test metrics: SNR mean={test_metrics['snr_mean']:.2f}, timestep mean={test_metrics['timestep_mean']:.0f}")
            
            # Test backward pass
            print(f"  Testing backward pass...")
            student_unet.train()
            optimizer.zero_grad()
            test_loss_grad, _ = compute_loss(
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
                snr_gamma=args.snr_gamma,
                use_discrete_timesteps=args.use_discrete_timesteps,
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
    global_step = resume_step
    progress_bar = tqdm(range(args.max_steps), initial=resume_step, disable=not accelerator.is_local_main_process)

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    if resume_step > 0:
        print(f"Resuming training from step {resume_step} to {args.max_steps} steps...")
        print(f"Remaining steps: {args.max_steps - resume_step}")
    else:
        print(f"Starting training for {args.max_steps} steps...")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}, LoRA alpha: {args.lora_alpha}")
    print(f"SNR gamma: {args.snr_gamma}")
    print(f"Max gradient norm: {args.max_grad_norm}")
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
                loss, metrics = compute_loss(
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
                    snr_gamma=args.snr_gamma,
                    use_discrete_timesteps=args.use_discrete_timesteps,
                    device=accelerator.device,
                    dtype=torch.float16,
                )

                # Backward pass
                accelerator.backward(loss)
                
                # Calculate gradient norm before optimizer step
                grad_norm = 0.0
                if accelerator.sync_gradients:
                    # Gradient clipping
                    if args.max_grad_norm > 0:
                        grad_norm = accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                        if hasattr(grad_norm, 'item'):
                            grad_norm = grad_norm.item()
                    else:
                        # Just compute gradient norm for logging
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
                    loss_value = metrics["loss"]
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
                        steps_per_sec = (global_step - resume_step) / elapsed_time if elapsed_time > 0 else 0.0
                        remaining_steps = args.max_steps - global_step
                        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0
                        eta_minutes = eta_seconds / 60.0
                        
                        print(f"\n[Step {global_step}/{args.max_steps}]")
                        print(f"  Loss: {loss_value:.6f} (avg over last {log_interval}: {avg_loss:.6f})")
                        print(f"  Gradient norm: {grad_norm:.4f} (avg: {avg_grad_norm:.4f})")
                        print(f"  SNR: mean={metrics['snr_mean']:.2f}, min={metrics['snr_min']:.4f}, max={metrics['snr_max']:.2f}")
                        print(f"  Timestep mean: {metrics['timestep_mean']:.0f}")
                        print(f"  Pred std: student={metrics['student_pred_std']:.4f}, teacher={metrics['teacher_pred_std']:.4f}")
                        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
                        print(f"  Speed: {avg_step_time:.3f}s/step, {images_per_sec:.2f} images/sec, {steps_per_sec:.2f} steps/sec")
                        print(f"  ETA: {eta_minutes:.1f} minutes ({eta_seconds:.0f} seconds)")
                        
                        if torch.cuda.is_available():
                            mem_allocated = torch.cuda.memory_allocated(accelerator.device) / 1e9
                            mem_reserved = torch.cuda.memory_reserved(accelerator.device) / 1e9
                            print(f"  GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
                        
                        # Check if model parameters are updating
                        if global_step == log_interval + resume_step:
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
                        checkpoint_metrics = {
                            "global_step": global_step,
                            "loss": loss_value,
                            "avg_loss": sum(loss_history[-100:]) / min(100, len(loss_history)) if loss_history else 0.0,
                            "grad_norm": grad_norm,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "epoch": epoch + 1,
                            "snr_gamma": args.snr_gamma,
                            "guidance_scale": args.guidance_scale,
                        }
                        import json
                        with open(checkpoint_dir / "training_metrics.json", "w") as f:
                            json.dump(checkpoint_metrics, f, indent=2)
                        
                        print(f"\n[Step {global_step}] Saved checkpoint to {checkpoint_dir}")
                        print(f"  Current loss: {loss_value:.6f}")
                        print(f"  Average loss (last 100 steps): {checkpoint_metrics['avg_loss']:.6f}")
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
        print(f"Average time per step: {total_training_time / (global_step - resume_step):.3f} seconds")
        print(f"Average steps per second: {(global_step - resume_step) / total_training_time:.2f}")
        
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
            loss_change = initial_loss - final_loss
            loss_change_pct = (loss_change / initial_loss * 100) if initial_loss != 0 else 0
            print(f"  Loss change: {loss_change:.6f} ({loss_change_pct:.2f}%)")
            
            # Check if training was successful
            if loss_change > 0:
                print(f"  ✅ Loss decreased during training (good!)")
            else:
                print(f"  ⚠️  Loss increased during training (check hyperparameters)")
        
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
            "avg_time_per_step": total_training_time / (global_step - resume_step) if (global_step - resume_step) > 0 else 0,
            "avg_steps_per_second": (global_step - resume_step) / total_training_time if total_training_time > 0 else 0,
            "loss_history": loss_history[-100:] if len(loss_history) > 100 else loss_history,  # Last 100 steps
            "gradient_norm_history": gradient_norm_history[-100:] if len(gradient_norm_history) > 100 else gradient_norm_history,
            "training_config": {
                "snr_gamma": args.snr_gamma,
                "guidance_scale": args.guidance_scale,
                "use_discrete_timesteps": args.use_discrete_timesteps,
                "student_steps": args.student_steps,
                "max_grad_norm": args.max_grad_norm,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
            }
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