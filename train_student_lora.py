#!/usr/bin/env python3
"""
Train a LoRA student model via distillation from a teacher SD1.5 model.

This script implements teacher-student distillation where:
- Teacher: Frozen SD1.5 at high steps (default 50)
- Student: SD1.5 with trainable LoRA at low steps (2/4/8)
"""

import argparse
import os
import random
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
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
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
            dataset_name: Hugging Face dataset name (e.g., 'lambdalabs/pokemon-blip-captions')
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
                image = Image.open(image).convert("RGB")
            caption = item.get("text", item.get("caption", ""))
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
        Tuple of (vae, tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler)
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

    # Load tokenizer and text encoder
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

    return vae, tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler


def compute_loss(
    vae: AutoencoderKL,
    tokenizer,
    text_encoder,
    teacher_unet: UNet2DConditionModel,
    student_unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    batch: Dict[str, torch.Tensor],
    teacher_steps: int = 50,
    student_steps: int = 4,
    guidance_scale: float = 7.5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Compute distillation loss.

    Args:
        vae: VAE encoder/decoder
        tokenizer: Text tokenizer
        text_encoder: Text encoder
        teacher_unet: Frozen teacher UNet
        student_unet: Trainable student UNet
        noise_scheduler: Noise scheduler
        batch: Batch of images and captions
        teacher_steps: Number of steps for teacher
        student_steps: Number of steps for student
        guidance_scale: Guidance scale for teacher
        device: Device
        dtype: Data type

    Returns:
        Loss tensor
    """
    images = batch["pixel_values"].to(device, dtype=dtype)
    texts = batch["text"]

    # Encode images to latents
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Encode text
    text_inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    # Sample timesteps for student (few steps)
    student_timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
    )
    student_timesteps = student_timesteps.long()

    # Sample timesteps for teacher (higher steps, more refined)
    # Use a subset of timesteps that correspond to student's range
    max_teacher_timestep = noise_scheduler.config.num_train_timesteps
    teacher_timesteps = torch.randint(
        max_teacher_timestep // 2,  # Focus on later timesteps for teacher
        max_teacher_timestep,
        (latents.shape[0],),
        device=device,
    )
    teacher_timesteps = teacher_timesteps.long()

    # Add noise to latents
    noise = torch.randn_like(latents)
    noisy_latents = noise_scheduler.add_noise(latents, noise, student_timesteps)

    # Teacher prediction (no grad)
    with torch.no_grad():
        # Use classifier-free guidance for teacher
        uncond_inputs = tokenizer(
            [""] * len(texts),
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]
        text_embeddings_guided = torch.cat([uncond_embeddings, text_embeddings])

        # Duplicate noisy latents for CFG
        noisy_latents_teacher = torch.cat([noisy_latents] * 2)
        teacher_timesteps_dup = torch.cat([teacher_timesteps] * 2)

        teacher_pred = teacher_unet(
            noisy_latents_teacher,
            teacher_timesteps_dup,
            encoder_hidden_states=text_embeddings_guided,
        ).sample

        # Apply CFG
        teacher_pred_uncond, teacher_pred_text = teacher_pred.chunk(2)
        teacher_pred = teacher_pred_uncond + guidance_scale * (teacher_pred_text - teacher_pred_uncond)

    # Student prediction (with grad)
    student_pred = student_unet(
        noisy_latents,
        student_timesteps,
        encoder_hidden_states=text_embeddings,
    ).sample

    # Distillation loss: MSE between teacher and student predictions
    loss = F.mse_loss(student_pred.float(), teacher_pred.float(), reduction="mean")

    return loss


def main() -> None:
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train LoRA student via distillation")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face dataset name (e.g., 'lambdalabs/pokemon-blip-captions')",
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
        help="Maximum training steps",
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
        default=50,
        help="Number of steps for teacher inference",
    )
    parser.add_argument(
        "--student_steps",
        type=int,
        default=4,
        help="Number of steps for student inference",
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
        # Default to pokemon dataset if nothing specified
        args.dataset_name = "lambdalabs/pokemon-blip-captions"
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

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    # Prepare models
    vae, tokenizer, text_encoder, teacher_unet, student_unet, noise_scheduler = prepare_models(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=accelerator.device,
        dtype=torch.float16,
    )

    # Setup optimizer (only LoRA parameters)
    optimizer = torch.optim.AdamW(
        student_unet.parameters(),
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

    # Training loop
    student_unet.train()
    global_step = 0
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_local_main_process)

    print(f"\nStarting training for {args.max_steps} steps...")
    print(f"Output directory: {output_dir}")

    for epoch in range(1000):  # Large number, break on max_steps
        for batch in train_dataloader:
            with accelerator.accumulate(student_unet):
                # Compute loss
                loss = compute_loss(
                    vae=vae,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    teacher_unet=teacher_unet,
                    student_unet=student_unet,
                    noise_scheduler=noise_scheduler,
                    batch=batch,
                    teacher_steps=args.teacher_steps,
                    student_steps=args.student_steps,
                    device=accelerator.device,
                    dtype=torch.float16,
                )

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

                global_step += 1

                # Save checkpoint
            if global_step % 500 == 0:
                    if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(student_unet)
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    unwrapped_model.save_pretrained(checkpoint_dir)
                    print(f"\nSaved checkpoint at step {global_step} to {checkpoint_dir}")

            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            break

    # Final save
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(student_unet)
        unwrapped_model.save_pretrained(output_dir / "final")
        print(f"\nTraining complete! Final checkpoint saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
