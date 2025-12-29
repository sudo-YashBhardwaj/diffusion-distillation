#!/usr/bin/env python3
"""
Flash-distilled text-to-image demo using Jasper's Flash Diffusion LoRA.

This script runs Stable Diffusion v1.5 with Flash LoRA using LCM-style scheduling
for fast inference with minimal steps.
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from diffusers import LCMScheduler, StableDiffusionPipeline
from PIL import Image


def get_device_and_dtype() -> Tuple[str, torch.dtype]:
    """
    Determine the best available device and dtype.

    Returns:
        Tuple of (device_name, dtype)
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        # MPS works better with float32 for stability
        return "mps", torch.float32
    else:
        return "cpu", torch.float32


def synchronize_device(device: str) -> None:
    """
    Synchronize the given device (CUDA or MPS).

    Args:
        device: Device name ('cuda', 'mps', or 'cpu')
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    # CPU doesn't need synchronization


def load_flash_pipeline(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_id: Optional[str] = "jasperai/flash-sd",
    lora_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """
    Load Stable Diffusion pipeline with Flash LoRA adapter.

    Args:
        base_model_id: Base Stable Diffusion model identifier
        lora_id: Hugging Face LoRA adapter identifier (if lora_path is None)
        lora_path: Local path to LoRA checkpoint (takes precedence over lora_id)
        device: Device to run on ('cuda', 'mps', or 'cpu')
        dtype: Data type for model weights

    Returns:
        Configured pipeline with LoRA adapter loaded and fused
    """
    print(f"Loading base model: {base_model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Load LoRA weights using Diffusers-native method
    if lora_path:
        print(f"Loading LoRA from local path: {lora_path}")
        pipe.load_lora_weights(lora_path)
    elif lora_id:
        print(f"Loading LoRA from Hugging Face: {lora_id}")
        pipe.load_lora_weights(lora_id)
    else:
        print("Warning: No LoRA specified, using base model only")

    if lora_path or lora_id:
        pipe.fuse_lora()

    # Set scheduler with trailing timestep spacing
    print("Setting LCMScheduler with trailing timestep spacing")
    pipe.scheduler = LCMScheduler.from_pretrained(
        base_model_id, subfolder="scheduler", timestep_spacing="trailing"
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe


def load_baseline_pipeline(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """
    Load vanilla Stable Diffusion pipeline (no LoRA, default scheduler).

    Args:
        base_model_id: Base Stable Diffusion model identifier
        device: Device to run on ('cuda', 'mps', or 'cpu')
        dtype: Data type for model weights

    Returns:
        Configured baseline pipeline without LoRA
    """
    print(f"Loading baseline model: {base_model_id} (no LoRA)")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    # Use default scheduler (no LCM scheduler for baseline)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe


def generate_images(
    pipe: StableDiffusionPipeline,
    prompt: str,
    num_inference_steps: int,
    num_images: int = 1,
    seed: Optional[int] = None,
    guidance_scale: float = 0.0,
) -> List[Image.Image]:
    """
    Generate images from text prompt.

    Args:
        pipe: Configured diffusion pipeline
        prompt: Text prompt for generation
        num_inference_steps: Number of denoising steps
        num_images: Number of images to generate
        seed: Random seed for reproducibility
        guidance_scale: Guidance scale (0.0 for LCM-style, no classifier-free guidance)

    Returns:
        List of generated PIL Images
    """
    generator = None
    if seed is not None:
        # Generator device must match pipeline device
        gen_device = pipe.device
        # MPS generators may have issues, use CPU generator for MPS
        if gen_device.type == "mps":
            gen_device = torch.device("cpu")
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    # Generate all images in a single pipeline call
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_images,
    )

    return result.images


def create_grid(images: List[Image.Image], cols: int = 3) -> Image.Image:
    """
    Create a grid image from a list of images.

    Args:
        images: List of PIL Images
        cols: Number of columns in the grid

    Returns:
        Combined grid image
    """
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * w, row * h))

    return grid


def create_comparison_grid(
    baseline_images: List[Image.Image],
    flash_images: List[Image.Image],
) -> Image.Image:
    """
    Create a side-by-side comparison grid with baseline on top row and flash on bottom row.

    Args:
        baseline_images: List of baseline images
        flash_images: List of flash images

    Returns:
        Combined comparison grid image
    """
    # Determine grid dimensions
    num_cols = max(len(baseline_images), len(flash_images))
    w, h = baseline_images[0].size

    # Create grid with 2 rows
    grid = Image.new("RGB", size=(num_cols * w, 2 * h))

    # First row: baseline images
    for idx, img in enumerate(baseline_images):
        grid.paste(img, (idx * w, 0))

    # Second row: flash images
    for idx, img in enumerate(flash_images):
        grid.paste(img, (idx * w, h))

    return grid


def save_results(
    images: List[Image.Image],
    outdir: Path,
    prompt: str,
    steps: int,
    seed: Optional[int],
    prefix: str = "image",
) -> None:
    """
    Save generated images and grid to output directory.

    Args:
        images: List of generated images
        outdir: Output directory path
        prompt: Prompt used for generation
        steps: Number of inference steps used
        seed: Seed used for generation
        prefix: Filename prefix (default: "image", use "baseline" for baseline images)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Save individual images
    for idx, img in enumerate(images):
        filename = f"{prefix}_s{steps}_i{idx}"
        if seed is not None:
            filename += f"_seed{seed}"
        filename += ".png"
        img.save(outdir / filename)
        print(f"  Saved: {outdir / filename}")

    # Save grid
    grid = create_grid(images)
    grid_filename = f"grid_{prefix}_s{steps}"
    if seed is not None:
        grid_filename += f"_seed{seed}"
    grid_filename += ".png"
    grid.save(outdir / grid_filename)
    print(f"  Saved grid: {outdir / grid_filename}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Flash-distilled text-to-image generation demo"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful landscape with mountains and a lake, sunset, highly detailed",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Number of inference steps (can specify multiple, e.g., --steps 2 4 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outputs",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate per step setting",
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        help="Compare Flash LoRA against vanilla SD1.5 baseline",
    )
    parser.add_argument(
        "--baseline_steps",
        type=int,
        default=30,
        help="Number of inference steps for baseline generation",
    )
    parser.add_argument(
        "--baseline_guidance",
        type=float,
        default=7.5,
        help="Guidance scale for baseline generation",
    )
    parser.add_argument(
        "--baseline_outdir",
        type=str,
        default=None,
        help="Output directory for baseline images (default: outdir/baseline)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Local path to LoRA checkpoint (takes precedence over --lora_id)",
    )
    parser.add_argument(
        "--lora_id",
        type=str,
        default="jasperai/flash-sd",
        help="Hugging Face LoRA identifier (used if --lora_path is not provided)",
    )

    args = parser.parse_args()

    # Determine device and dtype
    device, dtype = get_device_and_dtype()

    if device == "cpu":
        print("WARNING: No GPU acceleration available. Using CPU (will be very slow).")
        print("For better performance, use:")
        print("  - NVIDIA GPU with CUDA on Linux/Windows")
        print("  - Apple Silicon (M1/M2/M3) with MPS on macOS")
        print("Continuing with CPU...")

    print(f"Using device: {device}, dtype: {dtype}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device == "mps":
        print("MPS (Metal Performance Shaders) acceleration enabled")
        print("Note: MPS may be slower than CUDA but faster than CPU")

    try:
        # Load Flash pipeline
        flash_pipe = load_flash_pipeline(
            device=device,
            dtype=dtype,
            lora_path=args.lora_path,
            lora_id=args.lora_id if not args.lora_path else None,
        )

        # Load baseline pipeline if comparison is requested
        baseline_pipe = None
        if args.compare_baseline:
            baseline_pipe = load_baseline_pipeline(device=device, dtype=dtype)

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Determine baseline output directory
        baseline_outdir = Path(args.baseline_outdir) if args.baseline_outdir else outdir / "baseline"
        baseline_outdir.mkdir(parents=True, exist_ok=True)

        # Warmup runs for accurate benchmarking
        print("\nRunning warmup generation...")
        synchronize_device(device)
        _ = generate_images(
            pipe=flash_pipe,
            prompt="warmup",
            num_inference_steps=2,
            num_images=1,
            seed=0,
            guidance_scale=0.0,
        )
        if baseline_pipe is not None:
            _ = generate_images(
                pipe=baseline_pipe,
                prompt="warmup",
                num_inference_steps=args.baseline_steps,
                num_images=1,
                seed=0,
                guidance_scale=args.baseline_guidance,
            )
        synchronize_device(device)
        print("Warmup complete.")

        # Generate baseline images if comparison is requested
        baseline_images: Optional[List[Image.Image]] = None
        baseline_time: Optional[float] = None
        if args.compare_baseline:
            print(f"\n{'='*60}")
            print(f"Generating baseline (vanilla SD1.5) with {args.baseline_steps} steps...")
            print(f"Prompt: {args.prompt}")

            synchronize_device(device)
            start_time = time.time()

            baseline_images = generate_images(
                pipe=baseline_pipe,
                prompt=args.prompt,
                num_inference_steps=args.baseline_steps,
                num_images=args.num_images,
                seed=args.seed,
                guidance_scale=args.baseline_guidance,
            )

            synchronize_device(device)
            baseline_time = time.time() - start_time

            baseline_sec_per_image = baseline_time / len(baseline_images)
            print(f"Generated {len(baseline_images)} baseline image(s) in {baseline_time:.2f}s ({baseline_sec_per_image:.2f}s/image)")

            save_results(
                baseline_images,
                baseline_outdir,
                args.prompt,
                args.baseline_steps,
                args.seed,
                prefix="baseline",
            )

        # Run inference for each step count (Flash)
        flash_results = {}
        flash_timings = {}
        for steps in args.steps:
            print(f"\n{'='*60}")
            print(f"Generating Flash LoRA with {steps} steps...")
            print(f"Prompt: {args.prompt}")

            # Accurate timing with device synchronization
            synchronize_device(device)
            start_time = time.time()

            images = generate_images(
                pipe=flash_pipe,
                prompt=args.prompt,
                num_inference_steps=steps,
                num_images=args.num_images,
                seed=args.seed,
                guidance_scale=0.0,  # LCM-style schedulers don't use classifier-free guidance
            )

            synchronize_device(device)
            elapsed = time.time() - start_time

            sec_per_image = elapsed / len(images)
            print(f"Generated {len(images)} image(s) in {elapsed:.2f}s ({sec_per_image:.2f}s/image)")

            save_results(images, outdir, args.prompt, steps, args.seed)
            flash_results[steps] = images
            flash_timings[steps] = (elapsed, sec_per_image)

            # Create comparison grid if baseline is available
            if args.compare_baseline and baseline_images is not None:
                comparison_grid = create_comparison_grid(baseline_images, images)
                comparison_filename = f"compare_baseline{args.baseline_steps}_flash{steps}"
                if args.seed is not None:
                    comparison_filename += f"_seed{args.seed}"
                comparison_filename += ".png"
                comparison_path = outdir / comparison_filename
                comparison_grid.save(comparison_path)
                print(f"  Saved comparison: {comparison_path}")

        # Print timing table
        print(f"\n{'='*60}")
        print("TIMING SUMMARY")
        print(f"{'='*60}")
        if args.compare_baseline and baseline_time is not None:
            baseline_sec_per_image = baseline_time / len(baseline_images) if baseline_images else 0
            print(f"Baseline (SD1.5 vanilla):")
            print(f"  Steps: {args.baseline_steps}")
            print(f"  Total time: {baseline_time:.2f}s")
            print(f"  Time/image: {baseline_sec_per_image:.2f}s")
            print()
        print("Flash LoRA:")
        for steps in args.steps:
            total_time, sec_per_image = flash_timings[steps]
            print(f"  Steps: {steps:2d} | Total: {total_time:6.2f}s | Time/image: {sec_per_image:5.2f}s")

        print(f"\n{'='*60}")
        print("Generation complete!")
        print(f"Results saved to: {outdir.absolute()}")
        if args.compare_baseline:
            print(f"Baseline results saved to: {baseline_outdir.absolute()}")

    except ImportError as e:
        print(f"ERROR: Missing dependency - {e}")
        print("Please install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
