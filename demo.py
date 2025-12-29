#!/usr/bin/env python3
"""
Fast text-to-image generation with Flash Diffusion LoRA.

Generates images in 2-8 steps using Stable Diffusion v1.5 with Flash LoRA.
Includes baseline comparison and CLIP alignment metrics.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import LCMScheduler, StableDiffusionPipeline
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def get_device_and_dtype() -> Tuple[str, torch.dtype]:
    """Auto-detect best device (CUDA > MPS > CPU) and appropriate dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        # MPS works better with float32 for stability
        return "mps", torch.float32
    else:
        return "cpu", torch.float32


def synchronize_device(device: str) -> None:
    """Synchronize CUDA or MPS device for accurate timing."""
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
    """Load SD1.5 pipeline with Flash LoRA and LCM scheduler."""
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

    # Switch to LCM scheduler for fast inference
    print("Setting LCMScheduler from config")
    scheduler_config = pipe.scheduler.config.copy()
    try:
        scheduler_config["timestep_spacing"] = "trailing"
        pipe.scheduler = LCMScheduler.from_config(scheduler_config)
    except (TypeError, KeyError):
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    return pipe


def load_baseline_pipeline(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """Load vanilla SD1.5 pipeline (no LoRA) for baseline comparison."""
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
    """Generate images from text prompt."""
    generator = None
    if seed is not None:
        gen_device = pipe.device
        if gen_device.type == "mps":
            gen_device = torch.device("cpu")  # MPS generators can be problematic
        generator = torch.Generator(device=gen_device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_images,
    )

    return result.images


def create_grid(images: List[Image.Image], cols: int = 3) -> Image.Image:
    """Create a grid image from a list of images."""
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * w, row * h))

    return grid


def load_clip_model(device: str = "cuda") -> Tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP model for text-image alignment scoring."""
    print("Loading CLIP model for clip alignment calculation...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def calculate_clip_align(
    images: List[Image.Image],
    prompt: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str = "cuda",
) -> List[float]:
    """
    Calculate text-image alignment scores (0-100).
    
    Uses batched processing and vectorized cosine similarity for efficiency.
    """
    with torch.no_grad():
        # Process text once
        text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)
        
        # Batch process all images
        image_inputs = clip_processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_features = clip_model.get_image_features(**image_inputs)
        image_features = F.normalize(image_features, dim=-1)
        
        # Vectorized cosine similarity
        cosine_sims = (image_features @ text_features.T).squeeze(-1)
        scores = (100.0 * torch.clamp(cosine_sims, min=0.0)).cpu().tolist()
    
    return scores


def load_prompts_from_file(prompts_file: str) -> List[str]:
    """Load prompts from text file (one per line)."""
    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    return prompts


def create_comparison_grid(
    baseline_images: List[Image.Image],
    flash_images: List[Image.Image],
) -> Image.Image:
    """Create side-by-side comparison grid (baseline top, flash bottom)."""
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
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to text file with prompts (one per line). If provided, overrides --prompt",
    )
    parser.add_argument(
        "--flash_guidance",
        type=float,
        default=1.0,
        help="Guidance scale for Flash generations",
    )

    args = parser.parse_args()
    
    # Load prompts from file or use single prompt
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = [args.prompt]

    # Determine device and dtype
    device, dtype = get_device_and_dtype()

    if device == "cpu":
        print("Warning: No GPU detected. CPU inference will be very slow.")

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

        # Warmup for accurate timing
        print("\nRunning warmup...")
        synchronize_device(device)
        _ = generate_images(flash_pipe, "warmup", 2, 1, 0, args.flash_guidance)
        if baseline_pipe:
            _ = generate_images(baseline_pipe, "warmup", args.baseline_steps, 1, 0, args.baseline_guidance)
        synchronize_device(device)
        print("Warmup complete.\n")

        # Load CLIP model for clip alignment calculation
        clip_model, clip_processor = load_clip_model(device=device)
        
        # Store results for all prompts
        all_results: List[Dict] = []

        # Process each prompt
        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"PROCESSING PROMPT {prompt_idx + 1}/{len(prompts)}")
            print(f"{'='*60}")
            print(f"Prompt: {prompt}")
            
            prompt_results = {
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "flash_scores": {},
                "baseline_scores": None,
                "flash_timings": {},
                "baseline_timing": None,
            }

            # Generate baseline images if comparison is requested
            baseline_images: Optional[List[Image.Image]] = None
            baseline_time: Optional[float] = None
            baseline_scores: Optional[List[float]] = None
            if args.compare_baseline:
                print(f"\n{'='*60}")
                print(f"Generating baseline (vanilla SD1.5) with {args.baseline_steps} steps...")
                print(f"Prompt: {prompt}")

                synchronize_device(device)
                start_time = time.time()

                baseline_images = generate_images(
                    pipe=baseline_pipe,
                    prompt=prompt,
                    num_inference_steps=args.baseline_steps,
                    num_images=args.num_images,
                    seed=args.seed,
                    guidance_scale=args.baseline_guidance,
                )

                synchronize_device(device)
                baseline_time = time.time() - start_time

                baseline_sec_per_image = baseline_time / len(baseline_images)
                print(f"Generated {len(baseline_images)} baseline image(s) in {baseline_time:.2f}s ({baseline_sec_per_image:.2f}s/image)")
                
                prompt_results["baseline_timing"] = {
                    "total_time": baseline_time,
                    "time_per_image": baseline_sec_per_image,
                    "steps": args.baseline_steps,
                }

                # Create prompt-specific baseline subdirectory
                prompt_baseline_outdir = baseline_outdir / f"prompt_{prompt_idx:02d}"
                prompt_baseline_outdir.mkdir(parents=True, exist_ok=True)
                
                save_results(
                    baseline_images,
                    prompt_baseline_outdir,
                    prompt,
                    args.baseline_steps,
                    args.seed,
                    prefix="baseline",
                )
                
                # Calculate clip alignment for baseline images
                baseline_scores = calculate_clip_align(
                    baseline_images,
                    prompt,
                    clip_model,
                    clip_processor,
                    device=device,
                )
                avg_baseline_score = sum(baseline_scores) / len(baseline_scores)
                std_baseline_score = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0.0
                print(f"  Baseline clip_align_x100: {baseline_scores} (avg: {avg_baseline_score:.4f}, std: {std_baseline_score:.4f})")
                prompt_results["baseline_scores"] = {
                    "individual": baseline_scores,
                    "average": avg_baseline_score,
                    "std": std_baseline_score,
                }

            # Run inference for each step count (Flash)
            flash_results = {}
            flash_timings = {}
            flash_scores_by_steps: Dict[int, Dict] = {}
            for steps in args.steps:
                print(f"\n{'='*60}")
                print(f"Generating Flash LoRA with {steps} steps...")
                print(f"Prompt: {prompt}")

                # Accurate timing with device synchronization
                synchronize_device(device)
                start_time = time.time()

                images = generate_images(
                    pipe=flash_pipe,
                    prompt=prompt,
                    num_inference_steps=steps,
                    num_images=args.num_images,
                    seed=args.seed,
                    guidance_scale=args.flash_guidance,
                )

                synchronize_device(device)
                elapsed = time.time() - start_time

                sec_per_image = elapsed / len(images)
                print(f"Generated {len(images)} image(s) in {elapsed:.2f}s ({sec_per_image:.2f}s/image)")

                # Create prompt-specific subdirectory
                prompt_outdir = outdir / f"prompt_{prompt_idx:02d}"
                prompt_outdir.mkdir(parents=True, exist_ok=True)
                
                save_results(images, prompt_outdir, prompt, steps, args.seed)
                flash_results[steps] = images
                flash_timings[steps] = (elapsed, sec_per_image)
                
                # Calculate clip alignment for Flash LoRA images
                flash_scores = calculate_clip_align(
                    images,
                    prompt,
                    clip_model,
                    clip_processor,
                    device=device,
                )
                avg_flash_score = sum(flash_scores) / len(flash_scores)
                std_flash_score = statistics.stdev(flash_scores) if len(flash_scores) > 1 else 0.0
                print(f"  Flash LoRA clip_align_x100: {flash_scores} (avg: {avg_flash_score:.4f}, std: {std_flash_score:.4f})")
                flash_scores_by_steps[steps] = {
                    "individual": flash_scores,
                    "average": avg_flash_score,
                    "std": std_flash_score,
                }

                # Create comparison grid if baseline is available
                if args.compare_baseline and baseline_images is not None:
                    comparison_grid = create_comparison_grid(baseline_images, images)
                    comparison_filename = f"compare_baseline{args.baseline_steps}_flash{steps}"
                    if args.seed is not None:
                        comparison_filename += f"_seed{args.seed}"
                    comparison_filename += ".png"
                    comparison_path = prompt_outdir / comparison_filename
                    comparison_grid.save(comparison_path)
                    print(f"  Saved comparison: {comparison_path}")

            prompt_results["flash_scores"] = flash_scores_by_steps
            prompt_results["flash_timings"] = flash_timings
            all_results.append(prompt_results)

        # Print summary for all prompts
        print(f"\n{'='*60}")
        print("CLIP ALIGNMENT SUMMARY")
        print(f"{'='*60}")
        
        for result in all_results:
            print(f"\nPrompt {result['prompt_idx'] + 1}: {result['prompt'][:60]}...")
            print("  Flash LoRA:")
            for steps in args.steps:
                if steps in result["flash_scores"]:
                    avg_score = result["flash_scores"][steps]["average"]
                    print(f"    Steps {steps:2d}: {avg_score:.4f}")
            if result["baseline_scores"]:
                print(f"  Baseline: {result['baseline_scores']['average']:.4f}")
        
        # Calculate overall averages
        print(f"\n{'='*60}")
        print("OVERALL AVERAGE CLIP ALIGNMENT")
        print(f"{'='*60}")
        
        overall_flash_avg = {}
        for steps in args.steps:
            scores = []
            for result in all_results:
                if steps in result["flash_scores"]:
                    scores.append(result["flash_scores"][steps]["average"])
            if scores:
                overall_flash_avg[steps] = sum(scores) / len(scores)
                print(f"Flash LoRA ({steps} steps): {overall_flash_avg[steps]:.4f}")
        
        if args.compare_baseline:
            baseline_scores = []
            for result in all_results:
                if result["baseline_scores"]:
                    baseline_scores.append(result["baseline_scores"]["average"])
            if baseline_scores:
                overall_baseline_avg = sum(baseline_scores) / len(baseline_scores)
                print(f"Baseline: {overall_baseline_avg:.4f}")
        
        # Prepare CSV data
        csv_rows = []
        baseline_stats_by_prompt = {}
        
        for result in all_results:
            prompt_idx = result["prompt_idx"]
            
            # Add baseline row if it exists
            if result["baseline_scores"] and result["baseline_timing"]:
                baseline_stats_by_prompt[prompt_idx] = {
                    "avg_clip_align_x100": result["baseline_scores"]["average"],
                    "std_clip_align_x100": result["baseline_scores"]["std"],
                    "time_total_s": result["baseline_timing"]["total_time"],
                    "time_per_image_s": result["baseline_timing"]["time_per_image"],
                }
                csv_rows.append({
                    "prompt_idx": prompt_idx,
                    "steps": result["baseline_timing"]["steps"],
                    "is_baseline": True,
                    "avg_clip_align_x100": result["baseline_scores"]["average"],
                    "std_clip_align_x100": result["baseline_scores"]["std"],
                    "time_total_s": result["baseline_timing"]["total_time"],
                    "time_per_image_s": result["baseline_timing"]["time_per_image"],
                    "speedup_vs_baseline": None,
                    "delta_clip_vs_baseline": None,
                })
            
            # Add flash rows
            for steps in args.steps:
                if steps in result["flash_scores"] and steps in result["flash_timings"]:
                    total_time, time_per_image = result["flash_timings"][steps]
                    row = {
                        "prompt_idx": prompt_idx,
                        "steps": steps,
                        "is_baseline": False,
                        "avg_clip_align_x100": result["flash_scores"][steps]["average"],
                        "std_clip_align_x100": result["flash_scores"][steps]["std"],
                        "time_total_s": total_time,
                        "time_per_image_s": time_per_image,
                    }
                    
                    # Add baseline comparison if baseline exists
                    if prompt_idx in baseline_stats_by_prompt:
                        baseline_time_per_image = baseline_stats_by_prompt[prompt_idx]["time_per_image_s"]
                        baseline_avg_clip = baseline_stats_by_prompt[prompt_idx]["avg_clip_align_x100"]
                        row["speedup_vs_baseline"] = baseline_time_per_image / time_per_image
                        row["delta_clip_vs_baseline"] = result["flash_scores"][steps]["average"] - baseline_avg_clip
                    else:
                        row["speedup_vs_baseline"] = None
                        row["delta_clip_vs_baseline"] = None
                    
                    csv_rows.append(row)
        
        # Write CSV file
        csv_path = outdir / "results.csv"
        if csv_rows:
            fieldnames = [
                "prompt_idx", "steps", "is_baseline", "avg_clip_align_x100", "std_clip_align_x100",
                "time_total_s", "time_per_image_s", "speedup_vs_baseline", "delta_clip_vs_baseline"
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in csv_rows:
                    # Convert None to empty string for CSV
                    csv_row = {k: ("" if v is None else v) for k, v in row.items()}
                    writer.writerow(csv_row)
            print(f"\nResults CSV saved to: {csv_path}")
        
        # Save results to JSON (ensure all values are JSON-serializable)
        results_json_path = outdir / "clip_align_results.json"
        # Convert any non-serializable objects to basic types
        json_results = []
        for result in all_results:
            json_result = {
                "prompt": result["prompt"],
                "prompt_idx": result["prompt_idx"],
                "flash_scores": {},
                "baseline_scores": None,
                "flash_timings": {},
                "baseline_timing": None,
            }
            # Convert flash scores
            for steps, score_data in result["flash_scores"].items():
                json_result["flash_scores"][str(steps)] = {
                    "individual": [float(s) for s in score_data["individual"]],
                    "average": float(score_data["average"]),
                    "std": float(score_data["std"]),
                }
            # Convert flash timings
            for steps, timing_data in result["flash_timings"].items():
                json_result["flash_timings"][str(steps)] = {
                    "total_time": float(timing_data[0]),
                    "time_per_image": float(timing_data[1]),
                }
            # Convert baseline scores
            if result["baseline_scores"]:
                json_result["baseline_scores"] = {
                    "individual": [float(s) for s in result["baseline_scores"]["individual"]],
                    "average": float(result["baseline_scores"]["average"]),
                    "std": float(result["baseline_scores"]["std"]),
                }
            # Convert baseline timing
            if result["baseline_timing"]:
                json_result["baseline_timing"] = {
                    "total_time": float(result["baseline_timing"]["total_time"]),
                    "time_per_image": float(result["baseline_timing"]["time_per_image"]),
                    "steps": int(result["baseline_timing"]["steps"]),
                }
            json_results.append(json_result)
        
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed clip alignment results saved to: {results_json_path}")
        
        # Print timing table
        print(f"\n{'='*60}")
        print("TIMING SUMMARY")
        print(f"{'='*60}")
        print(f"Processed {len(prompts)} prompt(s)\n")
        
        # Per-prompt timing summary
        for result in all_results:
            print(f"Prompt {result['prompt_idx'] + 1}: {result['prompt'][:50]}...")
            if result["baseline_timing"]:
                bt = result["baseline_timing"]
                print(f"  Baseline: {bt['total_time']:.2f}s total, {bt['time_per_image']:.2f}s/image ({bt['steps']} steps)")
            print("  Flash LoRA:")
            for steps in args.steps:
                if steps in result["flash_timings"]:
                    total_time, sec_per_image = result["flash_timings"][steps]
                    print(f"    Steps {steps:2d}: {total_time:6.2f}s total, {sec_per_image:5.2f}s/image")
            print()
        
        # Overall average timing
        print(f"{'='*60}")
        print("OVERALL AVERAGE TIMING")
        print(f"{'='*60}")
        
        if args.compare_baseline:
            baseline_times = []
            baseline_times_per_image = []
            for result in all_results:
                if result["baseline_timing"]:
                    baseline_times.append(result["baseline_timing"]["total_time"])
                    baseline_times_per_image.append(result["baseline_timing"]["time_per_image"])
            if baseline_times:
                avg_baseline_total = sum(baseline_times) / len(baseline_times)
                avg_baseline_per_image = sum(baseline_times_per_image) / len(baseline_times_per_image)
                print(f"Baseline (SD1.5 vanilla):")
                print(f"  Average total time: {avg_baseline_total:.2f}s")
                print(f"  Average time/image: {avg_baseline_per_image:.2f}s")
                print()
        
        print("Flash LoRA:")
        for steps in args.steps:
            total_times = []
            per_image_times = []
            for result in all_results:
                if steps in result["flash_timings"]:
                    total_time, sec_per_image = result["flash_timings"][steps]
                    total_times.append(total_time)
                    per_image_times.append(sec_per_image)
            if total_times:
                avg_total = sum(total_times) / len(total_times)
                avg_per_image = sum(per_image_times) / len(per_image_times)
                print(f"  Steps {steps:2d}: {avg_total:6.2f}s avg total, {avg_per_image:5.2f}s avg/image")

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
