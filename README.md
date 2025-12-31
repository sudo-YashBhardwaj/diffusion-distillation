# Diffusion Distillation for Fast Text-to-Image Generation

A research implementation of teacher-student distillation for accelerating diffusion models, enabling fast text-to-image generation in 2-8 steps with significant speedup.

## Results

**Hardware**: NVIDIA RTX 4000 Ada 20GB  
**Evaluation**: 20 prompts × 10 images = 200 images per configuration  
**Baseline**: Stable Diffusion v1.5 (50 steps, CFG=7.5)

| Configuration | Time/Image | Speedup | CLIP Alignment | Quality Retention |
|--------------|------------|---------|----------------|-------------------|
| Baseline (50 steps) | 2.17s | 1.0× | 33.42 | 100% |
| Flash LoRA (2 steps) | 0.12s | **18.1×** | 21.96 | 65.7% |
| Flash LoRA (4 steps) | 0.16s | **13.6×** | 21.21 | 63.5% |
| Flash LoRA (8 steps) | 0.25s | **8.7×** | 20.81 | 62.3% |

**Key Findings**:
- **18× speedup** at 2 steps with 66% quality retention
- Sub-second inference (<0.3s/image) suitable for real-time applications
- Consistent performance across 20 diverse prompts

## Demo & Evaluation

Comprehensive evaluation framework with baseline comparison and quality metrics:

```bash
# Fast generation
python demo.py --prompt "a beautiful landscape" --steps 2 4 8

# Benchmark with baseline comparison
python demo.py \
  --compare_baseline \
  --prompts_file demo_prompts.txt \
  --num_images 10 \
  --steps 2 4 8 \
  --seed 42
```

**Features**:
- Official CLIPScore evaluation (text-image semantic similarity)
- CUDA-synchronized timing for accurate benchmarks
- Side-by-side comparison grids
- Automated CSV/JSON reporting

## Training

K-step distillation aligned with Flash Diffusion / LCM-LoRA:

**Method**:
- **Teacher**: Frozen SD1.5, performs K DDIM steps from timestep t with CFG (guidance_scale=7.5)
- **Student**: SD1.5 + trainable LoRA (~4M parameters), learns to match teacher's K-step output in one forward pass
- **Loss**: MSE between student's single-step prediction and teacher's multi-step target
- **Key Insight**: Student learns to skip multiple denoising steps at once

```bash
# Train on local dataset (COCO)
python train_flash_lora.py \
  --data_dir /path/to/coco \
  --output_dir ./checkpoints \
  --run_name "coco_lora_flash" \
  --max_steps 3000 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --num_ddim_steps 8 \
  --num_inference_steps 4

# Use trained LoRA
python demo.py \
  --lora_path ./checkpoints/coco_lora_flash/lora/final \
  --prompt "your prompt here" \
  --steps 4
```

**Implementation**:
- Teacher performs K DDIM steps with CFG to produce target latents
- Student predicts target in single forward pass (no CFG)
- Discrete timestep sampling from LCM inference schedule
- FP16 training with Accelerate, gradient accumulation, checkpoint resume
- Only LoRA parameters trainable (base model frozen)
- Trained on COCO dataset (118K image-caption pairs) for 3000 steps

## Technical Details

- **Base Model**: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
- **LoRA**: Rank-4 on UNet attention layers (to_k, to_q, to_v, to_out.0)
- **Scheduler**: LCMScheduler with trailing timestep spacing
- **Inference**: Low guidance scale (default=1.0) for LCM-style generation
- **Evaluation**: Official CLIPScore (100 × max(0, cosine_similarity))

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (tested on RTX 4000 Ada 20GB)
- PyTorch with CUDA support

## License

Uses Stable Diffusion v1.5 (CreativeML Open RAIL-M License) and Flash LoRA adapter.
