# Diffusion Distillation for Fast Text-to-Image Generation

A research implementation of teacher-student distillation for accelerating diffusion models, enabling high-quality text-to-image generation in 2-8 steps with minimal quality degradation.

## Results

**Hardware**: NVIDIA RTX 4000 Ada 20GB  
**Evaluation**: 20 prompts × 10 images = 200 images per configuration

| Configuration | Time/Image | Speedup | CLIP Alignment | Quality Retention |
|--------------|------------|---------|----------------|-------------------|
| Baseline (30 steps) | 3.06s | 1.0× | 33.32 | 100% |
| Flash LoRA (2 steps) | 0.28s | **10.9×** | 33.14 | 99.5% |
| Flash LoRA (4 steps) | 0.37s | **8.3×** | 33.21 | 99.7% |
| Flash LoRA (8 steps) | 0.56s | **5.5×** | 33.35 | 100.1% |

**Key Findings**:
- **10.9× speedup** at 2 steps with 99.5% quality retention
- Production-ready inference speeds (<0.3s/image)
- Consistent performance across diverse prompt types

## Demo & Evaluation

The demo script (`demo.py`) provides comprehensive evaluation:

- **Fast Generation**: Flash Diffusion LoRA in 2, 4, or 8 steps
- **Baseline Comparison**: Quality and speed metrics vs. vanilla SD1.5 (30 steps)
- **Quality Metrics**: Official CLIPScore (text-image semantic similarity)
- **Automated Benchmarking**: Batch evaluation with statistical analysis

```bash
# Fast generation
python demo.py --prompt "a beautiful landscape" --steps 2 4 8

# Comprehensive benchmark
python demo.py \
  --compare_baseline \
  --prompts_file demo_prompts.txt \
  --num_images 10 \
  --steps 2 4 8 \
  --seed 42
```

## Training

Full training pipeline for distilling custom LoRA adapters from scratch:

**Method**: Teacher-student distillation aligned with Flash Diffusion / LCM-LoRA
- **Teacher**: SD1.5 with DDIM scheduler, denoises t→0 in 50 steps with CFG=7.5
- **Student**: SD1.5 + trainable LoRA (~4M parameters) at discrete few-step timesteps
- **Loss**: MSE between student and teacher z₀ (denoised latent) reconstructions
- **Training**: Samples from discrete student schedule, teacher provides multi-step targets

```bash
# Train on HuggingFace dataset
python train_student_lora.py \
  --dataset_name "diffusers/tuxemon" \
  --output_dir ./checkpoints \
  --run_name "my_lora" \
  --max_steps 2000 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4

# Evaluate trained LoRA
python demo.py \
  --lora_path ./checkpoints/my_lora/lora/final \
  --compare_baseline \
  --prompt "your prompt here"
```

**Implementation Details**:
- `DDPMScheduler` for noise addition and alpha consistency
- `LCMScheduler` for student inference with trailing timestep spacing
- Multi-step teacher denoising with proper timestep sequencing
- Only LoRA parameters trainable (base model frozen)
- FP16 training with Accelerate, gradient accumulation for memory efficiency

## Technical Details

- **Base Model**: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
- **LoRA**: Rank-4 on UNet attention layers (to_k, to_q, to_v, to_out.0)
- **Scheduler**: LCMScheduler with trailing timestep spacing
- **Inference**: No CFG (guidance_scale=0) for LCM-style generation
- **Evaluation**: Official CLIPScore formula (100 × max(0, cosine_similarity))

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
