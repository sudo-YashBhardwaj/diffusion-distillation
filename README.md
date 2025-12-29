# Flash-Distilled Text-to-Image Demo & Training

A minimal, reproducible demo project for fast text-to-image generation using Jasper's Flash Diffusion LoRA on top of Stable Diffusion v1.5. **This project now includes a minimal distillation training script that trains a LoRA student from an SD1.5 teacher.**

This project demonstrates:
- How to use the Flash LoRA adapter with an LCM-style scheduler to generate high-quality images in just 2-8 inference steps
- How to train your own LoRA student model via teacher-student distillation

## Features

- **Fast inference**: Generate images in 2, 4, or 8 steps using LCM-style scheduling
- **Flash LoRA**: Uses Jasper's Flash Diffusion LoRA for optimized generation
- **Stable Diffusion v1.5**: Built on the reliable `runwayml/stable-diffusion-v1-5` base model
- **FP16 precision**: Optimized for NVIDIA GPUs with automatic mixed precision
- **Baseline comparison**: Compare Flash LoRA against vanilla SD1.5 side-by-side
- **Training support**: Train your own LoRA student via distillation from a teacher model

## Requirements

- **Hardware**: NVIDIA GPU with CUDA support (tested on RTX 4000 Ada 20GB, single GPU)
- **Software**: Python 3.8+, CUDA-capable PyTorch, Linux

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch with CUDA support

**Important**: PyTorch must be installed separately as a CUDA-enabled wheel. Install the appropriate version for your CUDA setup:

```bash
# For CUDA 12.1 (example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check your CUDA version with `nvidia-smi` and install the matching PyTorch build.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `diffusers` - Hugging Face Diffusers library
- `transformers` - Transformer models
- `accelerate` - Model acceleration utilities
- `safetensors` - Safe tensor serialization
- `peft` - Parameter-Efficient Fine-Tuning (for LoRA)
- `pillow` - Image processing
- `datasets` - Hugging Face datasets library

### 4. Authenticate with Hugging Face (if required)

Stable Diffusion v1.5 may require accepting the model license on Hugging Face. If you encounter authentication errors:

1. Create a Hugging Face account at https://huggingface.co
2. Accept the license for `runwayml/stable-diffusion-v1-5` at https://huggingface.co/runwayml/stable-diffusion-v1-5
3. Login using the CLI:

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted.

## Usage

### Basic usage

Generate images with default settings (2, 4, and 8 steps):

```bash
python demo.py --prompt "a beautiful landscape with mountains and a lake, sunset, highly detailed"
```

### Custom prompt

```bash
python demo.py --prompt "a futuristic city at night, neon lights, cyberpunk style"
```

### Specific step counts

```bash
python demo.py --prompt "a cat wearing sunglasses" --steps 4 8
```

### With seed for reproducibility

```bash
python demo.py --prompt "a serene forest path" --seed 42
```

### Generate multiple images

```bash
python demo.py --prompt "a majestic eagle in flight" --num_images 3
```

### Custom output directory

```bash
python demo.py --prompt "underwater coral reef" --outdir ./my_results
```

### Using a local LoRA checkpoint

```bash
python demo.py --lora_path ./checkpoints/lora_student/final --prompt "a beautiful sunset"
```

### All options

```bash
python demo.py \
  --prompt "a steampunk airship floating above clouds" \
  --steps 2 4 8 \
  --seed 12345 \
  --outdir ./outputs \
  --num_images 2
```

## Baseline Comparison

Compare Flash LoRA against vanilla Stable Diffusion v1.5 to see the speed and quality differences side-by-side.

### Basic comparison

```bash
python demo.py --compare_baseline --prompt "a beautiful sunset over mountains"
```

This generates:
- Baseline images using vanilla SD1.5 (30 steps, guidance_scale=7.5) - saved to `outdir/baseline/`
- Flash LoRA images (2, 4, 8 steps) - saved to `outdir/`
- Comparison grids showing baseline (top row) vs Flash (bottom row) for each step count

### Custom baseline settings

```bash
python demo.py \
  --compare_baseline \
  --baseline_steps 30 \
  --baseline_guidance 7.5 \
  --prompt "a futuristic cityscape at night"
```

### Custom baseline output directory

```bash
python demo.py \
  --compare_baseline \
  --baseline_outdir ./baseline_results \
  --prompt "an astronaut floating in space"
```

The baseline comparison uses vanilla Stable Diffusion v1.5 (no LoRA) with standard classifier-free guidance. This provides a fair comparison to demonstrate the speed improvements of Flash LoRA while maintaining quality.

## Training Your Own LoRA Student

This project includes a minimal distillation training script that trains a LoRA student model to generate images in few steps (2/4/8) by distilling knowledge from a teacher SD1.5 model running at higher steps.

### Quick Start: Training a LoRA Student

1. **Run a quick distillation training** (1000 steps, ~30-60 minutes on RTX 4000 Ada):

```bash
./scripts/train_lora.sh \
  --dataset_name "lambdalabs/pokemon-blip-captions" \
  --max_steps 1000 \
  --batch_size 1 \
  --grad_accum 4
```

Or using Python directly:

```bash
python train_student_lora.py \
  --dataset_name "lambdalabs/pokemon-blip-captions" \
  --output_dir ./checkpoints \
  --run_name "my_lora_student" \
  --max_steps 1000 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4
```

2. **Evaluate your trained LoRA**:

```bash
./scripts/eval_lora.sh \
  --lora_path ./checkpoints/my_lora_student/lora/final \
  --prompt "a pokemon character" \
  --steps "2 4 8"
```

### Training Details

The training script implements teacher-student distillation:

1. **Teacher**: Frozen SD1.5 model at high steps (default 50 steps with CFG)
2. **Student**: SD1.5 with trainable LoRA on UNet, trained to match teacher predictions at low steps (2/4/8)
3. **Loss**: MSE between teacher and student noise predictions
4. **Optimization**: Only LoRA parameters are trained (base model frozen)

### Training Configuration

Key training parameters:

- **Dataset**: Hugging Face dataset (e.g., `lambdalabs/pokemon-blip-captions`) or local folder with images/captions
- **Resolution**: 512x512 (default)
- **Batch size**: 1-2 (for 20GB GPU)
- **Gradient accumulation**: 4-8 steps (effective batch size = batch_size × grad_accum)
- **Learning rate**: 1e-4 (default)
- **Max steps**: 1000-2000 for quick runs, more for better quality
- **LoRA rank**: 4 (default), increase for more capacity
- **LoRA alpha**: 32 (default)

### Training with Local Data

If you have a local folder with images and captions:

```bash
python train_student_lora.py \
  --data_dir ./my_dataset \
  --output_dir ./checkpoints \
  --run_name "my_lora" \
  --max_steps 2000
```

Expected structure:
- `my_dataset/images/*.png` and `my_dataset/captions/*.txt`, OR
- `my_dataset/*.png` with matching `*.txt` files

### Training Script Options

```bash
python train_student_lora.py --help
```

Key arguments:
- `--dataset_name`: Hugging Face dataset name
- `--data_dir`: Local directory with images/captions
- `--output_dir`: Output directory for checkpoints (default: `./checkpoints`)
- `--run_name`: Run name for checkpoint directory
- `--max_steps`: Maximum training steps (default: 2000)
- `--batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--lora_rank`: LoRA rank (default: 4)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--resolution`: Image resolution (default: 512)

### Evaluation Script

The evaluation script runs baseline comparison with your trained LoRA:

```bash
./scripts/eval_lora.sh \
  --lora_path ./checkpoints/my_lora_student/lora/final \
  --prompt "your prompt here" \
  --steps "2 4 8" \
  --seed 42
```

This will:
- Generate baseline images (30 steps, CFG 7.5)
- Generate student LoRA images (2, 4, 8 steps, guidance=0 for LCM-style)
- Create comparison grids
- Print timing table

## Command-line Arguments

### demo.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | str | (see default in code) | Text prompt for image generation |
| `--steps` | int[] | `[2, 4, 8]` | Number of inference steps (can specify multiple) |
| `--seed` | int | `None` | Random seed for reproducibility |
| `--outdir` | str | `./outputs` | Output directory for generated images |
| `--num_images` | int | `1` | Number of images to generate per step setting |
| `--compare_baseline` | flag | `False` | Compare Flash LoRA against vanilla SD1.5 |
| `--baseline_steps` | int | `30` | Number of inference steps for baseline generation |
| `--baseline_guidance` | float | `7.5` | Guidance scale for baseline generation |
| `--baseline_outdir` | str | `None` | Output directory for baseline images (default: outdir/baseline) |
| `--lora_path` | str | `None` | Local path to LoRA checkpoint (takes precedence over --lora_id) |
| `--lora_id` | str | `jasperai/flash-sd` | Hugging Face LoRA identifier (used if --lora_path is not provided) |

## Output Files

The script generates the following files in the output directory:

### Individual Images
- `image_s{steps}_i{idx}_seed{seed}.png` - Individual generated images
  - `s{steps}`: Number of inference steps used
  - `i{idx}`: Image index (0, 1, 2, ...)
  - `seed{seed}`: Random seed (if specified)

### Grid Images
- `grid_s{steps}_seed{seed}.png` - Grid combining all images for a given step count

### Baseline Comparison (when `--compare_baseline` is used)
- `baseline_s{steps}_i{idx}_seed{seed}.png` - Individual baseline images (in `baseline/` subdirectory)
- `grid_baseline_s{steps}_seed{seed}.png` - Grid of baseline images
- `compare_baseline{baseline_steps}_flash{flash_steps}_seed{seed}.png` - Side-by-side comparison grid (baseline top row, Flash bottom row)

### Example Output Structure

```
outputs/
├── image_s2_i0_seed42.png
├── image_s4_i0_seed42.png
├── image_s8_i0_seed42.png
├── grid_s2_seed42.png
├── grid_s4_seed42.png
├── grid_s8_seed42.png
├── compare_baseline30_flash2_seed42.png
├── compare_baseline30_flash4_seed42.png
├── compare_baseline30_flash8_seed42.png
└── baseline/
    ├── baseline_s30_i0_seed42.png
    └── grid_baseline_s30_seed42.png
```

### Training Checkpoints

Training saves checkpoints to:
```
checkpoints/
└── {run_name}/
    └── lora/
        ├── checkpoint-500/
        ├── checkpoint-1000/
        └── final/
```

## Technical Details

### Why `guidance_scale=0`?

The LCM (Latent Consistency Model) scheduler used in this demo is designed to work without classifier-free guidance. Setting `guidance_scale=0` disables the guidance mechanism, which is appropriate for LCM-style schedulers that achieve fast convergence through their specialized timestep spacing.

### Model Loading

The demo loads:
1. **Base model**: `runwayml/stable-diffusion-v1-5` - The standard Stable Diffusion v1.5 checkpoint
2. **LoRA adapter**: `jasperai/flash-sd` (default) or your trained LoRA - Jasper's Flash Diffusion LoRA weights (loaded using Diffusers-native `load_lora_weights()` and fused with `fuse_lora()`)
3. **Scheduler**: `LCMScheduler` with `timestep_spacing="trailing"` for optimal fast inference

### Baseline Comparison

When `--compare_baseline` is enabled:
- **Baseline pipeline**: Vanilla SD1.5 (no LoRA) with default scheduler and standard classifier-free guidance (guidance_scale=7.5)
- **Flash pipeline**: SD1.5 with Flash LoRA, LCM scheduler, and guidance_scale=0.0
- Both pipelines use the same base model and are timed with CUDA synchronization for accurate benchmarking

### Training Method

The training script implements a minimal teacher-student distillation loop:

1. **Teacher**: Frozen SD1.5 UNet at high steps (50 steps) with classifier-free guidance
2. **Student**: SD1.5 UNet with trainable LoRA adapters at low steps (2/4/8)
3. **Process**:
   - Encode images to latents with VAE
   - Sample timesteps
   - Add noise to latents
   - Compute teacher prediction (no grad, with CFG)
   - Compute student prediction (with grad, no CFG for LCM-style)
   - Distillation loss = MSE(student_pred, teacher_pred)
4. **Optimization**: Only LoRA parameters are updated

The training uses:
- FP16 mixed precision (via Accelerate)
- Gradient accumulation for effective larger batch sizes
- Single GPU training (designed for 20GB VRAM)

### Performance

On an NVIDIA RTX 4000 Ada 20GB:
- **Flash 2 steps**: ~0.5-1.0 seconds per image
- **Flash 4 steps**: ~1.0-1.5 seconds per image
- **Flash 8 steps**: ~1.5-2.5 seconds per image
- **Baseline 30 steps**: ~3-5 seconds per image

Actual timing depends on prompt complexity and system configuration. The script includes accurate CUDA-synchronized benchmarking with warmup runs and prints a timing summary table at the end.

## Troubleshooting

### CUDA not available

If you see "CUDA is not available":
1. Verify you have an NVIDIA GPU: `nvidia-smi`
2. Install PyTorch with CUDA: See step 2 in Setup
3. Check CUDA drivers are installed and up to date

### Missing dependencies

If you see import errors:
```bash
pip install -r requirements.txt
```

### Authentication errors

If you see authentication errors when loading models:
1. Accept the model license on Hugging Face
2. Run `huggingface-cli login` and enter your token

### Out of memory during training

If you encounter CUDA out of memory errors during training:
- Reduce `--batch_size` to 1
- Increase `--gradient_accumulation_steps` to maintain effective batch size
- Reduce `--resolution` to 256 (lower quality but faster)
- Close other GPU-intensive applications

### Out of memory during inference

If you encounter CUDA out of memory errors during inference:
- Reduce `--num_images` to 1
- Close other GPU-intensive applications
- Consider using a smaller batch size (modify code if needed)

## License

This demo project uses:
- Stable Diffusion v1.5 (CreativeML Open RAIL-M License)
- Flash LoRA adapter (check Jasper AI's license terms)

## References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PEFT Library](https://github.com/huggingface/peft)
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [Flash Diffusion LoRA](https://huggingface.co/jasperai/flash-sd)
