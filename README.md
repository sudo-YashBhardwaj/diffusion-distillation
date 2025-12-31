# Diffusion Distillation for Fast Text-to-Image (SD1.5 → LoRA Student)

A research implementation of **teacher–student distillation** to accelerate **Stable Diffusion v1.5** for **2–8 step** text-to-image generation using a **parameter-efficient LoRA** (~4M trainable parameters).  

This repo includes a reproducible training pipeline (COCO), robust checkpoint/resume, and a benchmarking harness (timing + CLIP alignment + grids).

**Core idea:** a frozen SD1.5 teacher performs multi-step denoising (DDIM + CFG); a LoRA-equipped student learns to match the teacher's multi-step target in a single forward pass, effectively "skipping" denoising steps.

---

## Highlights

- **LoRA distillation training** for SD1.5 (rank-4, alpha-32) targeting UNet attention projections: `to_q/to_k/to_v/to_out.0`

- **Single-GPU training** tested on **RTX 4000 Ada 20GB**, mixed precision FP16 (Accelerate)

- **Benchmark harness**: CUDA-synchronized timing, CLIP alignment, side-by-side grids, CSV/JSON export

- **Reproducible evaluation** on **20 prompts × 10 images** per configuration

---

## Results

### A) Reproducing Jasper Flash LoRA (using this repo's benchmark harness)

**Hardware**: NVIDIA RTX 4000 Ada 20GB  

**Evaluation**: 20 prompts × 10 images = 200 images per configuration  

**Baseline**: SD1.5 (50 steps, CFG=7.5)  

**Metric**: CLIP alignment = `100 × max(0, cosine_similarity(text, image))`

| Configuration | Time/Image | Speedup vs Baseline | CLIP Alignment | Quality Retention |
|--------------|------------|---------------------|----------------|-------------------|
| Baseline (50 steps) | 2.17s | 1.0× | 33.42 | 100% |
| Flash LoRA (2 steps) | 0.12s | **18.1×** | 21.96 | 65.7% |
| Flash LoRA (4 steps) | 0.16s | **13.6×** | 21.21 | 63.5% |
| Flash LoRA (8 steps) | 0.25s | **8.7×** | 20.81 | 62.3% |

> Note: The table above corresponds to the **official Jasper Flash LoRA** benchmark run in this repo's evaluation harness. These numbers reproduce Jasper’s released adapter; our trained adapter is evaluated with the same harness.

> For results of **our trained LoRA**, run training + benchmarking below and inspect the generated `results.csv`.

**Key takeaways**

- **Sub-second inference** (<0.3s/image) with 2–8 steps

- **Up to 18× speedup** vs a 50-step SD1.5 baseline

- Quality trade-offs are measurable (CLIP alignment + qualitative grids)

---

## Quickstart

### 1) Install

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Demo & Evaluation

The evaluation framework (`demo.py`) supports:

* Fast few-step generation (2/4/8 steps)

* Baseline comparison (vanilla SD1.5, 30–50+ steps, CFG)

* CLIP alignment computation (CLIP ViT-B/32)

* CUDA-synchronized timing

* Side-by-side comparison grids

* CSV/JSON reporting

### A) Fast generation

```bash
python demo.py --prompt "a beautiful landscape" --steps 2 4 8
```

### B) Benchmark with baseline comparison (recommended)

```bash
python demo.py \
  --compare_baseline \
  --prompts_file demo_prompts.txt \
  --num_images 10 \
  --steps 2 4 8 \
  --seed 42
```

Outputs (under `--outdir`, default `./outputs`):

* Per-prompt images + grids

* Baseline images (if enabled)

* Side-by-side comparison grids

* `clipscore_results.json` (structured results)

---

## Training (LoRA Student via K-step Distillation)

### Method

* **Teacher:** Frozen SD1.5 UNet, performs **K DDIM denoising steps** starting from timestep `t` using **classifier-free guidance** (`guidance_scale=7.5`)

* **Student:** SD1.5 UNet with **trainable LoRA adapters** (base weights frozen), predicts the teacher's K-step target in **one forward pass** (**no CFG**)

* **Loss:** MSE between student prediction (mapped to the same latent space) and teacher target latent

* **Timestep sampling:** from a discrete schedule aligned with few-step inference (LCM-style), to reduce train/infer mismatch

### Dataset

This repo is set up to train on a local dataset directory (e.g., **COCO**), but the dataset loader also supports HF datasets for quick iteration.

Recommended:

* **COCO** (118K image–caption pairs) for general-purpose text-to-image behavior

* Use a smaller dataset only for smoke tests / debugging

### Train on local COCO

```bash
python train_flash_lora.py \
  --data_dir /path/to/coco \
  --output_dir ./checkpoints \
  --run_name "coco_lora_flash" \
  --max_steps 3000 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --num_ddim_steps 8 \
  --num_inference_steps 4
```

This writes LoRA checkpoints under:

```
./checkpoints/coco_lora_flash/lora/
  checkpoint-500/
  checkpoint-1000/
  ...
  final/
```

### Use your trained LoRA for inference

```bash
python demo.py \
  --lora_path ./checkpoints/coco_lora_flash/lora/final \
  --prompt "your prompt here" \
  --steps 2 4 8 \
  --seed 42 \
  --outdir ./eval_outputs \
  --compare_baseline
```

---

## Technical Details

* **Base model:** `runwayml/stable-diffusion-v1-5`

* **LoRA:** rank-4, alpha-32 on UNet attention projections (`to_k`, `to_q`, `to_v`, `to_out.0`)

* **Teacher:** frozen UNet + CFG (`guidance_scale=7.5`), multi-step DDIM supervision

* **Student inference:** `LCMScheduler(timestep_spacing="trailing")` for few-step generation

* **Precision:** FP16 (Accelerate), gradient accumulation, checkpoint resume

* **Evaluation metric:** CLIP alignment = `100 × max(0, cosine_similarity(text, image))` using CLIP ViT-B/32

* **Hardware tested:** NVIDIA RTX 4000 Ada 20GB

---

## Repository Layout

```
diffusion-distillation/
├── train_flash_lora.py        # Distillation training (teacher → LoRA student)
├── demo.py                    # Inference + benchmarking + CLIP alignment + grids
├── demo_prompts.txt           # 20 evaluation prompts
├── requirements.txt
├── README.md
├── checkpoints/               # Trained LoRA weights (generated)
└── outputs/                   # Evaluation outputs (generated)
```

---

## Requirements

* Python 3.8+

* CUDA-capable GPU strongly recommended (tested on RTX 4000 Ada 20GB)

* PyTorch with CUDA support

---

## License

Uses Stable Diffusion v1.5 (CreativeML Open RAIL-M License).

This repo can optionally load external LoRA adapters (e.g., Jasper Flash LoRA) for reproduction/benchmarking.
