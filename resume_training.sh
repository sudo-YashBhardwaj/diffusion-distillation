#!/bin/bash
# Resume training from checkpoint-1000

python train_student_lora.py \
  --data_dir /Data/yash.bhardwaj/coco \
  --output_dir ./checkpoints \
  --run_name coco_lora_fast \
  --max_steps 3000 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 64 \
  --resume_from_checkpoint ./checkpoints/coco_lora_fast/lora/checkpoint-1000 \
  2>&1 | tee -a training_fast.log
