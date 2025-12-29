#!/bin/bash
# Training script for LoRA student distillation

set -e

# Default values
DATASET_NAME="diffusers/tuxemon"
DATA_DIR=""
OUTPUT_DIR="./checkpoints"
RUN_NAME="lora_student_$(date +%Y%m%d_%H%M%S)"
MAX_STEPS=1000
BATCH_SIZE=1
GRAD_ACCUM=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "LoRA Student Distillation Training"
echo "=========================================="
echo "Dataset: ${DATASET_NAME:-$DATA_DIR}"
echo "Output: $OUTPUT_DIR/$RUN_NAME"
echo "Max steps: $MAX_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "=========================================="

# Build command
CMD="python train_student_lora.py"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --run_name $RUN_NAME"
CMD="$CMD --max_steps $MAX_STEPS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --gradient_accumulation_steps $GRAD_ACCUM"

if [ -n "$DATASET_NAME" ]; then
    CMD="$CMD --dataset_name $DATASET_NAME"
fi

if [ -n "$DATA_DIR" ]; then
    CMD="$CMD --data_dir $DATA_DIR"
fi

# Run training
echo "Starting training..."
echo "Command: $CMD"
echo ""

$CMD

echo ""
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR/$RUN_NAME/lora/"
