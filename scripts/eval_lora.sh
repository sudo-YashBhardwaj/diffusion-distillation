#!/bin/bash
# Evaluation script for trained LoRA student

set -e

# Default values
LORA_PATH=""
LORA_ID=""
PROMPT="a beautiful landscape with mountains and a lake, sunset, highly detailed"
OUTDIR="./eval_outputs"
STEPS="2 4 8"
SEED=42
NUM_IMAGES=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --lora_id)
            LORA_ID="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num_images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$LORA_PATH" ] && [ -z "$LORA_ID" ]; then
    echo "ERROR: Either --lora_path or --lora_id must be provided"
    exit 1
fi

echo "=========================================="
echo "LoRA Student Evaluation"
echo "=========================================="
if [ -n "$LORA_PATH" ]; then
    echo "LoRA path: $LORA_PATH"
else
    echo "LoRA ID: $LORA_ID"
fi
echo "Prompt: $PROMPT"
echo "Output: $OUTDIR"
echo "Steps: $STEPS"
echo "Seed: $SEED"
echo "=========================================="

# Build command
CMD="python demo.py"
CMD="$CMD --prompt \"$PROMPT\""
CMD="$CMD --outdir $OUTDIR"
CMD="$CMD --steps $STEPS"
CMD="$CMD --seed $SEED"
CMD="$CMD --num_images $NUM_IMAGES"
CMD="$CMD --compare_baseline"

if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora_path $LORA_PATH"
else
    CMD="$CMD --lora_id $LORA_ID"
fi

# Run evaluation
echo "Running evaluation..."
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "Evaluation complete!"
echo "Results saved to: $OUTDIR"
