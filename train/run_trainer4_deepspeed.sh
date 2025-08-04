#!/bin/bash

# Run trainer4 with DeepSpeed Zero compatibility
# This script demonstrates how to launch training with multiple backward passes

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

# Model and data configuration
MODEL_NAME="microsoft/DialoGPT-small"  # Small model for testing
DATASET_NAME="Anthropic/hh-rlhf"
OUTPUT_DIR="./output_trainer4_deepspeed"

# Training configuration
BATCH_SIZE=4
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
MAX_STEPS=100
LEARNING_RATE=5e-6

# Multiple backward configuration
MULTIPLE_BACKWARD=true
SECONDARY_LOSS_WEIGHT=0.1
L2_REG_WEIGHT=0.001

echo "Starting trainer4 with DeepSpeed Zero compatibility..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Multiple backward passes: $MULTIPLE_BACKWARD"
echo "Secondary loss weight: $SECONDARY_LOSS_WEIGHT"

# Launch with DeepSpeed
deepspeed --num_gpus=4 train_with_multiple_backward.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --deepspeed deepspeed_config_trainer4.json \
    --multiple_backward_passes $MULTIPLE_BACKWARD \
    --secondary_loss_weight $SECONDARY_LOSS_WEIGHT \
    --l2_reg_weight $L2_REG_WEIGHT \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --fp16 \
    --logging_steps 10 \
    --save_steps 50 \
    --eval_steps 50 \
    --warmup_steps 10 \
    --logging_dir $OUTPUT_DIR/logs \
    --report_to none \
    --remove_unused_columns false \
    --dataloader_num_workers 2

echo "Training completed!"