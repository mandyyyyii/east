
# put your training data as TRAINING_DATA_PATH. Entropy exponent and coefficient are calculated based on the training data.
EXPONENT=1
COEFF=0.6

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file gpu.yaml dpo.py \
    --dataset_name $TRAINING_DATA_PATH \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --learning_rate 2e-7 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --output_dir ./model/Llama-3.2-1B-NEW-dpo-math-3epoch-lr2e-7-bs16-accuracy_non_linear2 \
    --no_remove_unused_columns \
    --report_to wandb \
    --save_strategy "no" \
    --warmup_ratio 0.1 \
    --use_weighting  \
    --use_chosen_weight_non_linear $EXPONENT \
    --use_chosen_weight_non_linear_coeff $COEFF \
    --bf16 True