#!/bin/bash

# Initialize variables
LEARNING_RATE="2e-4"
NUM_TRAIN_EPOCHS="3"
CONCURRENCY_THREADS="2"
DATA_SYNTHESIS_MODE="low"
HALF=False

# Process parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --lr) LEARNING_RATE="$2"; shift ;;
        --epochs) NUM_TRAIN_EPOCHS="$2"; shift ;;
        --threads) CONCURRENCY_THREADS="$2"; shift ;;
        --mode) DATA_SYNTHESIS_MODE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Log the parameters being used
echo "Using training parameters:"
echo "  Learning rate: $LEARNING_RATE"
echo "  Number of epochs: $NUM_TRAIN_EPOCHS"
echo "  Concurrency threads: $CONCURRENCY_THREADS"
echo "  Data synthesis mode: $DATA_SYNTHESIS_MODE"

# If concurrency threads are set, configure related environment variables
if [ "$CONCURRENCY_THREADS" != "1" ]; then
  export OMP_NUM_THREADS=$CONCURRENCY_THREADS
  export MKL_NUM_THREADS=$CONCURRENCY_THREADS
  export NUMEXPR_NUM_THREADS=$CONCURRENCY_THREADS
  echo "Set thread environment variables to $CONCURRENCY_THREADS"
fi

# Add BF16 option based on the platform
if [ "$PLATFORM" != "apple" ]; then
  HALF=True
fi

# Execute training script with parameters from environment variables
python lpm_kernel/L2/train.py \
  --seed 42 \
  --model_name_or_path "${MODEL_BASE_PATH}" \
  --user_name "${USER_NAME}" \
  --dataset_name "resources/L2/data/merged.json" \
  --chat_template_format "chatml" \
  --add_special_tokens False \
  --append_concat_token False \
  --max_seq_length 512 \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --save_total_limit 2 \
  --logging_steps 20 \
  --log_level "info" \
  --logging_strategy "steps" \
  --save_strategy "steps" \
  --save_steps 5 \
  --push_to_hub False \
  --bf16 $HALF \
  --packing False \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler_type "cosine" \
  --weight_decay 1e-4 \
  --max_grad_norm 0.3 \
  --output_dir "${MODEL_PERSONAL_DIR}" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps $CONCURRENCY_THREADS \
  --gradient_checkpointing True \
  --use_reentrant True \
  --use_peft_lora True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules "all-linear" \
  --use_4bit_quantization False \
  --use_nested_quant False \
  --bnb_4bit_compute_dtype "bfloat16" \
  --is_cot False

