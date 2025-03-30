# start the training process in terminal(can not set the lora params)
# mlx_lm.lora --model mlx-community/Qwen2.5-7B-Instruct-4bit \
# --train \
# --batch-size 2 \
# --num-layers 28 \
# --max_seq_length 2048 \
# --learning-rate 1e-5 \
# --iters 200 \
# --steps_per_report 10 \
# --steps_per_eval 10 \
# --save_every 100 \
# --adapter_path "resources/model/output/mlx/adapters" \
# --data "resources/data/mlx_train_data" \

# start the training process and set the config in .yaml file(can set the lora params and other params)
mlx_lm.lora -c lora_config.yaml