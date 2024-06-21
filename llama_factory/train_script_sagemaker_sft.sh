#!/bin/bash

MODEL="/tmp/pretrain_model"

accelerate launch \
    --config_file ac_config.yaml \
    src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL \
    --dataset evol_instruct_code_12k \
    --dataset_dir data \
    --template llama3 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir /tmp/finetuned_model \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --save_on_each_node \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_steps 100 \
    --load_best_model_at_end \
    --learning_rate 1.0e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --fp16 \
    --plot_loss \
    --num_train_epochs 1.0 \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --ddp_timeout 180000000 \
    --load_best_model_at_end False \
    --quantization_bit 8 \
    --report_to wandb
    # --flash_attn True \
if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi