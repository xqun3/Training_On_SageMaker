#!/bin/bash
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker

MODEL="/tmp/pretrain_model"

accelerate launch \
    --config_file ac_config.yaml \
    src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL \
    --dataset evol_instruct_code_12k,glaive_toolcall \
    --dataset_dir data \
    --template deepseekcoder \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,o_proj,k_proj \
    --output_dir /tmp/finetuned_model \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --save_on_each_node \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --fp16 \
    --load_best_model_at_end False \
    --report_to wandb
    # --flash_attn True \
if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi