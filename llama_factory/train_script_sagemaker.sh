#!/bin/bash

DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS --nnodes $NODE_NUMBER --node_rank $NODE_INDEX --master_addr $SM_MASTER_ADDR --master_port 12345"
MODEL="/tmp/pretrain_model"

torchrun ${DISTRIBUTED_ARGS} \
    src/train_bash.py \
    --deepspeed ds_config.json \
    --stage pt \
    --do_train \
    --model_name_or_path $MODEL \
    --dataset evol_instruct_code_12k \
    --dataset_dir data \
    --template deepseekcoder \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --packing False \
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
    --quantization_bit 8 \
    --report_to wandb
    # --flash_attn True \
if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi