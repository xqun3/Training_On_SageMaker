#!/bin/bash

MODEL="/tmp/pretrain_model"

DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS --nnodes $NODE_NUMBER --node_rank $NODE_INDEX --master_addr $SM_MASTER_ADDR --master_port 12345"

torchrun ${DISTRIBUTED_ARGS} \
    src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL \
    --dataset evol_instruct_code_12k \
    --dataset_dir data \
    --template deepseek \
    --finetuning_type full \
    --output_dir /tmp/finetuned_model \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 128 \
    --save_on_each_node \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_steps 100 \
    --load_best_model_at_end \
    --learning_rate 1.0e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --fp16 False \
    --plot_loss \
    --num_train_epochs 1.0 \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --report_to wandb
    # --flash_attn True \
if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi
