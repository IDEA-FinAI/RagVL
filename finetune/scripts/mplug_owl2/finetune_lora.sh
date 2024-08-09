#!/bin/bash

LOAD='MAGAer13/mplug-owl2-llama2-7b'

DATA_FILE=../../tasks/WebQA_train_rerank_owl.json
deepspeed ../../../mplug_owl2/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --visual_abstractor_lr 2e-5 \
    --deepspeed ./zero3.json \
    --model_name_or_path $LOAD \
    --version v1 \
    --data_path $DATA_FILE \
    --image_folder ../../tasks \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ../../../checkpoints/mplug-owl2-2epoch-16batch_size-webqa-reranker-caption-lora \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb