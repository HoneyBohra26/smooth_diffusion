#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port 20688 train_smooth_diffusion.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --dataset_folder "C:/Users/bohra/Smooth-Diffusion" \
    --regularization_annotation "C:/Users/bohra/Smooth-Diffusion/regularization_images.jsonl" \
    --resolution 512 \
    --train_batch_size 3 \
    --num_train_epochs 99 \
    --checkpointing_steps 5000 \
    --learning_rate 1e-4 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --seed 42 \
    --output_dir output \
    --report_to tensorboard \
    --resume_from_checkpoint latest \
    --dataloader_num_workers 0 \
    --max_train_steps 30000 \
    --lambda_reg 1 \
    --rank 8 \
    --gradient_accumulation_steps 8
