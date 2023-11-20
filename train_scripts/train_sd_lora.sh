#!/bin/bash
accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py --input_perturbation 0.1 \
                               --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
                               --train_data_dir /mnt/data/minhnq54/zaic-2023-banner/train/images \
                               --image_column image \
                               --caption_column text \
                               --resolution 512 \
                               --train_batch_size 1 \
                               --snr_gamma 5.0 \
                               --checkpoints_total_limit 1 \
                               --dataloader_num_workers 4 \
                               --checkpointing_steps 3000
