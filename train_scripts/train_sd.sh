#!/bin/bash
accelerate launch --mixed_precision="fp16" train_text_to_image.py --input_perturbation 0.1 \
                               --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
                               --train_data_dir /mnt/data/minhnq54/zaic-2023-banner/train/images \
                               --image_column image \
                               --caption_column text \
                               --resolution 512 \
                               --center_crop \
                               --random_flip \
                               --num_train_epochs 50 \
                               --train_batch_size 2 \
                               --snr_gamma 5.0 \
                               --dataloader_num_workers 4 \
                               --output_dir /mnt/data/minhnq54/document-vision-models-and-logs/weights/zaic/sd \
                               --checkpoints_total_limit 1
