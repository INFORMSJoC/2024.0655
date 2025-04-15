#!/bin/bash
 python train.py  --seed 42 --gpu 0 --name fakeddit_text_image_0 \
--model TextImage --dataset fakeddit \
--uni_loss 1 --loss lq2_loss --train_batch_size 16 --learning_rate 5e-5 --gradient_accumulation_steps 1
