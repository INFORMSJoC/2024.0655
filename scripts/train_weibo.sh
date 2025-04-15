#!/bin/bash
 python train.py  --seed 42 --gpu 0 --name weibo_text_image_0 \
--model TextImage_weibo --dataset weibo2 \
--uni_loss 1 --loss lq2_loss --train_batch_size 16 