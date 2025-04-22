#!/bin/bash
 python train.py  --seed 42 --gpu 0 --name twitter_mmdy_0 \
--model MMDy \
--uni_loss 1 --loss lq2_loss 