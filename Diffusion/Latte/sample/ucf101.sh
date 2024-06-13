#!/bin/bash

python sample/sample.py \
--config ./configs/ucf101/ucf101_sample.yaml \
--ckpt  ./share_ckpts/ucf101.pt \
--save_video_path ./test