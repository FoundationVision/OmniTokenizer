#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=2 sample/sample_ddp.py \
--config ./configs/taichi/taichi_sample.yaml \
--ckpt ./share_ckpts/taichi-hd.pt \
--save_video_path ./test \
