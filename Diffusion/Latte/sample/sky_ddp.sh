#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=2 sample/sample_ddp.py \
--config ./configs/sky/sky_sample.yaml \
--ckpt ./share_ckpts/skytimelapse.pt \
--save_video_path ./test
