# Diffusion-based Visual Synthesis

Please refer to the original repo of [DiT](https://github.com/facebookresearch/DiT?tab=readme-ov-file) and [Latte](https://github.com/Vchitect/Latte) for the training and evaluation of Diffusion models. We provide the DiT [[dit.pt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/dit.pt)] and Latte [[latte_17frms.pt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/latte_17frms.pt)] checkpoints. 


## Training

To train the DiT model, please use the following command:

```
cd DiT/

torchrun --nnodes=1 --master_port 4299 --nproc_per_node=8 train.py --model DiT-XL/2 --data-path {IMAGENET_DIR} --vae "omnitokenizer" --vae-ckpt ./ckpts_pub/imagenet_ucf_vae.ckpt  --results-dir {PATH_TO_SAVE_CKPT}
```

To train the Latte model, please try:
```
cd Latte/

torchrun --nnodes=1 --master_port 4299 --nproc_per_node=8 Latte/train.py --config ./configs/ucf101/ucf101_train.yaml
```

Note that following the origin repo of Latte, you need to specify the following parameters in the config file:
- data_path: path  to UCF101 videos
- vae_type: type of the VAE, use "omnitokenizer" to try our VAE model
- pretrained_model_path: path to the VAE checkpoint, e.g., ./ckpts_pub/imagenet_ucf_vae.ckpt
- results_dir: path to save your checkpoints

## Evaluation

To evaluate your DiT model, please use:
```
torchrun --nnodes=1 --master_port 9571 --nproc_per_node=8 DiT/sample_ddp.py --model DiT-XL/2 --vae omnitokenizer --vae-ckpt ./ckpts_pub/imagenet_ucf_vae.ckpt --cfg-scale 2.0 --num-fid-samples 50000 --ckpt {PATH_TO_DIFFUSION_CKPT} --sample-dir {PATH_TO_SAVE_RESULTS}

torchrun --nnodes=1 --master_port 9572 --nproc_per_node=8 DiT/sample_ddp.py --model DiT-XL/2 --vae omnitokenizer --vae-ckpt ./ckpts_pub/imagenet_ucf_vae.ckpt --cfg-scale 1.0 --num-fid-samples 50000 --ckpt {PATH_TO_DIFFUSION_CKPT} --sample-dir {PATH_TO_SAVE_RESULTS}
```

Similarly, you can evaluate your Latte using:

```
torchrun --nnodes=1 --nproc_per_node=8 --master_port 42947 Latte/sample/sample_ddp.py \
--ckpt "./ckpts_pub/latte_17frms.pt" \
--config Latte/configs/ucf101/ucf101_sample_omnitokenizer.yaml \
--save_video_path "./sample_videos_omnitokenizer_2.0" \
--cfg_scale 2.0
```
