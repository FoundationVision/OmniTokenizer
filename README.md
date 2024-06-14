# OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation

Official pytorch implementation of the following paper:
<p align="left">
[OmniTokenizer: A Joint Image-Video Tokenizer for Visual Generation](https://arxiv.org/abs/2406.09399). 
<br>
<a href="https://www.wangjunke.info/">Junke Wang</a><sup>1,2</sup>, <a href="https://enjoyyi.github.io/">Yi Jiang</a><sup>3</sup>, <a href="https://shallowyuan.github.io/">Zehuan Yuan</a><sup>3</sup>, <a href="./">Binyue Peng</a><sup>3</sup>, <a href="https://zxwu.azurewebsites.net/">Zuxuan Wu</a><sup>1,2</sup>, <a href="https://fvl.fudan.edu.cn/">Yu-Gang Jiang</a><sup>1,2</sup>
<br>
<sup>1</sup>Shanghai Key Lab of Intell. Info. Processing, School of CS, Fudan University <br>
<sup>2</sup>Shanghai Collaborative Innovation Center of Intelligent Visual Computing, <sup>3</sup>Bytedance Inc.
</p>

<p align="left">
    <img src=assets/network.png width="852" height="284" />
</p>


We introduce OmniTokenizer, a joint imgae-video tokenizer which features the following properties:
- 🚀 **One model** and **one weight** for joint image and video tokenization;
- 🥇 **State-of-the-art reconstruction performance** on both image and video datasets;
- ⚡ High adaptability to **high resolution** and **long** video inputs;
- 🔥 Equipped with it, both **language model** and **diffusion model** could achieve competitive visual generation results.

Please refer to our [project page](https://www.wangjunke.info/OmniTokenizer/) for the reconstruction and generation results by OmniTokenizer.

## Setup

Please setup the environment using the following commands:

```
pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

Then download the datasets from the official websites. You can download the [annotation.zip](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/annotations.zip) processed by us and put them under ```./annotations```.

## Model Zoo for VQVAE and VAE

We release both VQVAE and VAE version of OmniTokenizer, that are pretrained on a wide range of image and video datasets:

 |  Type | Training Data  | FID | FVD | ckpt | 
 | ---------- | ---------- | ---------- | ----------- | ----------- | 
 | VQVAE | ImageNet | 1.28[^1] | - | [imagenet_only.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_only.ckpt) |
 | VQVAE | CelebAHQ | 1.85 | - | [celebahq.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/celebahq.ckpt) | 
 | VQVAE | FFHQ |2.58 | - | [ffhq.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/ffhq.ckpt) | 
 | VQVAE | ImageNet + UCF | 1.11 | 42.35 | [imagenet_ucf.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_ucf.ckpt) | 
 | VQVAE | ImageNet + K600 | 1.23 | 25.97 | [imagenet_k600.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_k600.ckpt) | 
 | VQVAE | ImageNet + MiT | 1.26 | 19.87 | [imagenet_mit.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_mit.ckpt) | 
 | VQVAE | ImageNet + Sthv2 | 1.21 | 20.30 | [imagenet_sthv2.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_sthv2.ckpt) | 
 | VQVAE | CelebAHQ + UCF | 1.93 | 45.59 | [celebahq_ucf.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/celebahq_ucf.ckpt) | 
 | VQVAE | CelebAHQ + K600 | 1.82 | 89.13 | [celebahq_k600.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/celebahq_k600.ckpt) | 
 | VQVAE | FFHQ + UCF | 1.91 | 57.93 | [ffhq_ucf.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/ffhq_ucf.ckpt) | 
 | VQVAE | FFHQ + K600 | 2.69 | 87.58 | [ffhq_k600.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/ffhq_k600.ckpt) | 
 | VAE | ImageNet + UCF | 0.69 | 23.44 | [imagenet_ucf_vae.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_ucf_vae.ckpt) | 
 | VAE | ImageNet + K600 | 0.78 | 13.02 | [imagenet_k600_vae.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_k600_vae.ckpt) |

[^1] We train this model w/o *scaled_dot_product_attention*, please comment line 446-460 in ```OmniTokenizer/modules/attention.py``` to reproduce this result.


We recommand you to try [imagenet_k600.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_k600.ckpt) as it is trained on large-scale image and video data. 

You can easily incorporate OmniTokenizer into your language model or diffusion model with:
```
from OmniTokenizer import OmniTokenizer_VQGAN
vqgan = OmniTokenizer_VQGAN.load_from_checkpoint(vqgan_ckpt, strict=False)

# tokens = vqgan.encode(img)
# recons = vqgan.decode(tokens)

```

## Tokenizer (VQVAE and VAE)

The training of VQVAE includes two stages: image-only training on a fixed resolution, and image-video joint training on multiple resolutions. After this, finetune the VQVAE model w/ KL loss to obtain a VAE model.

<p align="left">
    <img src=assets/training.png width="852" height="384" />
</p>

Please refer to ```scripts/recons/train.sh``` for the training of omnitokenizer. Explanation of the flags that are opted to change according to different settings:

- patch_size & temporal_patch_size: shape of the patches in patch embedding layer, also determine the downsample ratio
- enc_block: type of encoder blocks, 't' indices plain attention and 'w' indicates window attention
- n_codes: codebook size
- spatial_pos: type of spatial positional encoding
- use_vae: train in VAE mode or VQVAE mode
- resolution & sequence_length: spatial and temporal resolution for training
- resolution_scale: for multiple resolution training, proportion of the specificed resolution

For the evaluation of omnitokenizer, please refer to ```scripts/recons/eval_image_inet.sh```, ```scripts/recons/eval_image_face.sh```, ```scripts/recons/eval_video.sh```.


## LM-based Visual Synthesis

Please refer to ```scripts/lm_train``` and ```scripts/lm_gen``` for the training and evaluation of language model. We provide the checkpoints for ImageNet[[imagenet_class_lm.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet_class_lm.ckpt)], UCF [[ucf_class_lm.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/ucf_class_lm.ckpt)], and Kinetics-600 [[k600_fp_lm.ckpt](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/k600_fp_lm.ckpt)]. 

## Diffusion-based Visual Synthesis

We adopt [DiT](https://github.com/facebookresearch/DiT?tab=readme-ov-file) and [Latte](https://github.com/Vchitect/Latte) for diffusion-based visual generation. Please refer to [diffusion.md](Diffusion/README.md) for the training and evaluation instructions.

## Evaluation

Please refer to [evaluation.md](evaluation/README.md) for how to evaluate the reconstruction or generation results.

## Acknowledgments
Our code is partially built upon [VQGAN](https://github.com/CompVis/taming-transformers) and
[TATS](https://github.com/songweige/TATS). We also appreciate the wonderful tools provided by [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality).



## License

This project is licensed under the MIT license, as found in the LICENSE file.
