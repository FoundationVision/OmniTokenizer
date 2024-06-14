# Evaluation Protocols

## Visual Reconstruction

For the evaluation of visual reconstruction performance, we report the reconstruction FID / FVD on the whole validation split. Run ```recon_vqgan.py``` and the rFID / rFVD will be computed directly.

Note that you can also evaluate the rFID on your own datasets:

```
python3 pytorch-fid/src/pytorch_fid/__main__.py {PATH_TO_GENERATED_IMGS} {PATH_TO_GENERATED_IMGS}
```


## Image Generation

For image generation, we use ```pytorch-fid/evaluator.py``` for evaluation:

```
python3 pytorch-fid/evaluator.py {PATH_TO_GT_NPZ} {PATH_TO_YOUR_NPZ}
```

We provide a precomputed [npz file](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/imagenet.npz) as reference, you can replace {PATH_TO_GT_NPZ} with its path.

## Video Generation

For video generation, we use ```fvd_external.py``` for evaluation:

```
python3 fvd_external.py --dataset ucf/k600 --gen_dir {PATH_TO_GENERATED_VIDEOS} --gt_dir {PATH_TO_GT_VIDEOS} --resolution 128/64 --num_videos 2048
```
