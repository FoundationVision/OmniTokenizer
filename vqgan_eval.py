import os
import tqdm
import json
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import torch.nn as nn
from torchvision.models.inception import inception_v3
from einops import rearrange
from tqdm import tqdm
from PIL import Image

from OmniTokenizer import VideoData
from OmniTokenizer import OmniTokenizer_VQGAN, VQGAN
from OmniTokenizer.utils import save_video_grid
from OmniTokenizer.utils import shift_dim
from OmniTokenizer.fvd.fvd import get_fvd_logits, frechet_distance, load_fvd_model


def calculate_batch_codebook_usage_percentage(batch_encoding_indices,n_codes):
    # Flatten the batch of encoding indices into a single 1D tensor
    all_indices = batch_encoding_indices.flatten()

    # Initialize a tensor to store the percentage usage of each code
    codebook_usage = torch.zeros(n_codes, dtype=torch.long)
    
    # Count the number of occurrences of each index and get their frequency as percentages
    unique_indices, counts = torch.unique(all_indices, return_counts=True)
    
    # Populate the corresponding percentages in the codebook_usage_percentage tensor
    codebook_usage[unique_indices.long()] = counts
    
    return codebook_usage


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

parser = argparse.ArgumentParser()
parser = VideoData.add_data_specific_args(parser)
parser = VQGAN.add_model_specific_args(parser)
parser = OmniTokenizer_VQGAN.add_model_specific_args(parser)
parser.add_argument('--tokenizer', type=str, default="omnitokenizer")
parser.add_argument('--vqgan_ckpt', type=str, default=None)
parser.add_argument('--train', action="store_true")
parser.add_argument('--inference_type', type=str, choices=["image", "video"])
parser.add_argument('--infer_downsample', type=int, default=None)
parser.add_argument('--replacewithgt', type=int, default=None)
parser.add_argument('--save', type=str, default='./results/tats')
parser.add_argument('--dataset', type=str, default='ucf101')
parser.add_argument('--save_videos', action='store_true')
args = parser.parse_args()

n_row = 1 # int(np.sqrt(args.batch_size))

device = torch.device('cuda')

vqgan = OmniTokenizer_VQGAN(args)
load_weights = torch.load(args.vqgan_ckpt, map_location=torch.device("cpu"))["state_dict"]

vids_weights = {k: v for k, v in load_weights.items() if "video_discriminator" in k}
for k in vids_weights.keys():
    del load_weights[k]

msg = vqgan.load_state_dict(load_weights, strict=False)
print(f"Model loaded from {args.vqgan_ckpt}.")
print(f"Missing: {msg.missing_keys}")
print(f"Unexpected: {msg.unexpected_keys}")


vqgan.to(device)

vqgan.encoder.image_size = (args.resolution, args.resolution)
vqgan.decoder.image_size = (args.resolution, args.resolution)

try:
    num_codes = vqgan.codebook.n_codes
except:
    num_codes = vqgan.codebook.codebook_size

vqgan.codebook._need_init = False
vqgan.train = disabled_train
vqgan.to(device).eval()

save_dir = '%s/%s'%(args.save, args.dataset)
print('generating and saving video to %s...'%save_dir)
os.makedirs(save_dir, exist_ok=True)

data = VideoData(args)

if args.train:
    loader = data.train_dataloader()[0]
else:
    loader = data.val_dataloader()


use_vae = vqgan.use_vae

if args.inference_type == "video":
    i3d = load_fvd_model(device)

    os.makedirs(os.path.join(save_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "recons"), exist_ok=True)

    real_embeddings = []
    fake_embeddings = []

    total_usage = torch.zeros(num_codes).to(device)

    print('computing fvd embeddings for real/fake videos')
    i = 0
    for batch in tqdm(loader):
        with torch.no_grad():
            input_ = batch['video'] # B C T H W
            B = input_.shape[0]
            _, _, x, x_recons, vq_output = vqgan(input_.to(device), log_image=True)

            if args.infer_downsample is not None:
                real_videos = batch['video'] + 0.5
                fake_videos = torch.clamp(x_recons.detach().cpu()+0.5, 0, 1)
                B, C, T, H, W = real_videos.shape
                real_videos = rearrange(real_videos, "b c t h w -> (b t) c h w")
                fake_videos = rearrange(fake_videos, "b c t h w -> (b t) c h w")
                real_videos = F.interpolate(
                    real_videos, scale_factor=1/args.infer_downsample, mode="bilinear", align_corners=False
                )
                fake_videos = F.interpolate(
                    fake_videos, scale_factor=1/args.infer_downsample, mode="bilinear", align_corners=False
                )

                real_videos = rearrange(real_videos, "(b t) c h w -> b c t h w", b=B)
                fake_videos = rearrange(fake_videos, "(b t) c h w -> b c t h w", b=B)
                
            else:
                real_videos = batch['video'] + 0.5
                fake_videos = torch.clamp(x_recons.detach().cpu()+0.5, 0, 1)

            
            if args.replacewithgt is not None:
                # B C T H W
                fake_videos = torch.cat((real_videos[:, :, :args.replacewithgt], fake_videos[:, :, args.replacewithgt:]), dim=2)
                assert fake_videos.shape[2] == args.sequence_length

            real_embeddings.append(get_fvd_logits(shift_dim(real_videos * 255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))
            fake_embeddings.append(get_fvd_logits(shift_dim(fake_videos * 255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))
        

        if not use_vae:
            batch_codebook_usage = vq_output["batch_usage"]
            total_usage += batch_codebook_usage

        if args.save_videos:
            save_video_grid(fake_videos, os.path.join(save_dir, "recons", f'{args.dataset}_{i}.gif'), n_row)
            save_video_grid(real_videos, os.path.join(save_dir, "gt", f'{args.dataset}_{i}.gif'), n_row)
        
        i += 1
        
    print('caoncat fvd embeddings for real videos')
    real_embeddings = torch.cat(real_embeddings, 0)
    print('caoncat fvd embeddings for fake videos')
    fake_embeddings = torch.cat(fake_embeddings, 0)

    print('FVD = %.2f'%(frechet_distance(fake_embeddings, real_embeddings)))
    print('Usage = %.2f'%((total_usage > 0.).sum() / num_codes))


else:
    total_usage = torch.zeros(num_codes).to(device)
    total_counts = torch.zeros(num_codes)

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    def get_pred(x, resize=True):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()
    

    i = 0
    for batch in tqdm(loader):
        with torch.no_grad():
            _, _, x, x_recons, vq_output = vqgan(batch['video'].to(device), log_image=True)
        # fake_embeddings.append(get_fvd_logits(shift_dim((x_recons.detach().cpu()+0.5)*255, 1, -1).byte().data.numpy(), i3d=i3d, device=device))
        
        if not use_vae:
            encoding_indices = vq_output["encodings"].detach().cpu()
            code_counts = calculate_batch_codebook_usage_percentage(encoding_indices, num_codes)
            total_counts += code_counts

            batch_codebook_usage = vq_output["batch_usage"]
            total_usage += batch_codebook_usage
        
        paths = batch["path"]
        assert len(paths) == x.shape[0]

        for p, input_, recon_ in zip(paths, x, x_recons):
            path = os.path.join(save_dir, "input", p)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            input_ = input_.permute(1, 2, 0).detach().cpu()
            input_ = ((input_ + 0.5).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(input_)
            if args.infer_downsample is not None:
                img = img.resize((args.resolution // args.infer_downsample, args.resolution // args.infer_downsample), Image.ANTIALIAS)
            
            img.save(path)

            path = os.path.join(save_dir, "recon", p)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            recon_ = recon_.permute(1, 2, 0).detach().cpu()
            recon_ = (torch.clamp((recon_ + 0.5), 0, 1).numpy() * 255).astype(np.uint8)
            
            rec = Image.fromarray(recon_)
            if args.infer_downsample is not None:
                rec = rec.resize((args.resolution // args.infer_downsample, args.resolution // args.infer_downsample), Image.ANTIALIAS)
            rec.save(path)
        
        i += 1
    
    
    if "imagenet" in args.train_datalist[0]:
        os.system(
            f"python3 evaluation/pytorch-fid/src/pytorch_fid/__main__.py {os.path.join(save_dir, 'input', 'val')} {os.path.join(save_dir, 'recon', 'val')}"
        )
    elif "celebahq" in args.train_datalist[0]:
        os.system(
            f"python3 evaluation/pytorch-fid/src/pytorch_fid/__main__.py {os.path.join(save_dir, 'input/CelebAMask-HQ/CelebA-HQ-img')} {os.path.join(save_dir, 'recon/CelebAMask-HQ/CelebA-HQ-img')}"
        )
    elif "ffhq" in args.train_datalist[0]:
        os.system(
            f"python3 evaluation/pytorch-fid/src/pytorch_fid/__main__.py {os.path.join(save_dir, 'input', 'val')} {os.path.join(save_dir, 'recon', 'val')}"
        )
    
    print('Usage = %.2f'%((total_usage > 0.).sum() / num_codes))
