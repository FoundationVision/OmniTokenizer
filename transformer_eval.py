import os
import random
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from OmniTokenizer import load_transformer, load_vqgan
from OmniTokenizer import DecordVideoDataset
from OmniTokenizer.utils import save_video_grid
from OmniTokenizer.modules.gpt import sample_with_past, sample_with_past_cfg
import ddp_utils as utils
from einops import rearrange, repeat

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self



@torch.no_grad()
def class_condition_generation(gpt, batch_size, n_sample_class, class_label, save_dir, temperature=None, top_k=None, top_p=None, n_cond=0, starts_with_sos=False, cfg_ratio=None, scale_cfg=False, class_first=False):
    
    if args.inference_type == "image":
        latent_shape = [256 // 8, 256 // 8]

    else:
        latent_shape = [
            (17 - 1) // 4 + 1, 256 // 8, 256 // 8
        ]

    steps = np.prod(latent_shape)
    n_batch_class = n_sample_class // batch_size + 1
    # print(n_sample_class, batch_size, n_batch_class)

    for sample_id in range(n_batch_class):
        c_indices = repeat(torch.tensor([class_label]), '1 -> b 1', b=batch_size).to(gpt.device)  # class token
        
        if not starts_with_sos:
            index_sample = sample_with_past(c_indices, gpt.module.transformer, steps=steps,
                                            sample_logits=True, top_k=top_k, callback=None,
                                            temperature=temperature, top_p=top_p)


        elif starts_with_sos and cfg_ratio is None:
            sos = torch.zeros_like(c_indices, dtype=c_indices.dtype, device=c_indices.device)
            c_indices += 1
            if class_first:
                c_indices = torch.cat((c_indices, sos), dim=1)
            else:
                c_indices = torch.cat((sos, c_indices), dim=1)
            
            index_sample = sample_with_past(c_indices, gpt.module.transformer, steps=steps,
                                            sample_logits=True, top_k=top_k, callback=None,
                                            temperature=temperature, top_p=top_p)


        else:
            index_sample = sample_with_past_cfg(c_indices, gpt.module.transformer, steps=steps,
                                            sample_logits=True, top_k=top_k, callback=None,
                                            temperature=temperature, top_p=top_p, cfg_ratio=cfg_ratio, class_first=class_first, scale_cfg=scale_cfg
            )

        
        index = torch.clamp(index_sample-n_cond, min=0, max=gpt.module.first_stage_model.n_codes-1)
        x_sample = gpt.module.first_stage_model.decode(index, is_image=(args.inference_type == "image"))
        samples = torch.clamp(x_sample + 0.5, 0, 1) #torch.clamp(x_sample, -0.5, 0.5) + 0.5
        
        if args.inference_type == "video":
            for i, sample in enumerate(samples):
                video_id = sample_id * batch_size + i
                save_video_grid(sample.unsqueeze(0), os.path.join(save_dir, 'generation_class%d_%d.mp4'%(class_label, video_id)), 1)

        else:
            images_batch = torch.clamp(samples, 0, 1)
            for i, img in enumerate(images_batch):
                img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img).convert("RGB")
                image_id = sample_id * batch_size + i
                save_path = os.path.join(save_dir, 'generation_%d_%d.png'%(class_label, image_id))
                img.save(save_path)        
    
    
    return



def frame_prediction(loader, gpt, vqgan, args, save_dir):
    if args.inference_type == "image":
        latent_shape = [256 // 8, 256 // 8]

    else:
        latent_shape = [
            (17 - 1) // 4 + 1, 256 // 8, 256 // 8
        ]

    steps = np.prod(latent_shape)
    loader = iter(loader)
    num_batches = args.n_sample // (utils.get_world_size() * args.batch_size) 
    if args.n_sample % (utils.get_world_size() * args.batch_size) != 0:
        num_batches += 1
    
    for _ in tqdm(range(num_batches)):
        batch = next(loader)
        input_videos = batch["video"].to(vqgan.device)
        _, prefix_encodings = vqgan.module.encode(input_videos, is_image=False, include_embeddings=True)
        prefix_encodings = prefix_encodings[:, :2]
        B, _, H, W = prefix_encodings.shape
        prefix_encodings = prefix_encodings.view(B, -1)
        index_sample = sample_with_past(prefix_encodings, gpt.module.transformer, steps=int(steps - 2 * H * W), sample_logits=True, top_k=args.top_k, temperature=1.0, top_p=args.top_p)

        index = torch.clamp(index_sample, min=0, max=gpt.module.first_stage_model.n_codes-1)
        index = torch.cat((prefix_encodings, index), dim=1)
        index = rearrange(index, "b (t h w) -> b t h w", h=H, w=W)
        x_sample = gpt.module.first_stage_model.decode(index, is_image=False)
        samples = torch.clamp(x_sample + 0.5, 0, 1)
        input_videos = torch.clamp(input_videos + 0.5, 0, 1)

        
        paths = batch["path"]
        for in_video, sample, path in zip(input_videos, samples, paths):
            video_class = os.path.basename(os.path.dirname(path))
            video_name = os.path.basename(path)
            os.makedirs(os.path.join(save_dir, "recon"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "input"), exist_ok=True)
            save_video_grid(sample.unsqueeze(0), os.path.join(save_dir, "recon", video_class + "_" + video_name), 1)
            save_video_grid(in_video.unsqueeze(0), os.path.join(save_dir, "input", video_class + "_" + video_name), 1)      
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='omnitokenizer')
    parser.add_argument('--gpt_ckpt', type=str, default='')
    parser.add_argument('--vqgan_ckpt', type=str, default='')
    parser.add_argument('--inference_type', type=str, default='image', choices=["image", "video"])
    parser.add_argument('--save', type=str, default='./results/tats')
    parser.add_argument('--top_k', type=int, default=2048)
    parser.add_argument('--top_p', type=float, default=0.92)
    parser.add_argument('--n_sample', type=int, default=1000*50)
    parser.add_argument('--data_dir', type=str, default='ucf101')
    parser.add_argument("--data_list", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument('--class_cond', action='store_true')
    parser.add_argument('--class_first', action="store_true")
    parser.add_argument('--cfg_ratio', type=float, default=None)
    parser.add_argument('--no_scale_cfg', action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="number of distributed processes",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="url used to set up distributed training",
    )
    parser.add_argument("--distributed", default=False, type=bool)
    args = parser.parse_args()

    utils.init_distributed_mode(args)

    gpt = load_transformer(args.gpt_ckpt, vqgan_ckpt=args.vqgan_ckpt).cuda().eval()
    vqgan = load_vqgan("omnitokenizer", args.vqgan_ckpt, device="cuda")
    vqgan.codebook._need_init = False
    vqgan.train = disabled_train
    vqgan.eval()

    device = torch.device("cuda")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    save_dir = '%s/topp%.2f_topk%d'%(args.save, args.top_p, args.top_k)
    if utils.get_rank() == 0:
        print('generating and saving video to %s...'%save_dir)
        os.makedirs(save_dir, exist_ok=True)

    if args.distributed:
        gpt = torch.nn.parallel.DistributedDataParallel(
            gpt, device_ids=[args.gpu], find_unused_parameters=True
        )

        vqgan = torch.nn.parallel.DistributedDataParallel(
            vqgan, device_ids=[args.gpu], find_unused_parameters=True
        )

    if not args.class_cond:
        dataset = DecordVideoDataset(
            args.data_dir, args.data_list, sequence_length=17, train=False, resolution=256
        )
        
        if args.distributed:
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank()
            )
        else:
            sampler = None
                
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=False,
            sampler=sampler,
            shuffle=False
        )
        
        frame_prediction(
            dataloader, gpt, vqgan, args, save_dir
        )
    

    else:
        starts_with_sos = gpt.module.starts_with_sos
        num_classes = gpt.module.class_cond_dim
        num_classes_per_rank = num_classes // utils.get_world_size()

        if num_classes % utils.get_world_size() != 0:
            num_classes_per_rank += 1
        
        class_start = utils.get_rank() * num_classes_per_rank
        class_end = min(class_start + num_classes_per_rank, gpt.module.class_cond_dim)

        i = class_start
        for _ in tqdm(range(class_start, class_end), desc=f"Generate {args.n_sample // gpt.module.class_cond_dim + 1} cases for {i}-th class on rank{utils.get_rank()}"):
            class_condition_generation(
                gpt, args.batch_size, args.n_sample // gpt.module.class_cond_dim + 1, i, save_dir, temperature=1.0, top_k=args.top_k, top_p=args.top_p, 
                n_cond=gpt.module.class_cond_dim if not starts_with_sos else gpt.module.class_cond_dim + 1, starts_with_sos=starts_with_sos, cfg_ratio=args.cfg_ratio, scale_cfg=not args.no_scale_cfg, class_first=args.class_first 
            )
            i += 1
