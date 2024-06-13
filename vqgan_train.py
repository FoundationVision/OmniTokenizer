import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from OmniTokenizer import VQGAN, VideoData, OmniTokenizer_VQGAN
from OmniTokenizer.utils import inflate_dis, inflate_gen
from OmniTokenizer.modules.callbacks import ImageLogger, VideoLogger

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='omnitokenizer')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--no_init_idis', action="store_true")
    parser.add_argument('--inflation_pe', action="store_true")
    parser.add_argument('--freeze_trans', action="store_true")

    parser.add_argument('--init_vgen', type=str, default=None)
    parser.add_argument('--init_vdis', type=str, default=None)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQGAN.add_model_specific_args(parser)
    parser = OmniTokenizer_VQGAN.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()

    
    data = VideoData(args)
    model = OmniTokenizer_VQGAN(args)
    
    if args.pretrained is not None:
        load_weights = torch.load(args.pretrained)["state_dict"]

        if args.init_vgen is None:
            del load_weights["encoder.to_patch_emb.1.weight"]
            del load_weights["encoder.to_patch_emb.1.bias"]
            del load_weights["encoder.to_patch_emb.2.weight"]
            del load_weights["encoder.to_patch_emb.2.bias"]
            del load_weights["encoder.to_patch_emb.3.weight"]
            del load_weights["encoder.to_patch_emb.3.bias"]

            del load_weights["decoder.to_pixels.0.weight"]
            del load_weights["decoder.to_pixels.0.bias"]
        
        elif args.init_vgen == "keep":
            load_weights = load_weights
        
        else:
            load_weights = inflate_gen(load_weights, temporal_patch_size=args.temporal_patch_size, spatial_patch_size=args.patch_size, strategy=args.init_vgen, inflation_pe=args.inflation_pe)
        

        if args.use_vae:
            del load_weights["pre_vq_conv.1.weight"]
            del load_weights["pre_vq_conv.1.bias"]
        
        if args.init_vdis is None:
            print("#" * 50)
            print(f"Remove the weights of video discriminator.")
            print("#" * 50)

            vids_weights = {k: v for k, v in load_weights.items() if "video_discriminator" in k}
            for k in vids_weights.keys():
                del load_weights[k]


            if args.no_init_idis:
                print("#" * 50)
                print(f"Remove the weights of image discriminator.")
                print("#" * 50)
                idis_weights = {k: v for k, v in load_weights.items() if "image_discriminator" in k}
                for k in idis_weights.keys():
                    del load_weights[k]
        
        elif args.init_vdis == "keep":
            load_weights = load_weights
        
        else:
            load_weights = inflate_dis(load_weights, strategy=args.init_vdis)
        
        msg = model.load_state_dict(load_weights, strict=False)
        missing_keys = msg.missing_keys
        unexpec_keys = msg.unexpected_keys

        missing_keys = [
            i for i in missing_keys if "discriminator" not in i and "teacher" not in i
        ]

        unexpec_keys = [
            i for i in unexpec_keys if "video_perceptual" not in i
        ]

        print(f"Model loaded from {args.pretrained}.")
        print(f"Missing: {missing_keys}")
        print(f"Unexpected: {unexpec_keys}")

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{recon_loss:.2f}'))
    # callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1, filename='{epoch}-{step}-10000-{train-recon_loss:.2f}'))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(ImageLogger(batch_frequency=750, max_images=4, clamp=True))

    if len(args.data_path) > 1 or 'ucf' in args.data_path[0] or "k400" in args.data_path[0] or "sthv2" in args.data_path[0] or "moment" in args.data_path[0] or "k600" in args.data_path[0]:
        print("Log the reconstructed videos...")
        callbacks.append(VideoLogger(batch_frequency=1500, max_videos=4, clamp=True))

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(accelerator='ddp', gpus=args.gpus)
        if args.bf16:
            kwargs = dict(accelerator='ddp', gpus=args.gpus, precision="bf16")
            # kwargs["precision"] = "bf16"
        
        if args.fp16:
            kwargs = dict(accelerator='ddp', gpus=args.gpus, precision=16)
    
    # load the most recent checkpoint file
    base_dir = os.path.join(args.default_root_dir, 'omnitokenizer')
    version_id_used = 0
    
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        
        versions = os.listdir(base_dir)
        if len(versions) > 0:
            versions = sorted(versions, key = lambda x : int(x.split('_')[1]))
            log_folder = versions[-1]
            # version_id_used = int(log_folder.split('_')[1])

        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            
            if len(ckpt_file) == 0 and len(os.listdir(ckpt_folder)) > 0:
                ckpt_files = os.listdir(ckpt_folder)
                ckpt_files = [c for c in ckpt_files if c.startswith("epoch")]
                ckpt_files = sorted(ckpt_files, key = lambda x : int(x.split("=")[2].split("-")[0]))
                ckpt_file = ckpt_files[-1]
                # val_check_interval

            if len(ckpt_file) > 0:
                args.resume_from_checkpoint = os.path.join(ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s'%args.resume_from_checkpoint)

    wandb_logger = WandbLogger(project="omnitokenizer", name=os.path.basename(args.default_root_dir), save_dir=args.default_root_dir, config=args, version=version_id_used)
    trainer = pl.Trainer.from_argparse_args(args, log_every_n_steps=49, logger=wandb_logger, callbacks=callbacks, limit_val_batches=0, num_sanity_val_steps=0, max_steps=args.max_steps, **kwargs)
    

    if args.freeze_trans:
        for name, param in model.named_parameters():
            if ("enc_spatial_transformer" in name or "enc_temporal_transformer" in name or "dec_spatial_transformer" in name or "dec_temporal_transformer" in name) and "teacher" not in name:
                param.requires_grad = False
                print(f"freeze {name}.")

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

