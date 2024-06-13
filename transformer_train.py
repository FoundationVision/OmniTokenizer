import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from OmniTokenizer import Net2NetTransformer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from OmniTokenizer import VideoData
from OmniTokenizer.modules.callbacks import ImageLogger, VideoLogger


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='omnitokenizer')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Net2NetTransformer.add_model_specific_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    # data.train_dataloader()
    # data.test_dataloader()

    args.class_cond_dim = data.n_classes if not args.unconditional and args.cond_stage_key=='label' else None
    model = Net2NetTransformer(args, first_stage_key=args.first_stage_key, cond_stage_key=args.cond_stage_key)
    # print(ModelSummary(model))

    # configure learning rate
    bs, base_lr = args.batch_size, args.base_lr
    ngpu = args.gpus
    model.learning_rate = args.base_lr
    

    callbacks = []
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{train/loss:.2f}'))
    # callbacks.append(ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{train/loss:.2f}'))
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=3, filename='best_checkpoint'))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    

    if "ucf" in args.data_path[0] or "k400" in args.data_path[0] or "sthv2" in args.data_path[0] or "moment" in args.data_path[0] or "k600" in args.data_path[0]:
        print("Log the reconstructed videos...")
        callbacks.append(VideoLogger(batch_frequency=1500, max_videos=4, clamp=True))
    
    elif "imagenet" in args.data_path[0]:
        print("Log the reconstructed images...")
        callbacks.append(ImageLogger(batch_frequency=1500, max_images=4, clamp=True))

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(gpus=args.gpus,
                      # plugins=["deepspeed_stage_2"])
                      plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)])
        
        if args.bf16:
            kwargs["precision"] = "bf16"
        
        if args.fp16:
            kwargs["precision"] = 16

    # load the most recent checkpoint file
    base_dir = os.path.join(args.default_root_dir, 'videogen_llm')
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

    print(f"Train from the version: {version_id_used}.")

    wandb_logger = WandbLogger(project="videogen_llm", name=os.path.basename(args.default_root_dir), save_dir=args.default_root_dir, config=args, version=version_id_used)
    trainer = pl.Trainer.from_argparse_args(args, log_every_n_steps=49, logger=wandb_logger, callbacks=callbacks,
                                            max_steps=args.max_steps, **kwargs)


    trainer.fit(model, data)


if __name__ == '__main__':
    main()

