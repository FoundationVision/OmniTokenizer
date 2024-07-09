# Stage1: Image-only training on a fixed resolution
python3 vqgan_train.py --tokenizer 'omnitokenizer' --patch_embed 'linear' --patch_size 8 --temporal_patch_size 2 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --disc_layers 3 \
                      --enc_block "ttww" --dec_block "tttt" --twod_window_size 8 \
                      --casual_in_temporal_transformer --casual_in_peg --dim_head 64 --heads 8 \
                      --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
                      --num_nodes 4 --gpus 8 --sync_batchnorm --batch_size 8 --num_workers 8 --grad_accumulates 1 --grad_clip_val 1.0 --apply_noise --apply_blur \
                      --lr 1e-3 --lr_min 5e-5 --warmup_steps 50000 --warmup_lr_init 0. --dis_lr_multiplier 0.1 --dis_minlr_multiplier --dis_warmup_steps 500000 \
                      --progress_bar_refresh_rate 500 --max_steps 500000 \
                      --loader_type 'joint' --data_path {IMAGE_DIR} \
                      --default_root_dir {PATH_TO_SAVE_CKPT} \
                      --train_datalist {IMAGE_DATALIST} --val_datalist {IMAGE_DATALIST} \
                      --resolution 256 --sequence_length 17 --discriminator_iter_start 0 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 0.01 --video_gan_weight 1  --gan_feat_weight 4 --logitslaplace_weight 0.4 --initialize_vit --disloss_check_thres 0.001

# Stage2: Image and Video joint training on multiple resolutions
python3 vqgan_train.py --tokenizer 'omnitokenizer' --patch_embed 'linear' --patch_size 8 --temporal_patch_size 4 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --disc_layers 3 \
                      --enc_block "ttww" --dec_block "tttt" --twod_window_size 8 \
                      --casual_in_temporal_transformer --casual_in_peg --dim_head 64 --heads 8 \
                      --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
                      --num_nodes 4 --gpus 8 --sync_batchnorm --num_workers 8 --grad_accumulates 2 --force_alternation --grad_clip_val 1.0 --apply_noise \
                      --lr 5e-5 --lr_min 5e-5 --warmup_steps 50000 --warmup_lr_init 0. --dis_lr_multiplier 0.1 --dis_minlr_multiplier --dis_warmup_steps 500000 \
                      --progress_bar_refresh_rate 500 --max_steps 500000 \
                      --loader_type 'joint' --batch_size 4 8 --sample_ratio 1 1 \
                      --data_path {VIDEO_DIR} {IMAGE_DIR} \
                      --default_root_dir {PATH_TO_SAVE_CKPT} \
                      --train_datalist {VIDEO_DATALIST} {IMAGE_DATALIST} --val_datalist {VIDEO_DATALIST} {IMAGE_DATALIST} \
                      --resolution 256 --sequence_length 17 --fps -1 --discriminator_iter_start 0 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 0 --video_gan_weight 0.01  --gan_feat_weight 4 --disloss_check_thres 0.001 --pretrained {CKPT_OF_PREVIOUS_STAGE} --no_init_idis --init_vgen "average" --activation_in_disc "leaky_relu" --resolution_scale 0.5 0.75 1.0 1.25 --spatial_pos "rope"

# Stage3: Finetuning w/ KL loss to train a VAE
python3 vqgan_train.py --tokenizer 'omnitokenizer' --patch_embed 'linear' --patch_size 8 --temporal_patch_size 4 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --disc_layers 3 \
                      --enc_block "ttww" --dec_block "tttt" --twod_window_size 8 \
                      --casual_in_temporal_transformer --casual_in_peg --dim_head 64 --heads 8 \
                      --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
                      --num_nodes 4 --gpus 8 --sync_batchnorm --num_workers 8 --grad_accumulates 2 --force_alternation --grad_clip_val 1.0 --apply_noise \
                      --lr 5e-5 --lr_min 5e-5 --warmup_steps 50000 --warmup_lr_init 0. --dis_lr_multiplier 0.1 --dis_minlr_multiplier --dis_warmup_steps 500000 \
                      --progress_bar_refresh_rate 500 --max_steps 500000 \
                      --loader_type 'joint' --batch_size 4 8 --sample_ratio 1 1 \
                      --data_path {VIDEO_DIR} {IMAGE_DIR} \
                      --default_root_dir {PATH_TO_SAVE_CKPT} \
                      --train_datalist {VIDEO_DATALIST} {IMAGE_DATALIST} --val_datalist {VIDEO_DATALIST} {IMAGE_DATALIST} \
                      --resolution 256 --sequence_length 17 --fps -1 --discriminator_iter_start 0 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 0 --video_gan_weight 0.01  --gan_feat_weight 4 --disloss_check_thres 0.001 --pretrained {CKPT_OF_PREVIOUS_STAGE} --init_vgen "keep" --init_vdis "keep" --activation_in_disc "leaky_relu" --resolution_scale 0.5 0.75 1.0 1.25 --spatial_pos "rope" --use_vae --kl_weight 1e-6
