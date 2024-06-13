

python3 vqgan_eval.py --inference_type 'image' --patch_embed 'linear' --patch_size 8 --temporal_patch_size 2 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --disc_layers 3 \
                      --enc_block "ttww" --dec_block "tttt" --q_strides "1111" --twod_window_size 8 \
                      --casual_in_temporal_transformer --casual_in_peg --dim_head 64 --heads 8 --apply_noise --apply_blur \
                      --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
                      --vqgan_ckpt "./ckpts_pub/celebahq.ckpt" \
                      --batch_size 16 --data_path {PATH_TO_DATA_DIR} --train_datalist './annotations/celebahq.txt'  --val_datalist './annotations/celebahq.txt'  \
                     --loader_type "joint" --dataset 'celebahq_only' --save ./celebahq \
                      --resolution 256 --sequence_length 17 --discriminator_iter_start 0 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4

python3 vqgan_eval.py --inference_type 'image' --patch_embed 'linear' --patch_size 8 --temporal_patch_size 2 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --disc_layers 3 \
                      --enc_block "ttww" --dec_block "tttt" --q_strides "1111" --twod_window_size 8 \
                      --casual_in_temporal_transformer --casual_in_peg --dim_head 64 --heads 8 --apply_noise --apply_blur \
                      --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
                      --vqgan_ckpt "./ckpts_pub/ffhq.ckpt" \
                      --batch_size 16 --data_path {PATH_TO_DATA_DIR} --train_datalist './annotations/ffhq.txt' --val_datalist './annotations/ffhq.txt' \
                      --loader_type "joint" --dataset 'ffhq_only' --save ./ffhq \
                      --resolution 256 --sequence_length 17 --discriminator_iter_start 0 --norm_type batch --replacewithgt 0 \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4

python3 vqgan_eval.py --inference_type 'image' --patch_embed 'linear' --patch_size 8 --temporal_patch_size 4 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --disc_layers 3 \
                      --enc_block "ttww" --dec_block "tttt" --q_strides "1111" --twod_window_size 8 \
                      --casual_in_temporal_transformer --casual_in_peg --dim_head 64 --heads 8 --apply_noise --apply_blur --spatial_pos "rope" \
                      --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
                      --vqgan_ckpt "./ckpts_pub/celebahq_ucf.ckpt" \
                      --batch_size 16 --data_path {PATH_TO_DATA_DIR} --train_datalist './annotations/celebahq.txt'  --val_datalist './annotations/celebahq.txt'  \
                     --loader_type "joint" --dataset 'celebahq_ucf' --save ./celebahq \
                      --resolution 256 --sequence_length 17 --discriminator_iter_start 0 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4

python3 vqgan_eval.py --inference_type 'image' --patch_embed 'linear' --patch_size 8 --temporal_patch_size 4 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --disc_layers 3 \
                      --enc_block "ttww" --dec_block "tttt" --q_strides "1111" --twod_window_size 8 \
                      --casual_in_temporal_transformer --casual_in_peg --dim_head 64 --heads 8 --apply_noise --apply_blur --spatial_pos "rope" \
                      --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
                      --vqgan_ckpt "./ckpts_pub/ffhq_ucf.ckpt" \
                      --batch_size 16 --data_path {PATH_TO_DATA_DIR} --train_datalist './annotations/ffhq.txt' --val_datalist './annotations/ffhq.txt' \
                      --loader_type "joint" --dataset 'ffhq_ucf' --save ./ffhq \
                      --resolution 256 --sequence_length 17 --discriminator_iter_start 0 --norm_type batch --replacewithgt 0 \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4


python3 vqgan_eval.py --inference_type 'image' --patch_embed 'linear' --patch_size 8 --temporal_patch_size 4 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --disc_layers 3 \
                      --enc_block "ttww" --dec_block "tttt" --q_strides "1111" --twod_window_size 8 \
                      --casual_in_temporal_transformer --casual_in_peg --dim_head 64 --heads 8 --apply_noise --apply_blur --spatial_pos "rope" \
                      --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
                      --vqgan_ckpt "./ckpts_pub/celebahq_k600.ckpt" \
                      --batch_size 16 --data_path {PATH_TO_DATA_DIR} --train_datalist './annotations/celebahq.txt'  --val_datalist './annotations/celebahq.txt'  \
                     --loader_type "joint" --dataset 'celebahq_k600' --save ./celebahq \
                      --resolution 256 --sequence_length 17 --discriminator_iter_start 0 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4

python3 vqgan_eval.py --inference_type 'image' --patch_embed 'linear' --patch_size 8 --temporal_patch_size 4 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --disc_layers 3 \
                      --enc_block "ttww" --dec_block "tttt" --q_strides "1111" --twod_window_size 8 \
                      --casual_in_temporal_transformer --casual_in_peg --dim_head 64 --heads 8 --apply_noise --apply_blur --spatial_pos "rope" \
                      --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
                      --vqgan_ckpt "./ckpts_pub/ffhq_k600.ckpt" \
                      --batch_size 16 --data_path {PATH_TO_DATA_DIR} --train_datalist './annotations/ffhq.txt' --val_datalist './annotations/ffhq.txt' \
                      --loader_type "joint" --dataset 'ffhq_k600' --save ./ffhq \
                      --resolution 256 --sequence_length 17 --discriminator_iter_start 0 --norm_type batch --replacewithgt 0 \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4
