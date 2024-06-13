python3 transformer_train.py --tokenizer "omnitokenizer" --num_workers 32 --progress_bar_refresh_rate 500 \
                    --num_nodes 4 --gpus 8 --sync_batchnorm --batch_size 8 --cond_stage_key label \
                    --base_lr 1e-3 --lr_min 1e-3 --warmup_steps 0 --warmup_lr_init 0. \
                    --vqvae {VQVAE_CKPT} --default_root_dir {PATH_TO_SAVE_CKPTS} \
                    --loader_type 'joint' --data_path {DATA_DIR} \
                    --train_datalist './annotations/imagenet_train.txt' --val_datalist './annotations/imagenet_val.txt' \
                    --vocab_size 8192 --block_size 1025 --n_layer 24 --n_head 16 --n_embd 1536 \
                    --resolution 256 --sequence_length 17 --max_steps 4000000 \
                    --starts_with_sos --p_drop_cond 0.1 --class_first