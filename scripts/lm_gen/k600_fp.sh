
python3 -m torch.distributed.run --nproc_per_node=8 --master_port 4224 transformer_eval.py --inference_type "video" \
                      --gpt_ckpt "./ckpts_pub/k600_fp_lm.ckpt" --vqgan_ckpt "./ckpts_pub/imagenet_k600.ckpt" \
                      --batch_size 1 --save ./k600_fp_eval2048_0.9/ --n_sample 2048 \
                      --top_k 2048 --top_p 0.9 --data_dir {PATH_TO_DATA_DIR} --data_list ./annotations/k600_train.txt                     