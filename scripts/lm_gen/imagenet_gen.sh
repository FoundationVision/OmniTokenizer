python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42744 transformer_eval.py --inference_type "image" \
                      --gpt_ckpt "./ckpts_pub/imagenet_clas_lm.ckpt" --vqgan_ckpt "./ckpts_pub/imagenet_ucf.ckpt" \
                      --batch_size 3 --save ./inet_joint_2048_1.0_cfg1.5_noscale/ --n_sample 50000 --class_cond --class_first --cfg_ratio 1.5 --no_scale_cfg \
                      --top_k 2048 --top_p 1.0