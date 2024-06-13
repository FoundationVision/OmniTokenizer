python3 -m torch.distributed.run --nproc_per_node=8 --master_port 42914 transformer_eval.py --inference_type "video" \
                      --gpt_ckpt "./ckpts_pub/ucf_class_lm.ckpt" --vqgan_ckpt "./ckpts_pub/imagenet_ucf.ckpt" \
                      --batch_size 1 --save ./ucf_classcond_eval4096_0.9_cfg0.5_noscale/ --n_sample 2048 --class_cond --cfg_ratio 0.5 --no_scale_cfg \
                      --top_k 4096 --top_p 0.9           