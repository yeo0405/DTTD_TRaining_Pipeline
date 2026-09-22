#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 eval.py --dataset_root /home/yeo/Downloads/POSE/dataset\
                    --model /home/yeo/Downloads/POSE/TRANING_PIPELINE/result/result_SEP09_AB_m2p1_0909-1537/best.pth\
                    --base_latent 128 --embed_dim 256 --fusion_block_num 1 --layer_num_m 2 --layer_num_p 1\
                    --output eval_results\
                    --filter \
                    --visualize
                    # --debug \
                    # --visualize #--debug --visualize \