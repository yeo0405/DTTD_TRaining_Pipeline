#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

python3 eval_gt.py \
    --dataset_root /home/yeo/Downloads/POSE/dataset \
    --output eval_results \
    --visualize