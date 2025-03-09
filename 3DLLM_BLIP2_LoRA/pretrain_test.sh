#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

cd /13390024681/3D/3D-LLM/3DLLM_BLIP2-base

conda activate 3d

python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip2/train/pretrain.yaml
