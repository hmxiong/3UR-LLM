#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

strings /root/anaconda3/envs/3d/lib/libstdc++.so.6 | grep GLIBCXX
rm /lib/x86_64-linux-gnu/libstdc++.so.6
cp /root/anaconda3/envs/3d/lib/libstdc++.so.6 /lib/x86_64-linux-gnu

cd /13390024681/3D/3D-LLM/3DLLM_BLIP2-base

conda activate 3d

python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip2/train/opt/pretrain_3detr_opt.yaml
