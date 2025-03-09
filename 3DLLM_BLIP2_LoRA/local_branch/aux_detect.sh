python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip2/train/finetune_scanqa_aux_detection.yaml

python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip2/train/vdetr_style/pretrain_auxdect.yaml  # add pretrain

python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip2/train/finetune_scanrefer.yaml