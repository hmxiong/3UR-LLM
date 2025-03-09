# training with v-detr
bash test_vdetr.sh
bash train_vdetr.sh

# eval result on scanqa
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/pretrain_3detr_concate.yaml

cd calculate_scores on scanqa

python calculate_score_scanqa.py --folder /13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/3DQA_3DETR_CONCATE_32_only_scannet/20240330072 --epoch 39

python calculate_score_scanqa.py --folder /13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/3DQA_3DETR_CONCATE_32_only_scannet/20240401063 --epoch 39

/13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/3DQA_3DETR_CONCATE_TEST/20240402063
## on sqa3d
### use ours
python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path //13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/projects/blip2/train/sqa/finetune_sqa_test_set.yaml

(zero-shot)
python calculate_score_sqa3d.py --folder /13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/SQA3D_3DETR_CoT_test_set/20240218060 --epoch 0
(finetune)

### use 3d_llm 
python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path /13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/projects/blip2/train/finetune_sqa3d.yaml
(zero-shot)
(finetune)

python calculate_score_sqa3d.py --folder /13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/3DQA_3DETR_CONCATE_SQA3D/20240229122 --epoch 0

/13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/SQA3D_3DETR_CoT/20240131064/result

# training and test script
## judge
python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip2/train/judge/finetune_scanqa_3detr_concate_judge.yaml
## no judge
python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path lavis/projects/blip2/train/3detr_concate/finetune_scanqa_3detr_concate.yaml
