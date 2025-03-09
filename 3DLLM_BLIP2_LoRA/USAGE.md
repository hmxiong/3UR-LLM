# training with v-detr
bash test_vdetr.sh
bash train_vdetr.sh

# eval result v-detr
cd calculate_scores
python calculate_score_scanqa.py --folder /13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/3DQA_VDETR/20231206084 --epoch 20

python calculate_score_scanqa.py --folder /13390024681/3D/3D-LLM/3DLLM_BLIP2-base/lavis/output/BLIP2/3DQA/20231206100 --epoch 20