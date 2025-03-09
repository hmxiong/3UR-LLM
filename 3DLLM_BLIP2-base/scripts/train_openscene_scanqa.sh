#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

cd /13390024681/3D/V-DETR_LLM_yichao/src

conda activate 3d_llm

# vicuna_13b_4gpu /13390024681/3D/FastChat-0.1.10/output/vicuna-13b-v0 
# llama_v2_13b /13390024681/llama/llama/llama_v2_13b_hf

max_token=$1
numgpu=4
numepoch=4
exp=vicuna_7b_v0_4gpu_openscene_six_token
dataname=ScanQA_train
# LAMM_3dinstruct_3rscan_8k  LAMM_3dinstruct_10k
# Scannet_Detection_Instruction ScanQA_Instruction
visfeat_type=local
now=$(date +"%Y%m%d_%H%M%S")

common_dataset=(ScanQA_val)
base_data_path=../data/3D_Finetune
token_num=256
layer=-2
vicuna_ckpt_path=/13390024681/3D/FastChat-0.1.10/output/vicuna-7b-v0
# vicuna_ckpt_path=/13390024681/3D/V-DETR_LLM_yichao/src/vicuna-7b-v0-expand # new_tokenizer vicuna-13b-v0-3-expand
conv_mode=simple # when using llama2 it should be 'vicuna_v1_1' mode and when using vicuna it should be 'simple'
encoder_ckpt_path=/13390024681/3D/V-DETR_LLM_yichao/model_zoo/scannet_lseg.pth.tar
# 3detr_outputs/1630_3detr_decproposal_mink_vert_inboxassign_100kfps4096_repeat5nms_nofirst_NFA_query1024_bothColorXyz_540ep_01/checkpoint_best.pth
# train_scanqa train_debug
ckpt_dir=../ckpt/${dataname}
# use_xformers 是否选择使用xformers来节省显存
# use_class_token 是否选择使用class token
# box_scene_normalize 是否选择使用场景级别的归一化方式来完成
# use_special_token 使用special token来对原始的数据进行隔离
# use_system 选择使用系统prompt

rm -r ${ckpt_dir}/${exp}/log_rest/

mkdir -p ${ckpt_dir}/${exp}/log_rest/

deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 1213 train.py \
    --stage 2 \
    --cfg ./config/fine_tune/train_scanqa.yaml \
    --data_path  /13390024681/3D/V-DETR_LLM_yichao/data/3D_Finetune/meta_data/${dataname}.json \
    --vision_root_path /13390024681/3D/V-DETR_LLM_yichao/data/3D_Finetune \
    --delta_ckpt_path /13390024681/3D/V-DETR_LLM_yichao/ckpt/Scanrefer_Captions/vicuna_7b_v0_4gpu_openscene_alignment_scanrefer_caption/pytorch_model.pt \
    --max_tgt_len ${max_token} \
    --vision_type pcl \
    --use_system \
    --model lamm_peft \
    --encoder_pretrain openscene \
    --encoder_ckpt_path ${encoder_ckpt_path} \
    --vicuna_ckpt_path ${vicuna_ckpt_path} \
    --vision_feature_type ${visfeat_type} \
    --use_xformers \
    --use_class_token \
    --training \
    --order class_center_size \
    --num_vision_token ${token_num} \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log

answerdir=../answers/${dataname}
mkdir -p ${answerdir}/${exp}
results_path=../results/${dataname}
mkdir -p ${results_path}/${exp}

for dataset in ${common_dataset[*]}; do

    python inference_3d.py \
        --model lamm_peft \
        --encoder_pretrain openscene \
        --encoder_ckpt_path ${encoder_ckpt_path} \
        --vicuna_ckpt_path ${vicuna_ckpt_path} \
        --vision_root_path /13390024681/3D/V-DETR_LLM_yichao/data/3D_Finetune \
        --vision_type pcl \
        --use_system \
        --delta_ckpt_path ${ckpt_dir}/${exp}/pytorch_model.pt \
        --max_tgt_len ${max_token} \
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --training \
        --vision_feature_type ${visfeat_type} \
        --num_vision_token ${token_num} \
        --vision_output_layer ${layer} \
        --conv_mode ${conv_mode}  \
        --dataset-name ${dataset} \
        --base-data-path ${base_data_path} \
        --inference-mode common \
        --bs 1 \
        --answers-dir ${answerdir}/${exp} \
    
    python common_eval_3d.py \
        --dataset-name ${dataset} \
        --answer-file ${answerdir}/${exp} \
        --base-data-path ${base_data_path} \
        --model lamm_peft \
        --encoder_pretrain openscene \
        --delta_ckpt_path ${ckpt_dir}/${exp}/pytorch_model.pt \
        --max_tgt_len ${max_token} \
        --vision_output_layer ${layer} \
        --conv_mode ${conv_mode}  \
        --dataset-name ${dataset} \
        --base-data-path ${base_data_path} \
        2>&1 | tee ${results_path}/${exp}/eval_${dataset}.log
done
