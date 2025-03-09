# Copyright (c) Facebook, Inc. and its affiliates.
# from .VDETR.model_3detr import build_3detr
# from .VDETR.rpe_model_3detr import build_rpe_3detr
# from .VDETR.truerpe_model_3detr import build_truerpe_3detr
from .VDETR.dataops import DatasetConfig
# from .3detr.model import bu
from .ThreeDETR.models.model_3detr import build_3detr
import torch
import torch.nn as nn
import numpy as np
import os

MODEL_FUNCS = {
    "3detr":        build_3detr,
    # "3rpedetr":     build_rpe_3detr,
    # '3truerpedetr': build_truerpe_3detr,
}

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def build_model(args, dataset_config):
    model = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model

class ThreeDETRVisionTower(nn.Module):
    def __init__(self, vision_tower, select_layer, delay_load=False) -> None:
        super().__init__()
        self.is_loaded = False
        self.hidden_size = 256 # when using sa_4 feature
        self.vision_tower_name = vision_tower
        self.select_layer = select_layer

        self.load_model()
            
    def load_model(self):
        self.image_processor = None
        # if os.path.exists(os.path.join(self.vision_tower_name)):
        #     raise ValueError(f"PoineNet++ pretrained model not found at [{self.vision_tower_name}]!")  # 目前需要再进一步等待checkpoint给我发过来
         
        print("===> Loading 3DETR encoder...")
        ckpt = torch.load(self.vision_tower_name)
        args = ckpt["args"]
        dataset_config = DatasetConfig()
        self.vision_tower , self.output_processor= build_model(args, dataset_config) 
        self.vision_tower = self.vision_tower.to(torch.float32)
        print("===> Loading 3DETR encoder...")
        match_keys = []
        for key in self.vision_tower.state_dict().keys():
            if key in ckpt["model"]:
                match_keys.append(key)

        self.vision_tower.load_state_dict(ckpt["model"], strict=False)
        self.vision_tower.requires_grad_(False)
        print(
        f"===> Loaded {self.vision_tower_name}; {len(match_keys)} keys matched; {len(self.vision_tower.state_dict().keys())} keys in total."
         ) # print the loaded model

        self.is_loaded = True 

    @torch.no_grad()
    def forward(self, point_cloud):
        # print("point_cloud",point_cloud.shape) # bs 4000 9
        # assert 1==2
        
        point_cloud = point_cloud.to(torch.float32)[:,:,:3]
        self.vision_tower = self.vision_tower.to(torch.float32)

        enc_xyz, enc_features, box_predictions, hidden_states = self.vision_tower(point_cloud)

        # return_feature = box_predictions['hidden_states'][self.select_layer]
        hidden_states = hidden_states[-2]
        # reference_point  = box_predictions['reference_point']
        # print("hidden_states", hidden_states.shape) # [8, 256, 256]
        # print("enc_features", enc_features.shape) # [1024, 8, 256]
        # assert 1==2
        # enc_xyz = box_predictions['seed_xyz'].clone
        # print("refine boxes", reference_point.shape) # [8, 4096, 8, 3]
        # print("enc_xyz", enc_xyz.shape) # [8, 256, 4096]
        # assert 1==2
        # enc_xyz = enc_xyz.permute(0, 2, 1)
        enc_features = enc_features.permute(1, 0, 2)
        
        # print(enc_features.shape) # bs 256 4096 bs dim num_points
        # print(enc_xyz.shape) # bs 256 4096 bs dim num_points
        # print(box_predictions['hidden_states'][self.select_layer].shape)

        # assert 1==2

        return box_predictions, enc_features, hidden_states
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
    
    @property
    def data_type(self):
        return 'pcl'

if __name__ == "__main__":
    print("loading model files !!")
    ckpt = torch.load("/13390024681/3D/3D-LLM/model_zoo/scannet_masked_ep720.pth")
    print(ckpt.keys())