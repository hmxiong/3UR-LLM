import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

def check_file_exists(ann, pc_feat_root):
    file_path = os.path.join(pc_feat_root, ann["scene_id"] + ".pt")
    if os.path.exists(file_path):
        return ann
    else:
        print(file_path)
        return None

class YourClass:
    def __init__(self, annotation_root="/home/syc/yichao_blob_2/code/new_3dstyle_lm/data/haomiao_data/3D_LLM/ScanQA_v1.0_train.json"):
        self.annotation = json.load(open(annotation_root, "r"))
        # self.pc_feat_root = pc_feat_root
        self.pc_feat_root = "examples/voxelized_features_sam_nonzero_preprocess"  
        self.voxel_root = "examples/voxelized_voxels_sam_nonzero_preprocess"
    def process_annotations(self):
        # 使用 ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=30) as executor:
            # 创建并跟踪未来的任务
            future_to_ann = {executor.submit(check_file_exists, ann, self.pc_feat_root): ann for ann in self.annotation}

            self.annotation = []
            for future in as_completed(future_to_ann):
                result = future.result()
                if result is not None:
                    self.annotation.append(result)
                # 打印进度
                print(f"{len(self.annotation)}/{len(future_to_ann)} completed")
        with open("/home/syc/yichao_blob_2/code/new_3dstyle_lm/data/haomiao_data/3D_LLM/ScanQA_v1.0_train_filter.json", "w") as file:
            json.dump(self.annotation, file, indent=4) #_filter

# 使用示例
your_instance = YourClass()
your_instance.process_annotations()