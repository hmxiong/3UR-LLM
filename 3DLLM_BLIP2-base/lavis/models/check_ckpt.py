import torch

if __name__ == "__main__":
    print("loading model files !!")
    ckpt = torch.load("/13390024681/3D/3D-LLM/model_zoo/threedetr_scannet_750.pth")
    print(ckpt.keys())
    print(ckpt['args'].model_name)