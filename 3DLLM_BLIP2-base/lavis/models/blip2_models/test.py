import io
import os
import random
import json

import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.nn.utils import rnn
import torch.nn as nn
import numpy as np
from torchvision import transforms

from transformers import BertTokenizer
from Qformer import BertConfig, BertLMHeadModel

def init_Qformer(num_query_token, vision_width, cross_attention_freq=2):
    tokenizer = BertTokenizer.from_pretrained("/13390024681/3D/3D-LLM/data/bert-base-uncased")
    encoder_config = BertConfig.from_pretrained("/13390024681/3D/3D-LLM/data/bert-base-uncased")
    encoder_config.encoder_width = vision_width
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    # Qformer = BertLMHeadModel.load_state_dict("/13390024681/3D/3D-LLM/data/bert-base-uncased/pytorch_model.bin", state_dict=False)
    Qformer = BertLMHeadModel.from_pretrained("/13390024681/3D/3D-LLM/data/bert-base-uncased", config=encoder_config)
    # print(Qformer)
    assert 1==2
    query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens

def main():
    init_Qformer(32, 512)
    pass

if __name__ == "__main__":
    main()