import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers
from lavis.models.fastchat.model import load_model, get_conversation_template, add_model_args
# import model
# from transformers import LlamaTokenizer
# from transformers import LlamaForCausalLM

# from lavis.common.registry import registry
# from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

# transformers_version = version.parse(transformers.__version__)
# assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
# from transformers import LlamaTokenizer
# from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM

def generate_context(llm_path, device, temperature):
    # transformers_version = version.parse(transformers.__version__)
    # assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
    # from transformers import LlamaTokenizer, AutoTokenizer, AutoConfig
    # from transformers import AutoModelForCausalLM
    
    # llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, truncation_side="left")
    # # llm_model = LlamaForCausalLM.from_pretrained(
    # #     llm_path, torch_dtype=torch.float16
    # # )
    # config = AutoConfig.from_pretrained(
    #         llm_path,
    #         low_cpu_mem_usage=True,
    #         trust_remote_code=True,
    #     )
    # llm_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # llm_model.to(device)
    # llm_model.eval()
    # prompt = "tell me who you are"
    
    # llm_tokens = llm_tokenizer(
    #         prompt,
    #         padding="longest",
    #         return_tensors="pt"
    # ).to(device)
    # inputs_embeds = llm_model.get_input_embeddings()(llm_tokens.input_ids)
    # inputs_embeds = torch.cat([inputs_embeds], dim=1).to(device)
    # attention_mask = torch.cat([llm_tokens.attention_mask], dim=1).to(device)
    
    # outputs = llm_model.generate(
    #     inputs_embeds=inputs_embeds,
    #     attention_mask=attention_mask,
    #     do_sample=False,
    #     top_p=1.0,
    #     temperature=1.0,
    #     num_beams=5,
    #     max_length=256,
    #     min_length=1,
    #     # eos_token_id=self.eos_token_id,
    #     repetition_penalty=1.5,
    #     length_penalty=1,
    #     num_return_sequences=1,
    # )
    
    # output_text = llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # print(output_text)
    model, tokenizer = load_model(
        llm_path,
        device=device,
        num_gpus=1,
        max_gpu_memory=None,
        load_8bit=False,
        cpu_offloading=False,
        revision="main",
        debug=False,
    )

    # Build the prompt with a conversation template
    msg = "The following is the description provided:'The room has a computer and a printer sitting on a countertop. There is also a desk with a computer monitor and a keyboard on it. The room appears to be a workspace or office area, with various electronic devices and equipment present.'. Please judge based on the information provided that you can accurately answer the following questions: Is the computer closed or open? You only need to answer 'Yes' or 'No'"
    conv = get_conversation_template(llm_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    print("inputs shape",inputs.input_ids.shape)
    
    output_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 1e-5 else False,
        temperature=temperature,
        repetition_penalty=1.0,
        max_new_tokens=512,
    )
    
    print("output_ids", output_ids.shape)
    
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    # Print results
    print(f"{conv.roles[0]}: {msg}")
    print(f"{conv.roles[1]}: {outputs}")
    if 'No' in outputs:
        print(" he cannot")
    elif 'Yes' in outputs:
        print("he can")
    else:
        print("not judgeable")

if __name__ == "__main__":
    model_path = "/13390024681/3D/Create_3D_Data/model_zoo/vicuna-13b-v1.5"
    device = 'cuda'
    generate_context(model_path, device, temperature=0.1)