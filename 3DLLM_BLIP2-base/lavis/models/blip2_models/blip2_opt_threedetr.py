import os
import logging
import json

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.v_detr_encoder import VDETRVisionTower
from lavis.models.threedetr_encoder import ThreeDETRVisionTower
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer


@registry.register_model("blip2_3detr_opt")
class Blip2ThreeDETR_OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt1.3b": "configs/models/blip2/blip2_pretrain_opt1.3b.yaml",
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_question_len=32,
        max_answer_len=32,
        use_3d=False,
        threed_encoder=None,
        status=None,
        train_embeddings=False,
        use_cot=False,
        freeze_q_former=False
    ):
        super().__init__()
        logging.info("Using 3DETR as Backbone !!")
        logging.info("LLM: {}".format(opt_model))
        logging.info("We need to concate 3DETR queries and Q-Former Queries !!")
        logging.info("Qurey number:{}".format(num_query_token))
        logging.info("max_question_len:{}".format(max_question_len))
        logging.info("max_answer_len:{}".format(max_answer_len))
        
        self.tokenizer = self.init_tokenizer()
        self.query_text_output = True # whether to use instruction to q_former
        
        self.use_3d = use_3d
        self.use_cot = use_cot
        self.num_query_token = num_query_token
        
        if status is not None:
            logging.info("the status of the model is {}".format(status))
            self.status = status
        else:
            logging.info("the status of the model is None")
        
        if self.status == 'finetune':
            if self.use_cot:
                logging.info("use chain of thought during inference !!")
                with open("/13390024681/3D/3D-LLM/data/all_captions_refined.json") as f:
                    self.cot_prompts = json.load(f)
                self.cot_instruction = 'Based on the description below:{description}. Please provide answer to the questions:'
            else:
                self.cot_instruction = None
        else:
            self.cot_instruction = None

        assert threed_encoder is not None
        
        self.visual_encoder = self.init_pcl_encoder(pretrain=True, store_path=threed_encoder)
        self.vision_proj = nn.Linear(self.visual_encoder.hidden_size, 1408)
        self.enc_vision_proj = nn.Linear(self.visual_encoder.hidden_size, 1408)
        self.pcl_query_proj = nn.Linear(self.visual_encoder.hidden_size, 768)
        
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408, use_3d=self.use_3d) # this must be 1408 dim 
        if not self.query_text_output:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            # print("len(self.tokenizer)",len(self.tokenizer))
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
            # assert 1==2
        self.Qformer.cls = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)

        location_tokens = []
        # for i in range(32768):  
        #     location_tokens.append("<loc%d>" % i)  
        # self.opt_tokenizer.add_special_tokens({"additional_special_tokens": location_tokens})

        self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
        # self.opt_model.resize_token_embeddings(len(self.opt_tokenizer))
        
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        
        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]
        # self.opt_model.get_output_embeddings().requires_grad_(True)
        # self.opt_model.get_input_embeddings().requires_grad_(True)

        self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, self.opt_model.config.hidden_size)


        self.max_txt_len = max_question_len
        self.max_output_txt_len = max_answer_len
        self.prompt = ""
        self._lemmatizer = None

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    
    def init_pcl_encoder(self, pretrain=True, store_path="../model_zoo/VDETR/", num_token=256, device="cpu"):
        if pretrain and not os.path.exists(store_path):
            raise ValueError(f"3DETR pretrained model not found at [{store_path}]!")  # 
        model = ThreeDETRVisionTower(store_path, select_layer=-2)
        return model.to(device)

    def forward(self, samples):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            pc_embeds = samples["pc_feat"]

        # enc_xyz, enc_features, pcl_feature = self.visual_encoder(pc_embeds)
        # pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        # enc_features = self.enc_vision_proj(enc_features) # bs 1024 256 # maypbe this has too many datasets
        # all_pcl_feature = torch.cat([pcl_feature, enc_features], dim=1) # bs 4352 1408
        # all_pcl_feature = enc_features
        box_predictions, enc_features, pcl_feature = self.visual_encoder(pc_embeds)
        values, index = box_predictions["outputs"]["objectness_prob"].topk(self.num_query_token, dim=1, largest=True, sorted=True)
        chosen_pcl_query = pcl_feature.index_select(dim=1, index=index[0])
        chosen_pcl_query = self.pcl_query_proj(chosen_pcl_query)
        pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        enc_features = self.enc_vision_proj(enc_features) # bs 1024 256 # maypbe this has too many datasets
        all_pcl_feature = torch.cat([pcl_feature, enc_features], dim=1) # bs 4352 1408

        image_atts = torch.ones(all_pcl_feature.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
        
        query_tokens = self.query_tokens.expand(all_pcl_feature.shape[0], -1, -1)  # 768

        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]
        
        if self.cot_instruction is not None: # add chain of thought prompt
            cot_text_input = []
            for question, scene_id in zip(text_input, samples['scene_id']):
                cot_prompt = self.cot_instruction.format(description=self.cot_prompts[scene_id])
                input = cot_prompt + question
                cot_text_input.append(input)
                # print(input)
            text_input = cot_text_input
        
        if self.query_text_output: # add instruction to Q-Former
            text_Qformer = self.tokenizer(
                text_input,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(all_pcl_feature.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)
            
            # if self.use_3d:
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=all_pcl_feature,
                encoder_attention_mask=image_atts,
                return_dict=True,
                reference_point=None,
                enc_xyz=None
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=all_pcl_feature,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        # print("query out :", query_output.last_hidden_state.shape) # [2, 32, 768]
        # print("query token", query_tokens.shape) # [2, 32, 768] 选择的是使用32个query来浓缩信息
        # assert 1==2
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)


        text_input = [t for t in text_input]

        # answers = [(t + a) for (t, a) in zip(text_input, samples["answer"])]
        answers = [t for t in samples["answer"]]
        
        # print("answers", answers)
        # print("text_input", text_input)
        text_input_tokens = []
        text_output_tokens = []
        
        self.opt_tokenizer.padding_side = "right"
        self.opt_tokenizer.truncation_side = 'left'
        text_input_tokens = self.opt_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(all_pcl_feature.device)

        self.opt_tokenizer.truncation_side = 'right'
        text_output_tokens = self.opt_tokenizer(
            [t + self.opt_tokenizer.eos_token for t in answers],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(all_pcl_feature.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )
        
        # print("llm_tokens", llm_tokens['input_ids'], llm_tokens['input_ids'].shape, llm_tokens['attention_mask'])
         # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.opt_tokenizer.pad_token_id, -100
        )
        
        # print("input_part_targets_len", input_part_targets_len)
        
        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len): # for all question
            targets[i][:l] = -100
        
        # do not apply loss to the text input (i.e., instruction)
        # for i, l in enumerate(input_part_targets_len):
            # targets[i][:l] = -100
        
        # print("targets", targets[:,-25:-10])
        
        # targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100)

        empty_targets = torch.ones(atts_opt.size(), dtype=torch.long).to(pc_embeds.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        # print("targets", targets)
        # assert 1==2

        inputs_embeds = self.opt_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                labels=targets,
            )
        loss = outputs.loss
        
        return {"loss": loss}

    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            pc_embeds = samples["pc_feat"]
        
        enc_xyz, enc_features, pcl_feature = self.visual_encoder(pc_embeds)
        pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        enc_features = self.enc_vision_proj(enc_features) # bs 4096 1048 # maypbe this has too many datasets
        all_pcl_feature = torch.cat([pcl_feature, enc_features], dim=1) # bs 4352 1408
        # all_pcl_feature = enc_features

        image_atts = torch.ones(all_pcl_feature.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
        
        query_tokens = self.query_tokens.expand(all_pcl_feature.shape[0], -1, -1)  # 768

        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]
        
        if self.query_text_output: # add instruction to Q-Former
            text_Qformer = self.tokenizer(
                text_input,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(all_pcl_feature.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=all_pcl_feature,
                encoder_attention_mask=image_atts,
                return_dict=True,
                reference_point=None,
                enc_xyz=enc_xyz
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=all_pcl_feature,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        prompt = [prompt] * pc_embeds.size(0)

        opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(pc_embeds.device)
        input_ids = opt_tokens.input_ids
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        if use_nucleus_sampling:
            query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
            num_beams = 1
        else:
            query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

        outputs = self.opt_model.generate(
            input_ids=input_ids,
            query_embeds=query_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )

        prompt_length = opt_tokens.input_ids.shape[1]
        output_text = self.opt_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=False)
        output_text = [text.strip() for text in output_text]
        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            pc_embeds = samples["pc_feat"]
        
        box_predictions, enc_features, pcl_feature = self.visual_encoder(pc_embeds)
        values, index = box_predictions["outputs"]["objectness_prob"].topk(self.num_query_token, dim=1, largest=True, sorted=True)
        chosen_pcl_query = pcl_feature.index_select(dim=1, index=index[0])
        chosen_pcl_query = self.pcl_query_proj(chosen_pcl_query)
        pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        enc_features = self.enc_vision_proj(enc_features) # bs 4096 1048 # maypbe this has too many datasets
        all_pcl_feature = torch.cat([pcl_feature, enc_features], dim=1) # bs 4352 1408
        # all_pcl_feature = enc_features

        image_atts = torch.ones(all_pcl_feature.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
        
        # print("query_tokens", self.query_tokens.shape)
        
        query_tokens = self.query_tokens.expand(all_pcl_feature.shape[0], -1, -1)  # 768

        # print("query_tokens", query_tokens.shape)
        
        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]
        
        if self.cot_instruction is not None:
            cot_text_input = []
            for question, scene_id in zip(text_input, samples['scene_id']):
                cot_prompt = self.cot_instruction.format(description=self.cot_prompts[scene_id])
                input = cot_prompt + question
                cot_text_input.append(input)
                # print(input)
            text_input = cot_text_input
        
        # print("len text_input", text_input)
            
        if self.query_text_output: # add instruction to Q-Former
            text_Qformer = self.tokenizer(
                text_input,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(all_pcl_feature.device)
            # print("query_tokens", query_tokens.shape)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=all_pcl_feature,
                encoder_attention_mask=image_atts,
                return_dict=True,
                reference_point=None,
                enc_xyz=None
            )
            # print("query_tokens", query_tokens.shape) # 43
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=all_pcl_feature,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        # print("query_output", query_output.last_hidden_state.shape)
        
        inputs_opt = self.opt_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:]) # bs 43
        # print("inputs_opt", inputs_opt.shape)
        
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(pc_embeds.device) # 43

        input_tokens = self.opt_tokenizer(text_input, padding="longest", return_tensors="pt").to(pc_embeds.device) # 10
        input_ids = input_tokens.input_ids
        
        
        inputs_embeds = self.opt_model.get_input_embeddings()(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds],dim=1)
        
        attention_mask = torch.cat([atts_opt, input_tokens.attention_mask], dim=1) # 53  
          
        # if use_nucleus_sampling:
        #     input_embeds = inputs_opt.repeat_interleave(1, dim=0)
        # # num_beams = 1
        # else:
        # input_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

        # print("input_tokens",  input_tokens.input_ids.shape) # 10
        # print("input_ids",     input_ids.shape) # 10 
        # print("inputs_embeds", inputs_embeds.shape) # 53
        # print("attention_mask",attention_mask.shape) # 53
        
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with self.maybe_autocast(dtype=torch.float16):
            outputs = self.opt_model.generate(
                # input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
                # repetition_penalty=repetition_penalty,
                eos_token_id=self.eos_token_id,
            )

            prompt_length = input_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=False)
            print(output_text)
            output_text = [text.strip() for text in output_text]
            
        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_question_len = cfg.get("max_question_len", 20)
        max_answer_len = cfg.get("max_answer_len", 50)

        use_3d = cfg.get("use_3d", False)
        threed_encoder = cfg.get("threed_encoder", "")
        pretrained = cfg.get("pretrained", "")
        use_pretrained = cfg.get("use_pretrained", False)
        
        status = cfg.get("status", "")
        use_cot = cfg.get("use_cot", False)
        freeze_q_former = cfg.get("freeze_q_former", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_question_len=max_question_len,
            max_answer_len=max_answer_len,
            use_3d=use_3d,
            threed_encoder=threed_encoder,
            status=status,
            use_cot=use_cot,
            freeze_q_former=freeze_q_former
        )
        # print(model.Qformer)
        # assert 1==2
        # print("pratrained", pretrained)
        # assert 1==2/
        # if pretrained is not None:
        if use_pretrained:
            print("Loading pratrained check_point form {}".format(pretrained))
            model.load_checkpoint_from_config(cfg)

        return model
