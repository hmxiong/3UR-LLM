import os
import logging
import json

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
# from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.v_detr_encoder import VDETRVisionTower
from lavis.models.threedetr_encoder import ThreeDETRVisionTower
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from positional_encodings.torch_encodings import PositionalEncoding1D
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import GemmaForCausalLM, GemmaTokenizer, GemmaConfig
# tokenizer = AutoTokenizer.from_pretrained("/13390024681/3D/3D-LLM/model_zoo/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("/13390024681/3D/3D-LLM/model_zoo/gemma-2b", device_map="auto")

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))

@registry.register_model("blip2_3detr_gemma_concate")
class Blip2ThreeDETR_Gemma_Concate(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "gemma_2b": "configs/models/blip2/blip2_pretrain_gemma_2b.yaml",
        "gemma_7b": "configs/models/blip2/blip2_pretrain_gemma_7b.yaml",
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
        gemma_model="google/gemma_model",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        use_3d=False,
        threed_encoder=None,
        status=None,
        use_cot=False,
        use_lora=False
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        logging.info("Using 3DETR as Backbone !!")
        logging.info("Using Gemma as Language Model !!")
        
        if status is not None:
            logging.info("the status of the model is {}".format(status))
            self.status = status
        else:
            logging.info("the status of the model is None")
        
        if self.status == 'finetune':
            if self.use_cot:
                logging.info("use chain of thought during training !!")
                with open("/13390024681/3D/3D-LLM/data/all_captions_refined.json") as f:
                    self.cot_prompts = json.load(f)
                self.cot_instruction = 'Based on the description below:{description}. Please provide answer to the questions:'
            else:
                self.cot_instruction = None
        else:
            self.cot_instruction = None
            
        self.tokenizer = self.init_tokenizer()
        self.query_text_output = True # whether to use instruction to q_former
        self.use_3d = use_3d
        self.num_query_token = num_query_token
        self.use_cot = use_cot
        
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

        self.gemma_tokenizer = GemmaTokenizer.from_pretrained(gemma_model)
        self.gemma_config = GemmaConfig.from_pretrained(gemma_model)
        self.gemma_model = GemmaForCausalLM.from_pretrained(gemma_model)

        for name, param in self.gemma_model.named_parameters():
            param.requires_grad = False
            
        self.gemma_proj = nn.Linear(self.Qformer.config.hidden_size, self.gemma_model.config.hidden_size)

        self.max_txt_len = max_txt_len
        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def init_pcl_encoder(self, pretrain=True, store_path="../model_zoo/VDETR/", num_token=256, device="cpu"):
        if pretrain and not os.path.exists(store_path):
            raise ValueError(f"3DETR pretrained model not found at [{store_path}]!")  # 目前需要再进一步等待checkpoint给我发过来
        model = ThreeDETRVisionTower(store_path, select_layer=-2)
        return model.to(device)

    def forward(self, samples):
        with torch.cuda.amp.autocast(dtype=torch.float32):
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
        
        query_tokens = self.query_tokens.expand(all_pcl_feature.shape[0], -1, -1)  # 768

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
        
        # print(text_input)
        
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
        inputs_gemma = self.gemma_proj(query_output.last_hidden_state)
        atts_gemma = torch.ones(inputs_gemma.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
        
        self.gemma_tokenizer.padding_side = "right"
        self.gemma_tokenizer.truncation_side = 'left'
        
        text = [t for t in text_input]

        # answers = [(t + a + "\n") for (t, a) in zip(text, samples["answer"])] # 自回归形式将问题和回答先拼接到一起
        text_output = [t for t in samples["answer"]]
        
        # print("text", text)
        # print("text_output", text_output)
        # print("answers", answers)
        text_input_tokens = self.gemma_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(all_pcl_feature.device)
        
        # self.gemma_tokenizer.truncation_side = 'right'
        
        text_output_tokens = self.gemma_tokenizer(
            text_output,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(all_pcl_feature.device)
        
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )
        
        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.gemma_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
        
        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_gemma.size(), dtype=torch.long).to(all_pcl_feature.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        
        # gemma_tokens = self.gemma_tokenizer(
        #     answers,
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        # ).to(all_pcl_feature.device)

        # idxes = (torch.where(opt_tokens.input_ids == 1948)[1] + 2).cpu().numpy().tolist()

        # output_tokens = self.gemma_tokenizer(
        #     answers,
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(all_pcl_feature.device)

        # targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.gemma_tokenizer.pad_token_id, -100)

        # empty_targets = torch.ones(atts_gemma.size(), dtype=torch.long).to(pc_embeds.device).fill_(-100)
        # targets = torch.cat([empty_targets, targets], dim=1)
        
        # print("targets ", targets)
        
        inputs_embeds = self.gemma_model.model.embed_tokens(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_gemma, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_gemma, llm_tokens['attention_mask']], dim=1)
        # print("inputs_embeds", inputs_embeds)
        # print("attention_mask", attention_mask)
        # assert 1==2
        with self.maybe_autocast():
            outputs = self.gemma_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    
    @torch.no_grad()
    def generate(
        self,
        text_input,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=0.0,
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

        box_predictions, enc_features, pcl_feature = self.visual_encoder(pc_embeds)
        values, index = box_predictions["outputs"]["objectness_prob"].topk(self.num_query_token, dim=1, largest=True, sorted=True)
        chosen_pcl_query = pcl_feature.index_select(dim=1, index=index[0])
        chosen_pcl_query = self.pcl_query_proj(chosen_pcl_query)
        pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        enc_features = self.enc_vision_proj(enc_features) # bs 4096 1048 # maypbe this has too many datasets
        all_pcl_feature = torch.cat([pcl_feature, enc_features], dim=1) # bs 4352 1408
        # all_pcl_feature = enc_features

        image_atts = torch.ones(all_pcl_feature.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
        
        query_tokens = self.query_tokens.expand(all_pcl_feature.shape[0], -1, -1)  # 768

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
                enc_xyz=None
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=all_pcl_feature,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_gemma = self.gemma_proj(query_output.last_hidden_state)
        atts_gemma = torch.ones(inputs_gemma.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)

        input_tokens = self.gemma_tokenizer(text_input, padding="longest", return_tensors="pt").to(all_pcl_feature.device)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            inputs_embeds = self.gemma_model.model.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_gemma, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_gemma, input_tokens.attention_mask], dim=1)

            outputs = self.gemma_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            
            # output_text = self.gemma_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.gemma_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            # print("output_text", output_text)

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        use_nucleus_sampling=False,
        max_len=256,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        top_p=0.9,
        num_captions=1,
        temperature=1,
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs,
    ):
        # with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
        #     pc_embeds = samples["pc_feat"]

        # # pcl_feature, enc_features = self.visual_encoder(pc_embeds)
        # # pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        # # enc_features = self.enc_vision_proj(enc_features) # bs 4096 1048
        # # all_pcl_feature = torch.cat([pcl_feature, enc_features], dim=1) # bs 4352 1408
        
        # # image_atts = torch.ones(all_pcl_feature.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
        # enc_xyz, enc_features, pcl_feature = self.visual_encoder(pc_embeds)
        # pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        # enc_features = self.enc_vision_proj(enc_features) # bs 4096 1048 # maypbe this has too many datasets
        # all_pcl_feature = torch.cat([pcl_feature, enc_features], dim=1) # bs 4352 1408
        # # all_pcl_feature = enc_features

        # image_atts = torch.ones(all_pcl_feature.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
        
        # query_tokens = self.query_tokens.expand(all_pcl_feature.shape[0], -1, -1)  # 768

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
        
        output_text = self.generate(
            text_input,
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)
        
        return output_text
        # print(text_input)
        
        # if self.query_text_output: # add instruction to Q-Former
        #     text_Qformer = self.tokenizer(
        #         text_input,
        #         padding='longest',
        #         truncation=True,
        #         max_length=self.max_txt_len,
        #         return_tensors="pt",
        #     ).to(all_pcl_feature.device)
        #     query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
        #     Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask],dim=1)

        #     query_output = self.Qformer.bert(
        #         text_Qformer.input_ids,
        #         attention_mask=Qformer_atts,
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=all_pcl_feature,
        #         encoder_attention_mask=image_atts,
        #         return_dict=True,
        #         reference_point=None,
        #         enc_xyz=enc_xyz
        #     )
        # else:
        #     query_output = self.Qformer.bert(
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=all_pcl_feature,
        #         encoder_attention_mask=image_atts,
        #         return_dict=True,
        #     )

        # inputs_gemma = self.gemma_proj(query_output.last_hidden_state)
        # atts_gemma = torch.ones(inputs_gemma.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        # text = [t for t in text_input]

        # input_tokens = self.gemma_tokenizer(text, padding="longest", return_tensors="pt").to(pc_embeds.device)
        # input_ids = input_tokens.input_ids
        # encoder_atts = torch.cat([atts_gemma, input_tokens.attention_mask], dim=1)

        # with self.maybe_autocast():
        #     inputs_embeds = self.gemma_model.model.embed_tokens(input_tokens.input_ids)
        #     inputs_embeds = torch.cat([inputs_gemma, inputs_embeds], dim=1)
        #     attention_mask = torch.cat([atts_gemma, input_tokens.attention_mask], dim=1)

        #     outputs = self.gemma_model.generate(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         do_sample=use_nucleus_sampling,
        #         top_p=top_p,
        #         temperature=temperature,
        #         num_beams=num_beams,
        #         max_length=max_len,
        #         min_length=min_len,
        #         # eos_token_id=self.eos_token_id,
        #         repetition_penalty=repetition_penalty,
        #         length_penalty=length_penalty,
        #         num_return_sequences=num_captions,
        #     )
        #     outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        #     output_text = self.gemma_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #     output_text = [text.strip() for text in output_text]
        # return output_text

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
        gemma_model = cfg.get("gemma_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        use_3d = cfg.get("use_3d", False)
        threed_encoder = cfg.get("threed_encoder", "")
        pretrained = cfg.get("pretrained", "")
        use_pretrained = cfg.get("use_pretrained", False)
        status = cfg.get("status", "")
        use_cot = cfg.get("use_cot", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            gemma_model=gemma_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            use_3d=use_3d,
            threed_encoder=threed_encoder,
            status=status,
            use_cot=use_cot
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

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("/13390024681/3D/3D-LLM/model_zoo/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("/13390024681/3D/3D-LLM/model_zoo/gemma-2b-it", device_map="auto")

    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


