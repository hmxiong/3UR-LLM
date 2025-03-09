import os
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.v_detr_encoder import VDETRVisionTower
from lavis.models.threedetr_encoder import ThreeDETRVisionTower
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
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
        max_txt_len=32,
        use_3d=False,
        threed_encoder=None,
    ):
        super().__init__()
        print("Using 3DETR as Backbone !!")

        self.tokenizer = self.init_tokenizer()
        self.query_text_output = True # whether to use instruction to q_former
        self.use_3d = use_3d

        assert threed_encoder is not None
        
        self.visual_encoder = self.init_pcl_encoder(pretrain=True, store_path=threed_encoder)
        self.vision_proj = nn.Linear(self.visual_encoder.hidden_size, 1408)
        self.enc_vision_proj = nn.Linear(self.visual_encoder.hidden_size, 1408)
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
        for i in range(32768):  
            location_tokens.append("<loc%d>" % i)  
        self.opt_tokenizer.add_special_tokens({"additional_special_tokens": location_tokens})

        self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer))
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]
        # self.opt_model.get_output_embeddings().requires_grad_(True)
        # self.opt_model.get_input_embeddings().requires_grad_(True)

        self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, self.opt_model.config.hidden_size)


        self.max_txt_len = max_txt_len
        self.prompt = ""
        self._lemmatizer = None

    def init_pcl_encoder(self, pretrain=True, store_path="../model_zoo/VDETR/", num_token=256, device="cpu"):
        if pretrain and not os.path.exists(store_path):
            raise ValueError(f"3DETR pretrained model not found at [{store_path}]!")  # 
        model = ThreeDETRVisionTower(store_path, select_layer=-2)
        return model.to(device)

    def forward(self, samples):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            pc_embeds = samples["pc_feat"]

        enc_xyz, enc_features, pcl_feature = self.visual_encoder(pc_embeds)
        pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        enc_features = self.enc_vision_proj(enc_features) # bs 1024 256 # maypbe this has too many datasets
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


        text = [t for t in text_input]

        answers = [(t + a + "\n") for (t, a) in zip(text, samples["answer"])]

        opt_tokens = self.opt_tokenizer(
            answers,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(pc_embeds.device)

        idxes = (torch.where(opt_tokens.input_ids == 1948)[1] + 2).cpu().numpy().tolist()

        output_tokens = self.opt_tokenizer(
            answers,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(pc_embeds.device)

        targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100)

        empty_targets = torch.ones(atts_opt.size(), dtype=torch.long).to(pc_embeds.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

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
        use_nucleus_sampling=False,
        max_len=200,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs,
    ):
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            pc_embeds = samples["pc_feat"]

        # pcl_feature, enc_features = self.visual_encoder(pc_embeds)
        # pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        # enc_features = self.enc_vision_proj(enc_features) # bs 4096 1048
        # all_pcl_feature = torch.cat([pcl_feature, enc_features], dim=1) # bs 4352 1408
        
        # image_atts = torch.ones(all_pcl_feature.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)
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

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        text = [t for t in text_input]

        input_tokens = self.opt_tokenizer(text, padding="longest", return_tensors="pt").to(pc_embeds.device)
        input_ids = input_tokens.input_ids
        encoder_atts = torch.cat([atts_opt, input_tokens.attention_mask], dim=1)

        if use_nucleus_sampling:
            input_embeds = inputs_opt.repeat_interleave(1, dim=0)
        # num_beams = 1
        else:
            input_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with self.maybe_autocast(dtype=torch.float16):
            outputs = self.opt_model.generate(
                input_ids=input_ids,
                query_embeds=input_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                # eos_token_id=self.eos_token_id,
            )

            prompt_length = input_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=False)
            
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
        max_txt_len = cfg.get("max_txt_len", 32)

        use_3d = cfg.get("use_3d", False)
        threed_encoder = cfg.get("threed_encoder", "")
        pretrained = cfg.get("pretrained", "")
        use_pretrained = cfg.get("use_pretrained", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            use_3d=use_3d,
            threed_encoder=threed_encoder,
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
