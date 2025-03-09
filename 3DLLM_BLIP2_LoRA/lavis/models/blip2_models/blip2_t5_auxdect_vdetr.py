import logging
import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.v_detr_encoder import VDETRVisionTower
from positional_encodings.torch_encodings import PositionalEncoding1D
from .VDETR.helpers import GenericMLP
from .VDETR.dataops import DatasetConfig
from .VDETR.criterion import build_criterion
from functools import partial
import math
import torch.nn.functional as F


@registry.register_model("blip2_t5_auxdect_vdetr")
class Blip2T5_auxdect_vdetr(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5_auxdect", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
        "pretrain_flant5xl_auxdect": "configs/models/blip2/blip2_pretrain_flant5xl_auxdect.yaml",
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
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        encoder_ckpt_path=None,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        self.visual_encoder = self.init_pcl_encoder(pretrain=True, store_path="/13390024681/LLaVA/model_zoo/vdetr/checkpoint_best_73.pth")
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        
        self.vision_proj = nn.Linear(self.visual_encoder.hidden_size, 1408)
        self.enc_vision_proj = nn.Linear(self.visual_encoder.hidden_size, 1408)
        
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None #直接输入的learn query的embeding
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)

        location_tokens = []
        for i in range(32768):  
            location_tokens.append("<loc%d>" % i)  
        self.t5_tokenizer.add_special_tokens({"additional_special_tokens": location_tokens})

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)

        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data

        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)

        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 256, 1408 // 3) # 将voxel的 int_index 类比 b,seq_len,dim中的seq_len；生成sin、cos的embeding
        self.pos_embedding = pos_model(x).squeeze().cuda()

        self.max_txt_len = max_txt_len
        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.query_text_output = False

        # prepare detection head
        self.initial_detection_head()
        ckpt = torch.load(encoder_ckpt_path)
        detaction_args = ckpt["args"]
        self.dataset_config = DatasetConfig()
        # prepare criterion
        self.criterion = build_criterion(detaction_args, self.dataset_config)
        self.criterion.loss_functions.pop('loss_angle')
        self.criterion.loss_weight_dict.pop('loss_angle_cls_weight')
        self.criterion.loss_weight_dict.pop('loss_angle_reg_weight')
        # prepare learnable position
        self.query_pos = nn.Parameter(torch.zeros(1, num_query_token, 6))
        self.query_pos[...,0:3].data.normal_(mean=0.0, std=6)
        nn.init.constant_(self.query_pos[...,3:6].data, 1.0)
        
        
        
    def init_pcl_encoder(self, pretrain=True, store_path="../model_zoo/VDETR/", num_token=256, device="cpu"):
        if pretrain and not os.path.exists(store_path):
            raise ValueError(f"VDETR pretrained model not found at [{store_path}]!")  # 目前需要再进一步等待checkpoint给我发过来
        model = VDETRVisionTower(store_path, select_layer=-2)
        return model.to(device)
     
    def forward(self, samples):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            pc_embeds = samples["pc_feat"]

        pcl_feature, enc_features = self.visual_encoder(pc_embeds)
        pcl_feature = self.vision_proj(pcl_feature) # project pcl dimension to llm dimension
        enc_features = self.enc_vision_proj(enc_features) # bs 4096 1048
        all_pcl_feature = torch.cat([pcl_feature, enc_features], dim=1) # bs 4352 1408

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
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=all_pcl_feature,
                encoder_attention_mask=image_atts,
                return_dict=True,
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
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(all_pcl_feature.device)


        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=400,
                return_tensors="pt",
            ).to(all_pcl_feature.device)
            output_tokens = self.t5_tokenizer(
                samples["answer"],
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(all_pcl_feature.device)
            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, n in enumerate(samples["n_answers"]):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
                batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
                batch_atts_t5 += [atts_t5[b]] * n
                batch_inputs_t5 += [inputs_t5[b]] * n

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            # return {"loss": loss}
            
            
            # aux detection loss:
            box_features = outputs['encoder_last_hidden_state'][:,:self.query_tokens.shape[1]]
            query_pos = self.query_pos.repeat(box_features.shape[0],1,1)
            box_prediction = self.predict_box(box_features,query_pos)
            box_pred_dict = {"outputs": box_prediction}
            
            #prepare gt 
            
            detect_info = samples['detect_info']
            max_length = max(len(t) for t in detect_info)
            box_tensors = []
            box_presents = []
            for t in detect_info:
                box_tensor = torch.tensor(t,device=box_features.device,dtype=box_features.dtype)
                # print(box_tensor)
                padding_size = max_length - box_tensor.size(0)
                # 在右侧添加padding
                
                padded_tensor = F.pad(box_tensor,(0,0,0,padding_size), 'constant', 1)
                box_tensors.append(padded_tensor)
                
                box_present = [1]*box_tensor.size(0) + [0]*padding_size
                box_presents.append(torch.tensor(box_present,device=box_features.device,dtype=box_features.dtype))
            box_tensors = torch.stack(box_tensors, dim=0)
            box_presents = torch.stack(box_presents, dim=0)
            
            targets_boxes = {}          
            class_offset = 1
            targets_boxes['gt_box_sem_cls_label'] = box_tensors[...,0].long().contiguous()
            targets_boxes["gt_box_centers"] = box_tensors[...,class_offset+1:class_offset+4].contiguous()
            targets_boxes["gt_box_sizes"] = box_tensors[...,class_offset+3:class_offset+6].contiguous()
            targets_boxes["gt_box_angles"] = torch.zeros(box_features.shape[0], targets_boxes["gt_box_centers"].shape[1]).to(device=targets_boxes["gt_box_centers"].device,dtype=targets_boxes["gt_box_centers"].dtype)

            targets_boxes["gt_box_present"] = box_presents
            targets_boxes["gt_box_corners"] = self.dataset_config.box_parametrization_to_corners(targets_boxes["gt_box_centers"],targets_boxes["gt_box_sizes"],targets_boxes["gt_box_angles"])
            box_losses, box_loss_dict = self.criterion(box_pred_dict, targets_boxes)
            
            
            loss = loss + box_losses*0.4
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
        image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * pc_embeds.size(0)
        else:
            assert len(prompt) == pc_embeds.size(0), "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(prompt, padding="longest", return_tensors="pt").to(pc_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
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

        with torch.cuda.amp.autocast(dtype=torch.float32):
            pc = samples["pc"].long()
            all_pcs = torch.zeros((pc_embeds.shape))
            for j in range(pc.shape[0]):
                pcs = []
                for i in range(3):
                    pc_i = pc[j][:, i]
                    pcs.append(self.pos_embedding[pc_i])
                pcs = torch.cat(pcs, -1)
                all_pcs[j][:, :1407] = pcs
            all_pcs = all_pcs.cuda()

        pc_embeds = pc_embeds + 0.01 * all_pcs
        image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        prompt = self.prompt

        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(text_input, padding="longest", return_tensors="pt").to(pc_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        num_beams = 1
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.float32):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                # for description, also use repetition penalty = 1.5
            )
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=False)

        if self._apply_lemmatizer:
            output_text_new = self._lemmatize(output_text)
            output_text = output_text_new
            # if output_text_new!=output_text:
            #    print("old: %s, new: %s\n"%(output_text, output_text_new))
        # import pdb; pdb.set_trace()
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
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        encoder_ckpt_path = cfg.get("encoder_ckpt_path", None)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            encoder_ckpt_path=encoder_ckpt_path,
        )
        model.load_checkpoint_from_config(cfg)

        return model
    
    
    def initial_detection_head(self):
        mlp_func = partial(
            GenericMLP,
            # norm_fn_name='bn1d',
            activation='relu',
            use_conv=True,
            hidden_dims=[self.t5_model.config.hidden_size, self.t5_model.config.hidden_size],
            input_dim=self.t5_model.config.hidden_size,
        )
        self.center_head = mlp_func(output_dim=3)
        self.size_head = mlp_func(output_dim=3)
        self.class_head = mlp_func(output_dim=18)#hacking for Scannet
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.layers[-1].bias.data = torch.ones(18) * bias_value
        nn.init.constant_(self.center_head.layers[-1].weight.data, 0.0)
        nn.init.constant_(self.center_head.layers[-1].bias.data, 0.0)
        nn.init.constant_(self.size_head.layers[-1].weight.data, 0.0)
        nn.init.constant_(self.size_head.layers[-1].bias.data, 0.0)
    
    def predict_box(self,box_seq_features,query_pos,):
        
        pre_box_center_unnormalized = query_pos[...,0:3]
        pre_box_size_unnormalized = query_pos[...,3:6]
        npre_box = query_pos.shape[-2]
        #prepare_for_class
        box_class_logits = self.class_head(box_seq_features.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        semcls_prob = box_class_logits
        objectness_prob = box_class_logits.sigmoid().max(dim=-1)[0]#focal loss
        
        #prepare_for_size 
        # pre_box_size_unnormalized = torch.ones_like(query_xyz)
        box_seq_size_reg = self.size_head(box_seq_features.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        box_seq_size_unnormlized = torch.exp(box_seq_size_reg) * pre_box_size_unnormalized
                            
        #prepare_for_center
        # pre_box_center_unnormalized = query_xyz
        box_seq_center_reg = self.center_head(box_seq_features.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        box_seq_center_unnormlized = box_seq_center_reg * pre_box_size_unnormalized + pre_box_center_unnormalized
        
        #prepare_for_angle and conners
        box_angle = torch.zeros(box_seq_center_reg.shape[0], npre_box).to(device=box_seq_center_unnormlized.device,dtype=box_seq_center_unnormlized.dtype)
        
        box_corners = self.dataset_config.box_parametrization_to_corners(box_seq_center_unnormlized,box_seq_size_unnormlized,box_angle)
        
        box_prediction = {
            "sem_cls_logits": box_class_logits.contiguous(),
            "center_unnormalized": box_seq_center_unnormlized.contiguous(),
            "size_unnormalized": box_seq_size_unnormlized.contiguous(),
            "angle_continuous": box_angle.contiguous(),
            "objectness_prob": objectness_prob.contiguous(),
            "sem_cls_prob": semcls_prob.contiguous(),
            "box_corners": box_corners.contiguous(),
            "pre_box_center_unnormalized":pre_box_center_unnormalized.contiguous(),
            "center_reg":box_seq_center_reg.contiguous(),
            "pre_box_size_unnormalized":pre_box_size_unnormalized.contiguous(),
            "size_reg":box_seq_size_reg.contiguous(),   
        }

        return box_prediction
                
    