import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from positional_encodings.torch_encodings import PositionalEncoding1D
from .modeling_otter import extend_instance, _infer_decoder_layers_attr_name, VdetrLMMixin
from .VDETR import build_VDETR_llm, build_criterion
from functools import partial
from .VDETR.helpers import GenericMLP

@registry.register_model("blip2_t5_ottervdetr")
class Blip2T5_ottervdetr(Blip2Base):
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
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
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
        cross_attn_every_n_layers=4,
        encoder_ckpt_path="None",
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        
        self.num_query_token = num_query_token
        
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
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
        x = torch.zeros(1, 256, 1408 // 3)
        self.pos_embedding = pos_model(x).squeeze().cuda()

        self.max_txt_len = max_txt_len
        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        extend_instance(self.t5_model, VdetrLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(self.t5_model)
        self.t5_model.set_decoder_layers_attr_name(decoder_layers_attr_name)

        self.t5_model.init_otter(
            vision_args=self.vision_args,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            llm_hidden_dim=self.t5_model.config.hidden_size,
            dataset_config=self.visual_encoder.dataset_config,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.visual_encoder, self.vision_args, self.dataset_config = build_VDETR_llm(
            pretrain=True, store_path=encoder_ckpt_path, device=device
        )
        self.visual_encoder.only_encoder = True
        self.vision_matcher = build_criterion(self.vision_args, self.dataset_config)
        self.vision_matcher.loss_functions.pop('loss_angle')
        self.vision_matcher.loss_weight_dict.pop('loss_angle_cls_weight')
        self.vision_matcher.loss_weight_dict.pop('loss_angle_reg_weight')
        
        mlp_func = partial(
            GenericMLP,
            # norm_fn_name='bn1d',
            activation='relu',
            use_conv=True,
            hidden_dims=[self.llama_model.config.hidden_size, self.llama_model.config.hidden_size],
            input_dim=self.llama_model.config.hidden_size,
        )
        self.center_head = mlp_func(output_dim=3)
        self.size_head = mlp_func(output_dim=3)
        self.class_head = mlp_func(output_dim=18)#hacking for Scannet
        self.box_matching = self.args["box_matching"]
        self.box_losses_weight = self.args["box_losses_weight"]
        self.ignore_box_token = self.args["ignore_box_token"]
        import math
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.layers[-1].bias.data = torch.ones(18) * bias_value
        nn.init.constant_(self.center_head.layers[-1].weight.data, 0.0)
        nn.init.constant_(self.center_head.layers[-1].bias.data, 0.0)
        nn.init.constant_(self.size_head.layers[-1].weight.data, 0.0)
        nn.init.constant_(self.size_head.layers[-1].bias.data, 0.0)

    def forward(self, samples):
        with torch.cuda.amp.autocast(dtype=torch.float32):
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

        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)  # 768
        
        
        enc_fps_xyz = samples["pc"][:,:512]
        dec_fps_xyz = samples["pc"][:,:self.num_query_token]
        point_cloud_dims = [
            enc_fps_xyz[...,:3].min(dim=-2).values,
            enc_fps_xyz[...,:3].max(dim=-2).values,
        ]
        bs, npre_box, _ = dec_fps_xyz.shape
        ## all layer shared
        self.vision_locations_dict = {}
        self.vision_locations_dict["pre_center_unnormalized"] = dec_fps_xyz.to(self.t5_model.dtype)
        self.vision_locations_dict["pre_size_unnormalized"] = torch.ones_like(dec_fps_xyz).to(self.t5_model.dtype)
        self.vision_locations_dict["enc_xyz"] = enc_fps_xyz.to(self.t5_model.dtype)
        point_cloud_dims[0] = point_cloud_dims[0].to(self.t5_model.dtype)
        point_cloud_dims[1] = point_cloud_dims[1].to(self.t5_model.dtype)
        self.vision_locations_dict["point_cloud_dims"] = point_cloud_dims
        
        ## will refine different layer
        reference_center = dec_fps_xyz.clone().detach().to(self.t5_model.dtype)
        reference_size = torch.ones_like(reference_center).to(self.t5_model.dtype)
        reference_angle = torch.zeros(bs, npre_box).to(device=reference_center.device,dtype=reference_center.dtype).to(self.t5_model.dtype)
        reference_point = self.visual_encoder.dataset_config.box_parametrization_to_corners(reference_center,reference_size,reference_angle)
        reference_point[..., 1] *= -1 # X, -Z, Y
        reference_point[..., [0, 1, 2]] = reference_point[..., [0, 2, 1]]
        #load dict
        self.vision_locations_dict["reference_center"] = reference_center
        self.vision_locations_dict["reference_size"] = reference_size
        self.vision_locations_dict["reference_point"] = reference_point
        

        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=pc_embeds,
        #     encoder_attention_mask=image_atts,
        #     return_dict=True,
        # )
        # inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        inputs_t5 = torch.concat([pc_embeds, query_tokens], dim=1)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)
        
        self.vision_index_dict = {}
        self.vision_emb_start_index = 0
        self.vision_emb_end_index = inputs_t5.shape[1]
        self.num_fix_dectokens = self.num_query_token
        
        self.vision_index_dict["vision_enc_start_index"] = self.vision_emb_start_index
        self.vision_index_dict["vision_enc_end_index"] = self.vision_emb_end_index - self.num_fix_dectokens
        self.vision_index_dict["vision_dec_start_index"] = self.vision_emb_end_index - self.num_fix_dectokens
        self.vision_index_dict["vision_dec_end_index"] = self.vision_emb_end_index
        
        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=400,
                return_tensors="pt",
            ).to(pc_embeds.device)
            output_tokens = self.t5_tokenizer(
                samples["answer"],
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(pc_embeds.device)
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
            aux_outputs=[]
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
                vision_index_dict=self.vision_index_dict,
                vision_locations_dict=self.vision_locations_dict,
                aux_outputs=aux_outputs,
                output_attentions=True,
                output_hidden_states=True,  
            )
            
            loss = outputs.loss
            
            
            final_mask = torch.zeros_like(targets)
            final_mask[:, self.vision_emb_start_index:self.vision_emb_end_index][:, -self.num_fix_dectokens:] = 1
            final_mask = final_mask.bool()
            llm_hidden_feature = outputs.hidden_states[-1]
            bs, nseq, ndim = llm_hidden_feature.shape
            box_losses = torch.zeros_like(loss)
            center_loss = torch.zeros_like(loss)
            size_loss = torch.zeros_like(loss)
            # for ref task
            for batch_id in range(bs):
                box_seq_features = llm_hidden_feature[batch_id][final_mask[batch_id]]
                npre_box, ndim = box_seq_features.shape
                box_seq_features = box_seq_features.unsqueeze(0) #faking batch dim
                
                #query_xyz,query_emb
                query_xyz = self.query_xyz[batch_id].unsqueeze(0)
                
                #prepare_for_class
                box_class_logits = self.class_head(box_seq_features.transpose(1,2).contiguous()).transpose(1,2).contiguous()
                semcls_prob = box_class_logits
                objectness_prob = box_class_logits.sigmoid().max(dim=-1)[0]#focal loss
                
                #prepare_for_size 
                pre_box_size_unnormalized = torch.ones_like(query_xyz)
                box_seq_size_reg = self.size_head(box_seq_features.transpose(1,2).contiguous()).transpose(1,2).contiguous()
                box_seq_size_unnormlized = torch.exp(box_seq_size_reg) * pre_box_size_unnormalized
                                    
                #prepare_for_center
                pre_box_center_unnormalized = query_xyz
                box_seq_center_reg = self.center_head(box_seq_features.transpose(1,2).contiguous()).transpose(1,2).contiguous()
                box_seq_center_unnormlized = box_seq_center_reg * pre_box_size_unnormalized + pre_box_center_unnormalized
                
                #prepare_for_angle and conners
                box_angle = torch.zeros(box_seq_center_reg.shape[0], npre_box).to(device=box_seq_center_unnormlized.device,dtype=box_seq_center_unnormlized.dtype)
                
                box_corners = self.visual_encoder.dataset_config.box_parametrization_to_corners(box_seq_center_unnormlized,box_seq_size_unnormlized,box_angle)
                
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
                
                if torch.isnan(box_prediction["sem_cls_logits"].sigmoid()).any() or torch.isnan(box_prediction["center_reg"].sigmoid()).any():
                    print("ot layer torch.isnan(box_prediction[sem_cls_logits])", torch.where(torch.isnan(box_prediction["sem_cls_logits"])))
                    print("ot layer torch.where(torch.isnan(center_reg))", torch.where(torch.isnan(box_prediction["center_reg"])))
                    print("ot layer torch.where(torch.isnan(size_reg))", torch.where(torch.isnan(box_prediction["size_reg"])))
                    print("ot layer torch.where(torch.isnan(dec_feature))", torch.where(torch.isnan(box_seq_features)))
                    print("ot layer torch.where(torch.isnan(self.center_head.layers[-1].weight))", torch.where(torch.isnan(self.center_head.layers[-1].weight)))
                    print("ot layer torch.where(torch.isnan(self.class_head.layers[-1].weight))", torch.where(torch.isnan(self.class_head.layers[-1].weight)))
                    skip_matcher = True
                    print("skip_matcher:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("skip_matcher:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("skip_matcher:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("skip_matcher:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("skip_matcher:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("skip_matcher:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:
                    skip_matcher = False
                    
                box_pred_dict = {"outputs": box_prediction}
                if self.use_detr_mixin:
                    if self.encoder_pretrain.lower() == "vdetr":
                        #only support llama layer
                        # for layer in self.llama_model.base_model.model.model.layers:
                        #     if layer.__class__.__name__ == "VdetrLayer" and layer.detr_cross_attn_layer is not None:
                        #         aux_output = layer.box_prediction
                        #         #print(layer.reference_center)
                        #         aux_outputs.append(aux_output)
                        
                        box_pred_dict["aux_outputs"] = aux_outputs
                        
                        self.vision_matcher.is_bilable = False
                    
                        
                # box_info = {
                #         "box_present":box_present,
                #         "box_lists":box_lists,
                #                 }
                box_list = samples["box_lists"][batch_id]#list 
                box_present = samples["box_present"][batch_id]#tensor
                
                targets_boxes = {}
                if len(box_list[0])==7: #with class + box
                    class_offset = 1
                else:
                    class_offset = 0 #without class + box
                type2class_map = self.visual_encoder.dataset_config.type2class
                targets_boxes['gt_box_sem_cls_label'] = torch.tensor([type2class_map[box[0]] for box in box_list],device=llm_hidden_feature.device,dtype=llm_hidden_feature.dtype).unsqueeze(0).long().contiguous()
                targets_boxes["gt_box_centers"] = torch.tensor([box[class_offset:class_offset+3] for box in box_list],device=llm_hidden_feature.device,dtype=llm_hidden_feature.dtype).unsqueeze(0).contiguous()
                # # bottom center now
                # targets_boxes["gt_box_centers"][]
                
                targets_boxes["gt_box_sizes"] = torch.tensor([box[class_offset+3:class_offset+6] for box in box_list],device=llm_hidden_feature.device,dtype=llm_hidden_feature.dtype).unsqueeze(0).contiguous()
                targets_boxes["gt_box_angles"] = torch.zeros(targets_boxes["gt_box_centers"].shape[0], targets_boxes["gt_box_centers"].shape[1]).to(device=targets_boxes["gt_box_centers"].device,dtype=targets_boxes["gt_box_centers"].dtype)
                targets_boxes["gt_box_present"] = box_present.unsqueeze(0).to(device=targets_boxes["gt_box_centers"].device,dtype=targets_boxes["gt_box_centers"].dtype).contiguous()
                targets_boxes["gt_box_corners"] = self.visual_encoder.dataset_config.box_parametrization_to_corners(targets_boxes["gt_box_centers"],targets_boxes["gt_box_sizes"],targets_boxes["gt_box_angles"])
                if len(box_list)>0 and not skip_matcher:
                #if len(box_list)>0:
                    box_losses, box_loss_dict = self.vision_matcher(box_pred_dict, targets_boxes) 
                    center_loss = box_loss_dict['loss_center']
                    size_loss = box_loss_dict['loss_size']
                    giou_loss = box_loss_dict['loss_giou']
                    class_loss = box_loss_dict['loss_sem_cls']
                else:
                    print("there is no box")
                    # box_losses = torch.zeros_like(loss)
                    center_loss = torch.zeros_like(loss)
                    size_loss = torch.zeros_like(loss)
                    giou_loss = torch.zeros_like(loss)
                    class_loss = torch.zeros_like(loss)               
                
                # print("self.class_head.layers[0].weight.mean()",self.class_head.layers[0].weight.mean())
            
            loss += box_losses
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
        )
        model.load_checkpoint_from_config(cfg)

        return model
