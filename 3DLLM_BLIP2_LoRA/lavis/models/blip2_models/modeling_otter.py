from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from einops import rearrange, repeat
from accelerate.hooks import add_hook_to_module, AlignDevicesHook
import math

# from .configuration_otter import OtterConfig

# from flamingo.falcon.modelling_RW import RWForCausalLM
# from flamingo.mpt.modeling_mpt import MPTForCausalLM
# from flamingo.mpt_redpajama.mosaic_gpt import MosaicGPT

from transformers.models.auto import AutoModel, AutoModelForCausalLM, AutoTokenizer
from .VDETR.truerpe_transformer import TransformerDecoderLayer, GlobalDecoderLayer, BoxProcessor, inverse_sigmoid

from peft import get_peft_model, LoraConfig, TaskType

import sys
import random

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

import torch.distributed as dist


from VDETR.helpers import GenericMLP, PositionEmbeddingLearned
from functools import partial

# Add this line at the beginning of your script or in your main function
# dist.init_process_group(backend='nccl')

XFORMERS_AVAIL = False
XFORMERS_MSG_PRINTED = False  # Add this global variable
try:
    if not XFORMERS_MSG_PRINTED:  # Check if the message has been printed before
        import xformers.ops as xops
        from xformers_model import CLIPVisionModel, LlamaForCausalLM
        from transformers import LlamaTokenizer

        _xformers_version = importlib_metadata.version("xformers")
        if dist.is_initialized() and dist.get_rank() == 0:  # Check if the current process rank is 0
            print(f"Successfully imported xformers version {_xformers_version}")
except ImportError as e:
    if not XFORMERS_MSG_PRINTED:  # Check if the message has been printed before
        from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

        if dist.is_initialized() and dist.get_rank() == 0:  # Check if the current process rank is 0
            print(f"Failed to import xformers: {e}")
            XFORMERS_AVAIL = False
            print("No xformers found. You are recommended to install xformers via `pip install xformers` or `conda install -c xformers xformers`")
            XFORMERS_MSG_PRINTED = True  # Set the variable to True after printing the message

# from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "RWForCausalLM": "transformer.h",
    "MPTForCausalLM": "transformer.blocks",
    "MosaicGPT": "transformer.blocks",
}

MODEL_CLASSES = {
    "LlamaForCausalLM": "llama",
    "OPTForCausalLM": "opt",
    "GPTJForCausalLM": "gptj",
    "GPTNeoXForCausalLM": "gpt_neox",
    "MPTForCausalLM": "mpt",
    "MosaicGPT": "mpt",
}


def _infer_decoder_layers_attr_name(model: nn.Module):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (mixin, base_cls), {})  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def exists(val):
    return val is not None


class OtterPerceiverBlock(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8, mult: int = 4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        ff_dim = dim * mult
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, ff_dim, bias=False),
                nn.GELU(),
                nn.Linear(ff_dim, dim, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        residual_latents = latents
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, "b t n (h d) -> b h t n d", h=h)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=h)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        out = self.to_out(out) + residual_latents
        residual_out = out
        for layer in self.feed_forward:
            out = layer(out)
        return out + residual_out


class OtterPerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        # max_num_frames: int = 128,
        max_num_media: Optional[int] = None,
        max_num_frames: Optional[int] = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim)) if exists(max_num_frames) else None

        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(OtterPerceiverBlock(dim=dim, dim_head=dim_head, heads=heads, mult=ff_mult))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for block in self.layers:
            latents = block(x, latents)
        return self.norm(latents)


class OtterMaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        media_locations: Optional[torch.BoolTensor] = None,
        attend_previous: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            attend_previous: bool
                If false, ignores immediately preceding image and starts attending when following image
        """
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")#nximage==n*numimage

        k, v = self.to_kv(media).chunk(2, dim=-1)
        if not XFORMERS_AVAIL:
            q = rearrange(q, "b n (h d) -> b h n d", h=h)
            k = rearrange(k, "b n (h d) -> b h n d", h=h)
            v = rearrange(v, "b n (h d) -> b h n d", h=h)
            q = q * self.scale

            sim = torch.einsum("... i d, ... j d -> ... i j", q, k)#bs, nhead, ntxt, nximage
            if exists(media_locations):
                # at each boolean of True, increment the time counter (relative to media time)
                text_time = media_locations.cumsum(dim=-1) 
                #input (text0,text1,image0),(text2,text3,image1),byshen 
                #text_time (0,0,1),(1,1,2),byshen 
                media_time = torch.arange(T_img, device=x.device) + 1
                #media  image0,image1,byshen
                #media_time 1,2,byshen

                if not attend_previous:
                    text_time[~media_locations] += 1
                    # make sure max is still the number of images in the sequence
                    text_time[
                        text_time
                        > repeat(
                            torch.count_nonzero(media_locations, dim=1),
                            "b -> b i",
                            i=text_time.shape[1],
                        )
                    ] = 0

                # text time must equal media time if only attending to most immediate image
                # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
                mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

                text_to_media_mask = mask_op(
                    rearrange(text_time, "b i -> b 1 i 1"),
                    repeat(media_time, "j -> 1 1 1 (j n)", n=n),
                )#bs, nhead, ntxt, nximage
                sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

            sim = sim - sim.amax(dim=-1, keepdim=True).detach()
            attn = sim.softmax(dim=-1)

            if exists(media_locations) and self.only_attend_immediate_media:
                # any text without a preceding media needs to have attention zeroed out
                text_without_media_mask = text_time == 0
                text_without_media_mask = rearrange(text_without_media_mask, "b i -> b 1 i 1")
                attn = attn.masked_fill(text_without_media_mask, 0.0)

            out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
        else:
            q = rearrange(q, "b n (h d) -> b n h d", h=h)
            k = rearrange(k, "b n (h d) -> b n h d", h=h)
            v = rearrange(v, "b n (h d) -> b n h d", h=h)
            attn_mask = None
            out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, scale=self.scale)
        return self.to_out(out)


class OtterGatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_visual: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.attn = OtterMaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * ff_mult, bias=False),
                nn.GELU(),
                nn.Linear(dim * ff_mult, dim, bias=False),
            ]
        )
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x: torch.Tensor,
        media: torch.Tensor,
        media_locations: Optional[torch.BoolTensor] = None,
        attend_previous: bool = True,
    ) -> torch.Tensor:
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                attend_previous=attend_previous,
            )
            * self.attn_gate.tanh()
            + x
        )
        residual_x = x
        for ff in self.feed_forward:
            x = ff(x)
        x = x * self.ff_gate.tanh() + residual_x

        return x


class OtterLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer: nn.Module, decoder_layer: nn.Module):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Otter (https://github.com/dhansmair/otter-mini/)
    def condition_vis_x(self, vis_x) -> None:
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations) -> None:
        self.media_locations = media_locations

    def condition_attend_previous(self, attend_previous) -> None:
        self.attend_previous = attend_previous

    def forward(
        self,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **decoder_layer_kwargs,
    ):
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)

        if self.vis_x is None:
            raise ValueError("vis_x must be conditioned before forward pass")

        if self.media_locations is None:
            raise ValueError("media_locations must be conditioned before forward pass")

        lang_x = self.gated_cross_attn_layer(
            lang_x,
            self.vis_x,
            media_locations=self.media_locations,
            attend_previous=self.attend_previous,
        )
        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x


class OtterLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_otter(
        self,
        media_token_id: int,
        vis_hidden_size: int,
        cross_attn_every_n_layers: int,
        use_media_placement_augmentation: bool,
        only_attend_immediate_media:bool=True,
    ):
        """
        Initialize Otter by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """

        gated_cross_attn_layers = nn.ModuleList(
            [
                OtterGatedCrossAttentionBlock(
                    dim=self.config.hidden_size,
                    dim_visual=vis_hidden_size,
                    only_attend_immediate_media=only_attend_immediate_media,
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    OtterLayer(gated_cross_attn_layer, decoder_layer)
                    for gated_cross_attn_layer, decoder_layer in zip(gated_cross_attn_layers, self._get_decoder_layers())
                ]
            )
        )
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        self.initialized_otter = True

    def forward(self, *input, **kwargs):
        """Condition the Otter layers on the media locations before forward()"""
        if not self.initialized_otter:
            raise ValueError("Otter layers are not initialized. Please call `init_otter` first.")

        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else input[0]
        media_locations = input_ids == self.media_token_id
        # IMPORTANT: Force `attend_previous` to True when we place training data as <image>caption<|endofchunk|>
        # attend_previous = (
        #     (random.random() < 0.5) if self.use_media_placement_augmentation else False
        # )
        attend_previous = (random.random() < 0.5) if self.use_media_placement_augmentation else True
        # attend_previous = self.only_attend_previous

        if self.__class__.__name__ == "LlamaForCausalLM":
            for layer in self.get_decoder().layers:
                layer.condition_media_locations(media_locations)
                layer.condition_attend_previous(attend_previous)
        elif self.__class__.__name__ in ["MPTForCausalLM", "MosaicGPT"]:
            for layer in self.get_decoder().blocks:
                layer.condition_media_locations(media_locations)
                layer.condition_attend_previous(attend_previous)
        else:
            print("inavaliable text encoder")
        return super().forward(*input, **kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self) -> None:
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_attend_previous(None)


class VdetrLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_otter(
        self,
        vision_args: dict,
        cross_attn_every_n_layers: int,
        llm_hidden_dim: int=5120,
        dataset_config=None,
    ):
        """
        Initialize Otter by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        detr_cross_attn_layers = nn.ModuleList(
            [
                GlobalDecoderLayer(
                    d_model=vision_args.dec_dim,
                    nhead=vision_args.dec_nhead,
                    dim_feedforward=vision_args.dec_ffn_dim,
                    dropout=vision_args.dec_dropout,
                    pos_for_key=vision_args.pos_for_key,
                    args=vision_args
                )
                if ((layer_idx + 1) % cross_attn_every_n_layers == 0)
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    VdetrLayer(detr_cross_attn_layer, decoder_layer,dec_output_dim=vision_args.dec_dim,llm_hidden_dim=llm_hidden_dim,dataset_config=dataset_config,vision_args=vision_args)
                    for detr_cross_attn_layer, decoder_layer in zip(detr_cross_attn_layers, self._get_decoder_layers())
                ]
            )
        )
        self.initialized_otter = True
    def forward(self, *input, **kwargs):
        """Condition the Otter layers on the media locations before forward()"""
        if not self.initialized_otter:
            raise ValueError("Vdetr layers are not initialized. Please call `init_otter` first.")

        vision_locations_dict = kwargs["vision_locations_dict"]
        vision_index_dict = kwargs["vision_index_dict"]
 

        if self.__class__.__name__ == "LlamaForCausalLM":
            for layer in self.get_decoder().layers:
                #layer.condition_features_locations(vision_locations_dict)
                layer.condition_vision_index_dict(vision_index_dict)
        elif self.__class__.__name__ in ["MPTForCausalLM", "MosaicGPT"]:
            for layer in self.get_decoder().blocks:
                #layer.condition_features_locations(vision_locations_dict)
                layer.condition_vision_index_dict(vision_index_dict)
        else:
            print("inavaliable text encoder")
        return super().forward(*input, **kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self) -> None:
        for layer in self._get_decoder_layers():
            layer.condition_features_locations(None)
            layer.condition_vision_index_dict(None)



class VdetrLayer(nn.Module):
    def __init__(self, detr_cross_attn_layer: nn.Module, decoder_layer: nn.Module, dec_output_dim=256, llm_hidden_dim=5120, dataset_config=None,vision_args=None,):
        super().__init__()
        self.detr_cross_attn_layer = detr_cross_attn_layer
        self.decoder_layer = decoder_layer
        
        #vdetr setting
        self.reference_point = None
        self.reference_center = None
        self.reference_size = None
        self.reference_angle = None
        self.enc_xyz = None
        self.point_cloud_dims = None
        self.vision_args = vision_args
        self.dataset_config = dataset_config
        self.center_unnorm = vision_args.center_unnorm
        self.size_unnorm = vision_args.size_unnorm
        
        
        self.dec_output_dim = dec_output_dim
        self.llm_hidden_dim = llm_hidden_dim
        if self.detr_cross_attn_layer is not None:
            self.query_pos_projection = PositionEmbeddingLearned(6, self.dec_output_dim)
            self.box_processor = BoxProcessor(dataset_config, cls_loss="focalloss")
            
            # if self.llm_hidden_dim != self.dec_output_dim:
            self.llm_to_dec_projection = nn.Linear(llm_hidden_dim,dec_output_dim)
            self.dec_to_llm_projection = nn.Linear(dec_output_dim,llm_hidden_dim)
            self.llm_to_enc_projection = nn.Linear(llm_hidden_dim,dec_output_dim)
            # self.enc_to_llm_projection = nn.Linear(llm_hidden_dim,dec_output_dim)
            
            mlp_func = partial(
                GenericMLP,
                # norm_fn_name='bn1d',
                activation='relu',
                use_conv=True,
                hidden_dims=[dec_output_dim, dec_output_dim],
                input_dim=dec_output_dim,
            )
            self.center_head = mlp_func(output_dim=3)
            self.size_head = mlp_func(output_dim=3)
            self.sem_cls_head = mlp_func(output_dim=18)#hacking for Scannet
            self.detr_norm = nn.LayerNorm(dec_output_dim,eps=1e-4)
            
            #initial
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.sem_cls_head.layers[-1].bias.data = torch.ones(18) * bias_value
            
            nn.init.constant_(self.center_head.layers[-1].weight.data, 0.0)
            nn.init.constant_(self.center_head.layers[-1].bias.data, 0.0)
            nn.init.constant_(self.size_head.layers[-1].weight.data, 0.0)
            nn.init.constant_(self.size_head.layers[-1].bias.data, 0.0)
        
    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vision_enc_start_index is not None

    def get_proposal_box_predictions_refine(self, point_cloud_dims, box_features, pre_center_unnormalized=None,pre_size_unnormalized=None,straight_size_output=False):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.unsqueeze(0).permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)
        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.sem_cls_head(box_features).transpose(1, 2)

        #prepare pre_size and pre_center, byshen
        scene_size = point_cloud_dims[1]-point_cloud_dims[0]

        if pre_size_unnormalized is None: #onstage, not have the proposal size
            class_idx = cls_logits.sigmoid().max(dim=-1)[1]
            size_per_class = torch.tensor(self.dataset_config.mean_size_arr, device=cls_logits.device).float()
            pre_size_unnormalized = size_per_class[class_idx]#class_mean_size,will use in unnorm_predition             

        #center
        assert  pre_center_unnormalized!=None
        if self.center_unnorm:
            center_reg =  self.center_head(box_features).transpose(1, 2).contiguous().view(num_layers * batch,num_queries,3).contiguous()
            center_unnormalized = center_reg * pre_size_unnormalized + pre_center_unnormalized
            center_normalized = (center_unnormalized - point_cloud_dims[0].unsqueeze(1).repeat(1,num_queries,1))/scene_size.unsqueeze(1).repeat(1,num_queries,1)
        else:
            center_normalized_before_sigmoid = inverse_sigmoid(center_normalized)
            center_normalized_offset_before_sigmoid = self.center_head(box_features).transpose(1, 2)
            center_normalized = (center_normalized_before_sigmoid+center_normalized_offset_before_sigmoid).sigmoid()
            scene_size = point_cloud_dims[1]-point_cloud_dims[0]
            center_unnormalized = center_normalized*scene_size.unsqueeze(1).repeat(1,num_queries,1)+point_cloud_dims[0].unsqueeze(1).repeat(1,num_queries,1)
            center_reg = None
            
        #size
        assert  pre_size_unnormalized!=None
        if self.size_unnorm and not straight_size_output: 
            size_reg = self.size_head(box_features).transpose(1, 2).contiguous().view(num_layers * batch,num_queries,3).contiguous()
            #size_reg = size_reg.clamp(max=10)#hacking
            size_unnormalized = torch.exp(size_reg)*pre_size_unnormalized
            size_normalized = size_unnormalized/scene_size.unsqueeze(1).repeat(1,num_queries,1)
        elif straight_size_output:
            size_reg = None
            #size_normalized_before_sigmoid = inverse_sigmoid(pre_size_normalized)
            size_normalized_before_sigmoid = self.size_head(box_features).transpose(1, 2) 
            size_normalized = (size_normalized_before_sigmoid).sigmoid()
            size_unnormalized = self.box_processor.compute_predicted_size(
            size_normalized, point_cloud_dims
                )          
        else:
            size_reg = None
            pre_size_normalized = pre_size_unnormalized
            size_normalized_before_sigmoid = inverse_sigmoid(pre_size_normalized)
            size_normalized_offset_before_sigmoid = self.size_head(box_features).transpose(1, 2) 
            size_normalized = (size_normalized_offset_before_sigmoid+size_normalized_before_sigmoid).sigmoid()
            size_unnormalized = self.box_processor.compute_predicted_size(
            size_normalized, point_cloud_dims
                )

        #angle
        # angle_logits = self.angle_cls_head"](box_features).transpose(1, 2)
        # angle_residual_normalized = self.angle_residual_head"](
        #     box_features
        # ).transpose(1, 2)
        # angle_residual = angle_residual_normalized * (
        #     np.pi / angle_residual_normalized.shape[-1]
        # )
        # angle_continuous, angle_prob = self.box_processor.compute_predicted_angle(
        #     angle_logits, angle_residual
        # )


        #conners
        angle_continuous = torch.zeros(center_unnormalized.shape[0], center_unnormalized.shape[1]).to(device=center_unnormalized.device,dtype=center_unnormalized.dtype)
        box_corners = self.box_processor.box_parametrization_to_corners(
            center_unnormalized, size_unnormalized, angle_continuous
        )               
        with torch.no_grad():
            (
                semcls_prob,
                objectness_prob,
            ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits)

        box_prediction = {
            "sem_cls_logits": cls_logits.contiguous(),
            "center_normalized": center_normalized.contiguous(),
            "center_unnormalized": center_unnormalized.contiguous(),
            "size_normalized": size_normalized.contiguous(),
            "size_unnormalized": size_unnormalized.contiguous(),
            "angle_continuous": angle_continuous.contiguous(),
            "objectness_prob": objectness_prob.contiguous(),
            "sem_cls_prob": semcls_prob.contiguous(),
            "box_corners": box_corners.contiguous(),
        }
        if self.center_unnorm:
            box_prediction["pre_box_center_unnormalized"] = pre_center_unnormalized.contiguous()
            box_prediction["center_reg"] = center_reg.contiguous()
        if self.size_unnorm:
            box_prediction["pre_box_size_unnormalized"] = pre_size_unnormalized.contiguous()
            box_prediction["size_reg"] = size_reg.contiguous()    
        return box_prediction

    def condition_features_locations(self, vision_locations_dict) -> None:
        self.pre_center_unnormalized =  vision_locations_dict["pre_center_unnormalized"].clone().detach() if vision_locations_dict is not None else None#bs, ndec, 8, 3
        self.pre_size_unnormalized =  vision_locations_dict["pre_size_unnormalized"].clone().detach() if vision_locations_dict is not None else None#bs, ndec, 8, 3
        
        self.reference_point = vision_locations_dict["reference_point"].clone().detach() if vision_locations_dict is not None else None#bs, ndec, 8, 3
        self.reference_center = vision_locations_dict["reference_center"].clone().detach() if vision_locations_dict is not None else None
        self.reference_size = vision_locations_dict["reference_size"].clone().detach() if vision_locations_dict is not None else None
        #self.reference_angle = vision_locations_dict["reference_angle"].clone().detach() if vision_locations_dict is not None else None
        self.reference_angle = None
        self.enc_xyz = vision_locations_dict["enc_xyz"] if vision_locations_dict is not None else None
        self.point_cloud_dims = vision_locations_dict["point_cloud_dims"] if vision_locations_dict is not None else None
        
    def condition_vision_index_dict(self, vision_index_dict) -> None:
        self.vision_enc_start_index = vision_index_dict["vision_enc_start_index"] if vision_index_dict is not None else None
        self.vision_enc_end_index = vision_index_dict["vision_enc_end_index"] if vision_index_dict is not None else None
        self.vision_dec_start_index = vision_index_dict["vision_dec_start_index"] if vision_index_dict is not None else None
        self.vision_dec_end_index = vision_index_dict["vision_dec_end_index"] if vision_index_dict is not None else None
        

    def condition_attend_previous(self, attend_previous) -> None:
        self.attend_previous = attend_previous

    def forward(
        self,
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **decoder_layer_kwargs,
    ):
        lang_x_origin = lang_x.clone()
        if self.detr_cross_attn_layer is None or (not self.training and "past_key_value" in decoder_layer_kwargs.keys() and decoder_layer_kwargs["past_key_value"] is not None):
            if torch.isnan(lang_x).any():
                print("origin layer lang_x", torch.where(torch.isnan(lang_x)))
                print("self.decoder_layer.input_layernorm.weight",self.decoder_layer.input_layernorm.weight)
            return self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        
        if "vision_locations_dict" in decoder_layer_kwargs.keys():
            vision_locations_dict = decoder_layer_kwargs["vision_locations_dict"]
            self.condition_features_locations(vision_locations_dict)
        
        enc_feature = self.llm_to_dec_projection(lang_x[ : , self.vision_enc_start_index:self.vision_enc_end_index])
        dec_feature = self.llm_to_enc_projection(lang_x[ : , self.vision_dec_start_index:self.vision_dec_end_index])
        dec_feature_before_detr_cross = dec_feature.detach().clone()
        if torch.isnan(dec_feature).any():
            print("mixin layer torch.where(torch.isnan(dec_feature_before_detr_cross))", torch.where(torch.isnan(dec_feature)))
        
        query_reference = torch.cat([self.reference_center, self.reference_size], dim=-1)
        query_pos = self.query_pos_projection(query_reference).permute(2,0,1)
        
        dec_feature,detr_attn = self.detr_cross_attn_layer(
            dec_feature.permute(1,0,2).contiguous(),enc_feature.permute(1,0,2).contiguous(),
            self.reference_point, self.reference_angle, self.enc_xyz, self.point_cloud_dims,
            query_pos=query_pos,return_attn_weights=True,
        )
        
        
        box_prediction = \
            self.get_proposal_box_predictions_refine(
                point_cloud_dims=self.point_cloud_dims,
                box_features=self.detr_norm(dec_feature),
                pre_center_unnormalized=self.pre_center_unnormalized,
                pre_size_unnormalized=self.pre_size_unnormalized,

                )
        box_prediction["detr_attn"] = detr_attn
        #print("mixin sem_cls_logits.min",box_prediction["sem_cls_logits"].sigmoid().min())
        # if torch.isnan(box_prediction["sem_cls_logits"].sigmoid()).any() or torch.isnan(dec_feature).any():
        #     print("mixin layer torch.isnan(box_prediction[sem_cls_logits])", torch.where(torch.isnan(box_prediction["sem_cls_logits"])))
        #     print("mixin layer torch.where(torch.isnan(center_reg))", torch.where(torch.isnan(box_prediction["center_reg"])))
        #     print("mixin layer torch.where(torch.isnan(size_reg))", torch.where(torch.isnan(box_prediction["size_reg"])))
        #     print("mixin layer torch.where(torch.isnan(dec_feature))", torch.where(torch.isnan(dec_feature)))
        #     print("mixin layer torch.where(torch.isnan(dec_feature_before_detr_cross))", torch.where(torch.isnan(dec_feature_before_detr_cross)))
        #     print("mixin layer torch.where(torch.isnan(detr_cross_attn_layer.linear1.weight))", torch.where(torch.isnan(self.detr_cross_attn_layer.linear1.weight)))#
        #     print("mixin layer torch.where(torch.isnan(detr_attn))", torch.where(torch.isnan(detr_attn)))
        #     print("mixin layer torch.where(torch.isnan(self.center_head.layers[-1].weight))", torch.where(torch.isnan(self.center_head.layers[-1].weight)))
        #     print("mixin layer torch.where(torch.isnan(self.norm1.eps))", sss)
        if "aux_outputs" in decoder_layer_kwargs.keys():
            decoder_layer_kwargs["aux_outputs"].append(box_prediction)
        
        reference_point = box_prediction['box_corners'].clone().detach()
        reference_point[..., 1] *= -1 # X, -Z, Y
        reference_point[..., [0, 1, 2]] = reference_point[..., [0, 2, 1]]
        reference_center = box_prediction['center_unnormalized'].clone().detach()
        reference_size = box_prediction['size_unnormalized'].clone().detach()
        # update vision_locations_dict
        vision_locations_dict["reference_point"] = reference_point
        vision_locations_dict["reference_center"] = reference_center
        vision_locations_dict["reference_size"] = reference_size
        
        
        
        #update 
        
        num_token = lang_x.shape[1]
        dec_token_emb = self.dec_to_llm_projection(dec_feature).permute(1,0,2).contiguous()
        
        padding_dec_token_emb = F.pad(dec_token_emb,(0,0,self.vision_dec_start_index,num_token-self.vision_dec_end_index),"constant", 0)
        lang_x = lang_x + padding_dec_token_emb
        
        #lang_x[ : , self.vision_dec_start_index:self.vision_dec_end_index] = dec_feature
        # output, attn = layer(quant_output, memory, 
        #                 reference_point, reference_angle, enc_xyz, point_cloud_dims,
        #                 tgt_mask=tgt_mask,
        #                 memory_mask=memory_mask,
        #                 tgt_key_padding_mask=tgt_key_padding_mask,
        #                 memory_key_padding_mask=memory_key_padding_mask,
        #                 pos=pos, query_pos=query_pos,
        #                 return_attn_weights=return_attn_weights)

        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        # print("min lang_x.sum is ",lang_x[0].sum())
        # if torch.isnan(lang_x[0]).any():
        #     print("minxin lang_x is ",lang_x)
        #     print("minxin lang_x_origin is ",lang_x_origin)
        #     print("mixin layer lang_x", torch.where(torch.isnan(lang_x[0])))
        #     print("mixin layer lang_x_origin", torch.where(torch.isnan(lang_x_origin)))
        #     print("mixin layer padding_dec_token_emb", torch.where(torch.isnan(padding_dec_token_emb)))
        #     print("self.decoder_layer.input_layernorm.weight",self.decoder_layer.input_layernorm.weight)
        #     lang_x = self.decoder_layer(lang_x_origin, attention_mask=attention_mask, **decoder_layer_kwargs)
            
        return lang_x
