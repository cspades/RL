# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Small matched Nemotron6-MoE MIMO and integrated refit providers."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
    MegatronMIMOProvider,
)
from megatron.core.activations import fast_gelu, squared_relu
from megatron.core.extensions.transformer_engine import TERowParallelLinear
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.multimodal.llava_model import LLaVAModel, pixel_shuffle
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec


HYBRID_PATTERN = "MEMEM*EMEMEM*EMEMEM*"
RADIO_MODALITY = "images"
RADIO_ENCODER = "radio_encoder"


def _pixel_shuffle_dynamic_resolution(
    embeddings: torch.Tensor,
    image_sizes: torch.Tensor,
    patch_dim: int,
    scale_factor: float = 0.5,
) -> torch.Tensor:
    """Apply RADIO pixel shuffle independently to variable-resolution images."""
    sequence_lengths = torch.prod(image_sizes // patch_dim, dim=-1)
    image_embeddings = torch.split(embeddings, sequence_lengths.tolist(), dim=-2)
    shuffled = []
    for image_embedding, image_size in zip(image_embeddings, image_sizes):
        height = int(image_size[0]) // patch_dim
        width = int(image_size[1]) // patch_dim
        image_embedding = image_embedding.reshape(
            image_embedding.shape[0], height, width, -1
        )
        batch, height, width, channels = image_embedding.shape
        image_embedding = image_embedding.view(
            batch,
            height,
            int(width * scale_factor),
            int(channels / scale_factor),
        )
        image_embedding = image_embedding.permute(0, 2, 1, 3).contiguous()
        image_embedding = image_embedding.view(
            batch,
            int(width * scale_factor),
            int(height * scale_factor),
            int(channels / (scale_factor * scale_factor)),
        )
        image_embedding = image_embedding.permute(0, 2, 1, 3).contiguous()
        shuffled.append(
            image_embedding.reshape(image_embedding.shape[0], -1, image_embedding.shape[-1])
        )
    return torch.cat(shuffled, dim=-2)


class RADIOEncoderWrapper(RADIOViTModel):
    """RADIO adapter used by the colocated MIMO modality submodule.

    This lives in an installed NeMo-RL module because Megatron-LM's
    ``examples.mimo`` tree is intentionally not an importable package.
    """

    def __init__(
        self,
        transformer_config,
        transformer_layer_spec: ModuleSpec,
        pg_collection,
        img_h: int,
        img_w: int,
        patch_dim: int,
        class_token_len: int,
        drop_class_token: bool = True,
        apply_pixel_shuffle: bool = True,
        force_eval_mode: bool = False,
        dynamic_resolution: bool = False,
    ) -> None:
        super().__init__(
            transformer_config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            patch_dim=patch_dim,
            img_h=img_h,
            img_w=img_w,
            class_token_len=class_token_len,
            add_class_token=True,
            max_img_h=2048,
            max_img_w=2048,
            has_cpe=True,
            embedder_bias=False,
            dynamic_resolution=dynamic_resolution,
            force_eval_mode=force_eval_mode,
            pg_collection=pg_collection,
        )
        self.drop_class_token = drop_class_token
        self.apply_pixel_shuffle = apply_pixel_shuffle

    def _patchify_dynamic_images(
        self, images: torch.Tensor, imgs_sizes: torch.Tensor
    ) -> torch.Tensor:
        """Convert padded processor pixels to RADIO's packed patch format."""
        patch_dim = self.patch_dim
        patch_features = 3 * patch_dim * patch_dim
        if images.ndim == 3 and images.shape[0] == 1:
            if images.shape[-1] != patch_features:
                raise ValueError(
                    "Patchified RADIO input has the wrong feature width: "
                    f"expected {patch_features}, got {images.shape[-1]}."
                )
            return images
        if images.ndim != 4:
            raise ValueError(
                "Dynamic-resolution RADIO input must be padded pixels [N,C,H,W] "
                "or packed patches [1,total_patches,C*P*P]; "
                f"got shape {tuple(images.shape)}."
            )
        if images.shape[0] != imgs_sizes.shape[0]:
            raise ValueError(
                f"Received {images.shape[0]} images but "
                f"{imgs_sizes.shape[0]} image sizes."
            )

        patches = []
        for image, size in zip(images, imgs_sizes):
            height, width = (int(value) for value in size.tolist())
            if height % patch_dim or width % patch_dim:
                raise ValueError(
                    f"Image size {(height, width)} is not divisible by "
                    f"patch_dim={patch_dim}."
                )
            image = image[:, :height, :width]
            channels = image.shape[0]
            rows = height // patch_dim
            columns = width // patch_dim
            patches.append(
                image.reshape(
                    channels,
                    rows,
                    patch_dim,
                    columns,
                    patch_dim,
                )
                .permute(1, 3, 0, 2, 4)
                .reshape(rows * columns, channels * patch_dim * patch_dim)
            )
        return torch.cat(patches, dim=0).unsqueeze(0).contiguous()

    def _build_packed_seq_params(
        self, imgs_sizes: torch.Tensor
    ) -> PackedSeqParams:
        """Build RADIO's per-image THD boundaries from image dimensions."""
        patch_dim = self.patch_dim
        sequence_lengths = [
            (int(height) // patch_dim) * (int(width) // patch_dim)
            for height, width in imgs_sizes.tolist()
        ]
        cumulative_lengths = [0]
        for sequence_length in sequence_lengths:
            cumulative_lengths.append(cumulative_lengths[-1] + sequence_length)
        cu_seqlens = torch.tensor(
            cumulative_lengths,
            dtype=torch.int32,
            device=imgs_sizes.device,
        )
        max_seqlen = max(sequence_lengths, default=0)
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

    def forward(
        self,
        x: torch.Tensor,
        imgs_sizes: Optional[torch.Tensor] = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        context = torch.no_grad() if self.force_eval_mode else nullcontext()
        with context:
            if self.dynamic_resolution:
                if imgs_sizes is None:
                    raise ValueError(
                        "imgs_sizes is required for dynamic-resolution RADIO input."
                    )
                x = self._patchify_dynamic_images(x, imgs_sizes)
                if packed_seq_params is None:
                    packed_seq_params = self._build_packed_seq_params(imgs_sizes)
            embeddings = super().forward(
                x.to(dtype=self.embedder.weight.dtype),
                imgs_sizes=imgs_sizes,
                packed_seq_params=packed_seq_params,
            )

        if self.drop_class_token:
            if self.dynamic_resolution and imgs_sizes is not None and self.class_token_len:
                keep = torch.ones(
                    embeddings.shape[-2],
                    dtype=torch.bool,
                    device=embeddings.device,
                )
                sequence_lengths = torch.prod(
                    imgs_sizes // self.patch_dim, dim=-1
                )
                offset = 0
                for sequence_length in sequence_lengths:
                    keep[offset : offset + self.class_token_len] = False
                    offset += int(sequence_length) + self.class_token_len
                embeddings = embeddings[:, keep, :]
            else:
                embeddings = embeddings[:, self.class_token_len :, :]

        if self.apply_pixel_shuffle:
            if self.dynamic_resolution and imgs_sizes is not None:
                embeddings = _pixel_shuffle_dynamic_resolution(
                    embeddings, imgs_sizes, self.patch_dim
                )
            else:
                embeddings = pixel_shuffle(embeddings, scale_factor=0.5)
        return embeddings

def _projection_submodules() -> MLPSubmodules:
    return MLPSubmodules(
        linear_fc1=ColumnParallelLinear,
        linear_fc2=TERowParallelLinear,
    )


@dataclass
class NemotronMockIntegratedProvider(HybridModelProvider):
    """Ordinary Hybrid+RADIO LLaVA provider matching the MIMO policy."""

    hybrid_layer_pattern: str = HYBRID_PATTERN
    num_layers: int | None = 20
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_query_groups: int = 4
    kv_channels: int = 64
    ffn_hidden_size: int = 512
    vocab_size: int = 131072
    seq_length: int = 1024
    make_vocab_size_divisible_by: int = 128

    mamba_num_heads: int = 16
    mamba_head_dim: int = 64
    mamba_num_groups: int = 4
    mamba_state_dim: int = 64
    linear_conv_kernel_dim: int = 4

    num_moe_experts: int = 8
    moe_router_topk: int = 2
    moe_ffn_hidden_size: int = 512
    moe_shared_expert_intermediate_size: int = 1024
    moe_router_score_function: str = "sigmoid"
    moe_router_topk_scaling_factor: float = 2.5
    moe_router_enable_expert_bias: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_aux_loss_coeff: float = 1.0e-4
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_shared_expert_overlap: bool = False

    activation_func: Any = squared_relu
    gated_linear_unit: bool = False
    add_bias_linear: bool = False
    normalization: str = "RMSNorm"
    position_embedding_type: str = "none"
    is_hybrid_model: bool = True
    language_model_type: str = "nemotron6-moe"
    vision_model_type: str = "radio"
    calculate_per_token_loss: bool = False
    mtp_num_layers: int | None = 0
    scatter_embedding_sequence_parallel: bool = False
    share_embeddings_and_output_weights: bool = False

    # Matches the tokenizer/processor metadata supplied by the public Nemotron
    # Omni model_name; weights remain random and are never loaded from it.
    image_token_index: int = 18
    img_h: int = 512
    img_w: int = 512
    patch_dim: int = 16
    class_token_len: int = 8
    dynamic_resolution: bool = True
    pixel_shuffle: bool = True
    drop_vision_class_token: bool = True
    vision_projection_type: str = "affine"
    gradient_accumulation_fusion: bool = False

    def _language_config(self):
        config = self._copy_config_without_runtime_process_groups(deep=True)
        config.language_model_type = self.language_model_type
        config.calculate_per_token_loss = False
        return config

    def _vision_config(self):
        config = self._language_config()
        config.num_layers = 32
        config.hidden_size = 1280
        config.num_attention_heads = 16
        config.num_query_groups = 16
        config.kv_channels = 80
        config.ffn_hidden_size = 5120
        config.gated_linear_unit = False
        config.activation_func = fast_gelu
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.normalization = "LayerNorm"
        config.layernorm_epsilon = 1.0e-6
        config.layernorm_zero_centered_gamma = False
        config.qk_layernorm = False
        config.vision_model_type = self.vision_model_type
        config.num_moe_experts = None
        config.moe_ffn_hidden_size = None
        config.moe_shared_expert_intermediate_size = None
        config.moe_grouped_gemm = False
        config.is_hybrid_model = False
        config.hybrid_layer_pattern = None
        config.sequence_parallel = False
        config.expert_model_parallel_size = 1
        config.expert_tensor_parallel_size = 1
        config.mtp_num_layers = 0
        return config

    def _projection_config(self):
        config = self._language_config()
        config.num_layers = 1
        # TransformerConfig validates attention-head divisibility even though
        # the projector itself has no attention layers.
        config.num_attention_heads = config.tensor_model_parallel_size
        config.num_query_groups = config.num_attention_heads
        config.ffn_hidden_size = 5120
        config.gated_linear_unit = False
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.num_moe_experts = None
        config.moe_ffn_hidden_size = None
        config.moe_shared_expert_intermediate_size = None
        config.moe_grouped_gemm = False
        config.is_hybrid_model = False
        config.hybrid_layer_pattern = None
        config.sequence_parallel = False
        config.expert_model_parallel_size = 1
        config.expert_tensor_parallel_size = 1
        return config

    def build_language_model_spec(self, pp_rank: int = 0) -> ModuleSpec:
        config = self._language_config()
        pp_size = config.pipeline_model_parallel_size
        return ModuleSpec(
            module=HybridModel,
            params={
                "config": config,
                "hybrid_stack_spec": hybrid_stack_spec,
                "vocab_size": self.vocab_size,
                "max_sequence_length": self.seq_length,
                "pre_process": pp_rank == 0,
                "post_process": pp_rank == pp_size - 1,
                "hybrid_layer_pattern": self.hybrid_layer_pattern,
                "position_embedding_type": "none",
                "share_embeddings_and_output_weights": False,
                "scatter_embedding_sequence_parallel": False,
            },
        )

    def build_mimo_modality_submodules_spec(self) -> dict[str, ModuleSpec]:
        vision_config = self._vision_config()
        encoder = ModuleSpec(
            module=RADIOEncoderWrapper,
            params={
                "transformer_config": vision_config,
                "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
                "pg_collection": None,
                "img_h": self.img_h,
                "img_w": self.img_w,
                "patch_dim": self.patch_dim,
                "class_token_len": self.class_token_len,
                "drop_class_token": self.drop_vision_class_token,
                "apply_pixel_shuffle": self.pixel_shuffle,
                "force_eval_mode": False,
                "dynamic_resolution": self.dynamic_resolution,
            },
        )
        projector = ModuleSpec(
            module=MultimodalProjector,
            params={
                "config": self._projection_config(),
                "submodules": _projection_submodules(),
                "projector_type": self.vision_projection_type,
                "input_size": 1280 * (4 if self.pixel_shuffle else 1),
            },
        )
        return {
            RADIO_MODALITY: ModuleSpec(
                module=VisionModalitySubmodules,
                params={},
                submodules={
                    "encoders": {RADIO_ENCODER: encoder},
                    "input_projections": [projector],
                },
            )
        }

    def special_token_ids(self) -> dict[str, int]:
        return {RADIO_MODALITY: self.image_token_index}

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        language_config = self._language_config()
        return LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=hybrid_stack_spec,
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.seq_length,
            vision_transformer_config=self._vision_config(),
            vision_transformer_layer_spec=get_vit_layer_with_transformer_engine_spec(),
            drop_vision_class_token=self.drop_vision_class_token,
            vision_projection_config=self._projection_config(),
            vision_projection_layer_spec=_projection_submodules(),
            vision_projection_type=self.vision_projection_type,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            language_position_embedding_type="none",
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            img_h=self.img_h,
            img_w=self.img_w,
            patch_dim=self.patch_dim,
            hybrid_layer_pattern=self.hybrid_layer_pattern,
            image_token_index=self.image_token_index,
            pixel_shuffle=self.pixel_shuffle,
            dynamic_resolution=self.dynamic_resolution,
            class_token_len=self.class_token_len,
            pg_collection=self._pg_collection,
        )


@dataclass
class NemotronMockMIMOProvider(MegatronMIMOProvider):
    """Colocated MIMO adapter which exposes standard-provider config fields."""

    standard_provider: NemotronMockIntegratedProvider = field(
        default_factory=NemotronMockIntegratedProvider
    )

    def __getattr__(self, name: str):
        standard_provider = self.__dict__.get("standard_provider")
        if standard_provider is not None and hasattr(standard_provider, name):
            return getattr(standard_provider, name)
        raise AttributeError(name)

    def validate(self) -> None:
        self.standard_provider.validate()


def build_nemotron_mock_provider(
    provider_config: dict[str, Any],
    *,
    is_refit_destination: bool,
) -> HybridModelProvider | MegatronMIMOProvider:
    """Build the matched policy or refit-destination provider."""
    standard_provider = NemotronMockIntegratedProvider(**provider_config)
    standard_provider.finalize()
    if is_refit_destination:
        return standard_provider

    parallelism = MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=2,
                expert_model_parallel_size=2,
                expert_tensor_parallel_size=1,
                data_parallel_size=1,
                rank_offset=0,
            ),
            RADIO_MODALITY: ModuleParallelismConfig(
                tensor_model_parallel_size=2,
                data_parallel_size=1,
                rank_offset=0,
            ),
        }
    )
    return NemotronMockMIMOProvider(
        standard_provider=copy.deepcopy(standard_provider),
        megatron_mimo_parallelism_config=parallelism,
    )
