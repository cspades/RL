# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import gc
import threading
import time
import warnings
from dataclasses import replace
from typing import Any, AsyncGenerator, Optional

import requests
import torch
from megatron.core.inference.config import (
    AsyncScheduleMode,
    InferenceConfig,
    KVCacheManagementMode,
    PrefixCachingCoordinatorPolicy,
    VideoProcessingConfig,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.resharding.refit import (
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core.utils import unwrap_model

from nemo_rl.data.video_utils import _CACHED_VIDEO_FRAME_MANIFEST_MAGIC
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.megatron.utils import (
    build_image_preprocessing_config,
    log_gpu_memory,
    resolve_torch_dtype,
)
from nemo_rl.models.megatron.memory_saver import (
    HAVE_TORCH_MEMORY_SAVER,
    pause_inference_weights,
    resume_inference_weights,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name


class MegatronGenerationMixin:
    """Megatron inference lifecycle and generation helpers."""

    inference_model = None
    _colocated_reshard_plan = None

    def _gen_model(self) -> MegatronModule:
        """Return the dedicated inference model when colocated reshard uses one."""
        return self.inference_model if self.inference_model is not None else self.model

    def _init_inference_engine_state(self) -> None:
        """Reset all inference-engine attributes to their uninitialized state."""
        self.llm = None
        self.base_url = None
        self._inference_engine_initialized = False
        self._inference_engine_asleep = True
        self._inference_lifecycle_lock = threading.Lock()

    def _inference_model_and_media_parts(self):
        """Return the language model used by MCore and its optional media model."""
        from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import (
            NemotronOmniModel,
        )
        from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import (
            NemotronVLModel,
        )
        from megatron.core.models.multimodal.llava_model import LLaVAModel

        model = unwrap_model(self._gen_model())
        if isinstance(model, (list, tuple)):
            if len(model) != 1:
                raise NotImplementedError("Virtual pipeline models are not supported.")
            model = model[0]

        if isinstance(model, NemotronVLModel):
            media_model = model.llava_model
            if media_model is None:
                return model, None
            return media_model.language_model, media_model
        if isinstance(model, (NemotronOmniModel, LLaVAModel)):
            return model.language_model, model
        return model, None

    @staticmethod
    def _get_vlm_inference_wrapper_cls(media_model):
        """Select the inference adapter for a model's multimodal contract."""
        from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import (
            NemotronOmniModel,
        )
        from megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper import (
            NemotronOmniInferenceWrapper,
        )
        from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
            VLMInferenceWrapper,
        )

        if isinstance(media_model, NemotronOmniModel):
            return NemotronOmniInferenceWrapper
        return VLMInferenceWrapper

    @staticmethod
    def _load_hf_image_processor(model_name: str):
        """Load the model's HF image processor, whichever auto class it registers."""
        from transformers import AutoImageProcessor, AutoProcessor

        try:
            return AutoImageProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception:
            processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            return processor.image_processor

    def _build_image_preprocessing_config(self, media_model):
        """Build raw-image preprocessing settings."""
        if media_model is None:
            return None

        model_name = self.cfg["model_name"]
        try:
            processor = getattr(self, "processor", None)
            if processor is None:
                processor = self._load_hf_image_processor(model_name)
            else:
                processor = getattr(processor, "image_processor", processor)
            image_preprocessing_config = build_image_preprocessing_config(
                processor,
                dynamic_resolution=bool(
                    getattr(media_model, "dynamic_resolution", True)
                ),
            )
        except Exception as exc:
            warnings.warn(
                f"Could not derive image preprocessing settings for {model_name}: "
                f"{exc}. Requests that carry raw images (base64 image_url over the "
                "HTTP server) will be rejected by the engine.",
                stacklevel=2,
            )
            return None

        model_patch_dim = getattr(media_model, "patch_dim", None)
        if (
            model_patch_dim is not None
            and int(model_patch_dim) != image_preprocessing_config.patch_dim
        ):
            raise ValueError(
                f"{model_name}'s image processor patches at "
                f"{image_preprocessing_config.patch_dim}px but the model expects "
                f"{int(model_patch_dim)}px."
            )
        return image_preprocessing_config

    def _initialize_inference_engine(self, mcore_generation_config: dict) -> None:
        """Initialize the persistent inference engine and client."""
        if self._inference_engine_initialized:
            return

        from megatron.core.inference.apis import MegatronAsyncLLM
        from megatron.core.inference.config import MambaInferenceStateConfig
        from megatron.core.utils import get_attr_wrapped_model

        inference_model, media_model = self._inference_model_and_media_parts()
        pg_collection = get_attr_wrapped_model(self._gen_model(), "pg_collection")
        model_config = inference_model.config

        buffer_size_gb = mcore_generation_config["buffer_size_gb"]
        num_cuda_graphs = mcore_generation_config["num_cuda_graphs"]
        block_size_tokens = mcore_generation_config["block_size_tokens"]
        enable_chunked_prefill = mcore_generation_config["enable_chunked_prefill"]
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config[
            "use_cuda_graphs_for_non_decode_steps"
        ]
        max_tokens = mcore_generation_config["max_tokens"]

        # The value may be overwritten by `recompute_kv_cache_after_weight_updates`.
        kv_cache_management_mode = mcore_generation_config["kv_cache_management_mode"]
        needs_static_kv_pointers = kv_cache_management_mode != "persist"

        materialize_only_last_token_logits = mcore_generation_config[
            "materialize_only_last_token_logits"
        ]
        num_speculative_tokens = mcore_generation_config["num_speculative_tokens"]
        max_requests = mcore_generation_config.get("max_requests")

        # Omni / hybrid: Mamba state lives on the nested language HybridModel.
        mamba_inference_state_config = MambaInferenceStateConfig.from_model(
            inference_model
        )
        is_hybrid_model = mamba_inference_state_config is not None
        if is_hybrid_model:
            if (
                mcore_generation_config.get("mamba_inference_ssm_states_dtype")
                is not None
            ):
                mamba_inference_state_config.ssm_states_dtype = resolve_torch_dtype(
                    mcore_generation_config["mamba_inference_ssm_states_dtype"]
                )
            if (
                mcore_generation_config.get("mamba_inference_conv_states_dtype")
                is not None
            ):
                mamba_inference_state_config.conv_states_dtype = resolve_torch_dtype(
                    mcore_generation_config["mamba_inference_conv_states_dtype"]
                )

        # logging_step_interval is a power-user argument that should be NotRequired.
        logging_step_interval = mcore_generation_config.get("logging_step_interval")
        # This will be fixed in upstream MCore, allowing an argument of `None`.
        if logging_step_interval is None:
            logging_step_interval = 0

        # flashinfer's fused-RoPE kernel only dispatches fp16/bf16 q/k.
        use_flashinfer_fused_rope = model_config.params_dtype in (
            torch.float16,
            torch.bfloat16,
        )

        image_preprocessing_config = self._build_image_preprocessing_config(media_model)
        video_preprocessing_config = None
        if image_preprocessing_config is not None and hasattr(media_model, "patch_dim"):
            video_image_preprocessing_config = image_preprocessing_config
            if "video_target_num_patches" in mcore_generation_config:
                video_image_preprocessing_config = replace(
                    image_preprocessing_config,
                    dynamic_resolution_max_patches=int(
                        mcore_generation_config["video_target_num_patches"]
                    ),
                )
            model_temporal_patch_size = int(
                getattr(
                    getattr(media_model, "vision_model", None),
                    "temporal_patch_dim",
                    1,
                )
            )
            configured_temporal_patch_size = int(
                mcore_generation_config.get(
                    "video_temporal_patch_size", model_temporal_patch_size
                )
            )
            if configured_temporal_patch_size != model_temporal_patch_size:
                raise ValueError(
                    "mcore_generation_config.video_temporal_patch_size must "
                    "match the loaded vision model: "
                    f"{configured_temporal_patch_size} != "
                    f"{model_temporal_patch_size}."
                )
            video_preprocessing_config = VideoProcessingConfig(
                image_config=video_image_preprocessing_config,
                num_frames=int(
                    mcore_generation_config.get("video_num_frames", 8)
                ),
                temporal_patch_size=configured_temporal_patch_size,
                frame_manifest_magic=_CACHED_VIDEO_FRAME_MANIFEST_MAGIC,
            )

        inference_config = InferenceConfig(
            block_size_tokens=block_size_tokens,
            buffer_size_gb=buffer_size_gb,
            num_cuda_graphs=num_cuda_graphs,
            max_tokens=max_tokens,
            max_sequence_length=mcore_generation_config["max_model_len"],
            kv_cache_management_mode=KVCacheManagementMode(kv_cache_management_mode),
            static_kv_memory_pointers=needs_static_kv_pointers,
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=use_flashinfer_fused_rope,
            sampling_backend="flashinfer",
            async_sched_mode=AsyncScheduleMode(
                mcore_generation_config.get("async_sched_mode", "legacy")
            ),
            use_synchronous_zmq_collectives=True,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=mcore_generation_config["enable_prefix_caching"],
            vision_embedding_cache_max_bytes=int(
                mcore_generation_config.get("vision_embedding_cache_max_bytes", 0)
            ),
            prefix_caching_coordinator_policy=PrefixCachingCoordinatorPolicy(
                "first_prefix_block"
            ),
            pg_collection=pg_collection,
            mamba_inference_state_config=mamba_inference_state_config,
            # Reserve more KV-cache space when speculative decoding is enabled.
            mamba_memory_ratio=(
                0.1 + 0.1 * num_speculative_tokens if is_hybrid_model else None
            ),
            logging_step_interval=logging_step_interval,
            num_speculative_tokens=num_speculative_tokens,
            logprobs_mode=mcore_generation_config.get(
                "logprobs_mode", "raw_logprobs"
            ),
            max_requests=max_requests,
            image_preprocessing_config=image_preprocessing_config,
            video_preprocessing_config=video_preprocessing_config,
        )

        if "inference_cuda_graph_scope" in mcore_generation_config:
            model_config.inference_cuda_graph_scope = InferenceCudaGraphScope[
                mcore_generation_config["inference_cuda_graph_scope"]
            ]

        # Identify the Megatron multimodal inference wrapper class for this model.
        if media_model is None:
            llm_model = self._gen_model()
            inference_wrapper_cls = None
        else:
            llm_model = media_model
            inference_wrapper_cls = self._get_vlm_inference_wrapper_cls(media_model)

        self.llm = MegatronAsyncLLM(
            model=llm_model,
            tokenizer=self.megatron_tokenizer,
            inference_config=inference_config,
            use_coordinator=True,
            inference_wrapper_cls=inference_wrapper_cls,
        )

        self._inference_engine_initialized = True
        self._inference_engine_asleep = False
        print(f"[Rank {self.rank}] Initialized persistent inference engine")

    def _sleep(self) -> None:
        """Pause + suspend the engine. No-op if already asleep."""
        with self._inference_lifecycle_lock:
            if self._inference_engine_asleep:
                return
            self.llm.run_sync(self._sleep_engine())
            self._inference_engine_asleep = True
            print(f"[Rank {self.rank}] paused inference engine")

    async def _sleep_engine(self) -> None:
        await self.llm.pause()
        await self.llm.suspend()

    def _wake(self) -> None:
        """Resume + unpause the engine. No-op if already awake."""
        with self._inference_lifecycle_lock:
            if not self._inference_engine_asleep:
                return
            self.llm.run_sync(self._wake_engine())
            self._inference_engine_asleep = False
            print(f"[Rank {self.rank}] resumed inference engine")

    async def _wake_engine(self) -> None:
        await self.llm.resume()
        await self.llm.unpause()

    def _setup_openai_api_server(self) -> str:
        """Start the OpenAI-compatible HTTP server on this worker."""
        from megatron.core.inference.apis import (
            MultimodalPromptConfig,
            ServeConfig,
        )

        from nemo_rl.distributed.virtual_cluster import (
            _get_free_port_local,
            _get_node_ip_local,
        )

        ip = _get_node_ip_local()
        free_port = _get_free_port_local()
        prompt_config = self.cfg["generation"]["mcore_generation_config"].get(
            "multimodal_prompt_config"
        )

        serve_config = ServeConfig(
            port=free_port,
            parsers=self.cfg["generation"]["mcore_generation_config"]["parsers"],
            verbose=False,
            multimodal_prompt_config=(
                MultimodalPromptConfig.from_dict(prompt_config)
                if prompt_config
                else None
            ),
        )
        self.llm.run_sync(self.llm.serve(serve_config, blocking=False))

        base_url = f"http://{ip}:{free_port}/v1"
        max_wait_time = 300
        start_time = time.time()
        with requests.Session() as session:
            while True:
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(
                        f"[Megatron HTTP] Rank {self.rank} OpenAI server failed "
                        f"to start within {max_wait_time}s"
                    )
                try:
                    response = session.get(f"{base_url}/health", timeout=10)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
        return base_url

    def _maybe_start_openai_api_server(self) -> None:
        """Start the OpenAI HTTP server on rank 0 when configured to expose it."""
        rank = torch.distributed.get_rank()
        if (
            self.cfg["generation"]["mcore_generation_config"]["expose_http_server"]
            and rank == 0
        ):
            print(f"[Rank {rank}] Starting HTTP Server")
            self.base_url = self._setup_openai_api_server()
        else:
            print(f"[Rank {rank}] HTTP Server not started")
            self.base_url = None

    def shutdown_inference_engine(self) -> None:
        """Stop the engine and tear down the coordinator + HTTP server."""
        if self.llm is None:
            return
        t = threading.Thread(
            target=asyncio.run, args=(self.llm.shutdown(),), name="mcore-llm-shutdown"
        )
        t.start()
        t.join()
        self.llm = None
        self._inference_engine_initialized = False
        self._inference_engine_asleep = True

    def finish_generation(self) -> None:
        """Wind down a generation cycle."""
        print(f"[Rank {self.rank}] finishing generation", flush=True)
        log_gpu_memory("finish_generation START")

        inference_model, _ = self._inference_model_and_media_parts()
        lang_module = unwrap_model(inference_model)

        if self.is_generation_colocated:
            if self._inference_engine_initialized and not self._inference_engine_asleep:
                self._sleep()
            cuda_graph_impl = self.cfg["generation"]["mcore_generation_config"][
                "cuda_graph_impl"
            ]
            if cuda_graph_impl != "none":
                toggle_cuda_graphs(lang_module, set_to="none")

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(
            rotary_module.forward, "cache_parameters"
        ):
            rotary_module.forward.cache_clear()

        if self.is_generation_colocated:
            if self.inference_model is not None:
                self._offload_inference_model()
            gc.collect()
            torch.cuda.empty_cache()

        log_gpu_memory("finish_generation END")

    def prepare_for_generation(self, tags=None, **kwargs) -> None:
        """Enter inference mode and start (or wake) the inference engine.

        Called in both colocated and non-colocated setups.
        Even in non-colocated mode, Megatron's engine has to be intentionally paused before a refit
        (and its weights are not detachable), so we have to switch modes around every refit.
        """
        log_gpu_memory("prepare_for_generation START")
        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]

        if self._colocated_reshard_plan is not None:
            self._build_colocated_inference_model(self.cfg)

        if self.is_generation_colocated and self.inference_model is None:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
            # Gather parameters collectively before DP inference.
            if self._forward_pre_hook_enabled():
                self._disable_forward_pre_hook_until_next_train_step(param_sync=True)

        if self.inference_model is not None:
            self._reshard_into_inference_model()

        inference_model, _ = self._inference_model_and_media_parts()
        inference_model.config.flash_decode = False
        lang_module = unwrap_model(inference_model)
        lang_module.eval()

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(
            rotary_module.forward, "cache_parameters"
        ):
            rotary_module.forward.cache_clear()

        cuda_graph_impl = mcore_generation_config["cuda_graph_impl"]
        if cuda_graph_impl != "none":
            toggle_cuda_graphs(lang_module, set_to=cuda_graph_impl)

        # Keep the engine paused during weight transfer.
        if tags is None or "weights" not in tags:
            if not self._inference_engine_initialized:
                self._initialize_inference_engine(mcore_generation_config)
                self._maybe_start_openai_api_server()
            else:
                self._wake()

        log_gpu_memory("prepare_for_generation END")

    def report_dp_openai_server_base_url(self) -> Optional[str]:
        """Return this worker's OpenAI server base URL (None if not the leader)."""
        return self.base_url

    def _build_sampling_params(
        self,
        greedy: bool,
        stop_words: Optional[list[str]],
        *,
        return_prompt_tokens: bool = False,
    ) -> SamplingParams:
        """Build mcore SamplingParams for a single request."""
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        return SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=True,
            return_log_probs=True,
            num_tokens_to_generate=self.cfg["generation"]["max_new_tokens"],
            termination_id=self.megatron_tokenizer.eod,
            stop_words=stop_words,
            return_prompt_tokens=return_prompt_tokens,
        )

    def _merge_stop_strings(
        self, batch_stop_strings: Optional[list[Optional[list[str]]]]
    ) -> Optional[list[str]]:
        """Union the config's stop_strings with the given per-sample stop strings."""
        stop_set: set[str] = set()
        if self.cfg["generation"]["stop_strings"]:
            stop_set.update(self.cfg["generation"]["stop_strings"])
        if batch_stop_strings is not None:
            for sample_ss in batch_stop_strings:
                if sample_ss:
                    stop_set.update(sample_ss)
        return list(stop_set) if stop_set else None

    def _collapse_image_spans(self, prompt_tokens: list[int]) -> list[int]:
        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        img_start = tokenizer.convert_tokens_to_ids("<img>")
        img_end = tokenizer.convert_tokens_to_ids("</img>")
        image_token = tokenizer.convert_tokens_to_ids("<image>")

        collapsed = []
        offset = 0
        while offset < len(prompt_tokens):
            if prompt_tokens[offset] != img_start:
                collapsed.append(prompt_tokens[offset])
                offset += 1
                continue
            try:
                end = prompt_tokens.index(img_end, offset + 1)
            except ValueError as exc:
                raise ValueError("Unterminated <img> span in image prompt.") from exc
            collapsed.extend((img_start, image_token, img_end))
            offset = end + 1
        return collapsed

    def _build_prompt_and_multimodal_data(self, data, index: int):
        length = int(data["input_lengths"][index].item())
        expanded_prompt = data["input_ids"][index, :length].tolist()
        imgs, imgs_sizes, num_frames = self._sample_vision_tensors(data, index)
        media_cache_keys = data.get("media_cache_key")
        media_cache_key = (
            media_cache_keys[index] if media_cache_keys is not None else None
        )
        if media_cache_key is not None and not isinstance(media_cache_key, str):
            raise TypeError("media_cache_key entries must be strings or None.")
        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if imgs is None:
            if tokenizer.convert_tokens_to_ids("<img>") in expanded_prompt:
                raise ValueError(
                    "Megatron image generation requires per-sample pixel_values and "
                    "imgs_sizes from RL preprocessing, but none were provided."
                )
            return expanded_prompt, None

        prompt = self._collapse_image_spans(expanded_prompt)
        image_token = tokenizer.convert_tokens_to_ids("<image>")
        num_placeholders = prompt.count(image_token)

        assert imgs_sizes is not None
        is_video = num_frames is not None and bool(torch.any(num_frames > 1).item())
        if is_video:
            if int(num_frames.sum().item()) != int(imgs_sizes.shape[0]):
                raise ValueError(
                    "Video num_frames must partition imgs_sizes exactly: "
                    f"sum(num_frames)={int(num_frames.sum().item())}, "
                    f"imgs_sizes={imgs_sizes.shape[0]}."
                )
            _, media_model = self._inference_model_and_media_parts()
            model_temporal_patch_size = int(
                getattr(
                    getattr(media_model, "vision_model", None),
                    "temporal_patch_dim",
                    1,
                )
            )
            temporal_patch_size = int(
                self.cfg["generation"]["mcore_generation_config"].get(
                    "video_temporal_patch_size", model_temporal_patch_size
                )
            )
            expected_placeholders = sum(
                (int(frame_count) + temporal_patch_size - 1)
                // temporal_patch_size
                for frame_count in num_frames.tolist()
            )
            if num_placeholders != expected_placeholders:
                raise ValueError(
                    f"Video prompt has {num_placeholders} placeholder(s), "
                    f"expected {expected_placeholders} tubelet placeholder(s)."
                )
            multi_modal_data: dict[str, Any] = {
                "video": {
                    "imgs": imgs,
                    "imgs_sizes": imgs_sizes,
                    "num_frames": num_frames,
                }
            }
            if media_cache_key is not None:
                multi_modal_data["media_cache_key"] = media_cache_key
            return prompt, multi_modal_data

        if int(imgs_sizes.shape[0]) != num_placeholders:
            raise ValueError(
                f"Image prompt has {num_placeholders} placeholder(s) "
                f"for {imgs_sizes.shape[0]} imgs_sizes row(s)."
            )
        multi_modal_data = {
            "image": {"imgs": imgs, "imgs_sizes": imgs_sizes}
        }
        if media_cache_key is not None:
            multi_modal_data["media_cache_key"] = media_cache_key
        return prompt, multi_modal_data

    def _sample_vision_tensors(self, data, index: int):
        """Return per-sample vision tensors from RL processor PackedTensors."""
        from nemo_rl.data.multimodal_utils import PackedTensor

        pixel_values = data.get("pixel_values")
        imgs_sizes = data.get("imgs_sizes")
        packed_num_frames = data.get("num_frames")
        if pixel_values is None and imgs_sizes is None:
            if packed_num_frames is not None:
                raise ValueError("num_frames was provided without vision tensors.")
            return None, None, None
        if pixel_values is None or imgs_sizes is None:
            raise ValueError(
                "Megatron image generation requires both pixel_values and imgs_sizes."
            )
        if not isinstance(pixel_values, PackedTensor) or not isinstance(
            imgs_sizes, PackedTensor
        ):
            raise TypeError(
                "Megatron image generation expects pixel_values and imgs_sizes "
                "as per-sample PackedTensor values."
            )
        if packed_num_frames is not None and not isinstance(
            packed_num_frames, PackedTensor
        ):
            raise TypeError(
                "Megatron video generation expects num_frames as a "
                "per-sample PackedTensor value."
            )

        imgs = pixel_values.tensors[index]
        sizes = imgs_sizes.tensors[index]
        num_frames = (
            packed_num_frames.tensors[index]
            if packed_num_frames is not None
            else None
        )
        if imgs is None and sizes is None:
            return None, None, None
        if imgs is None or sizes is None:
            raise ValueError(
                "Megatron image generation requires matching per-sample "
                "pixel_values and imgs_sizes."
            )

        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        if sizes.ndim == 1:
            sizes = sizes.unsqueeze(0)
        if num_frames is not None:
            num_frames = num_frames.to(dtype=torch.int32).reshape(-1)
        return imgs, sizes, num_frames

    def _prepare_data_for_generation(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> tuple[list[list[int]], list[Optional[Any]], list[SamplingParams]]:
        """Build prompts, optional multimodal dictionaries, and sampling params."""
        if data is not None:
            assert isinstance(data, BatchedDataDict), (
                f"data must be a BatchedDataDict, got type: {type(data)}"
            )
            is_right_padded, error_msg = verify_right_padding(
                data, pad_value=self.tokenizer.pad_token_id
            )
            if not is_right_padded:
                warnings.warn(
                    f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
                )

        batch_stop_strings = data.get("stop_strings", [])
        prompts: list[list[int]] = []
        multi_modal_data_list: list[Optional[Any]] = []
        sampling_params: list[SamplingParams] = []
        for i in range(data.size):
            prompt, multi_modal_data = self._build_prompt_and_multimodal_data(data, i)
            sample_stop_strings = (
                batch_stop_strings[i] if i < len(batch_stop_strings) else None
            )
            stop_words = self._merge_stop_strings(
                [sample_stop_strings] if sample_stop_strings else None
            )
            prompts.append(prompt)
            multi_modal_data_list.append(multi_modal_data)
            sampling_params.append(
                self._build_sampling_params(
                    greedy,
                    stop_words,
                    return_prompt_tokens=multi_modal_data is not None,
                )
            )

        return prompts, multi_modal_data_list, sampling_params

    def _parse_result_to_batched_data_dict(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        result: list,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Pack DynamicInferenceRequest results into a GenerationOutputSpec batch."""
        input_lengths = data["input_lengths"]
        input_ids = data["input_ids"]
        batch_size = input_ids.size(0)
        max_gen_seq_len = max(len(x.generated_tokens) for x in result)
        padded_input_length = input_ids.size(1)

        expected_prompt_lengths = [int(length) for length in input_lengths.tolist()]
        inference_prompt_lengths = [
            len(x.prompt_tokens)
            if getattr(x, "prompt_tokens", None) is not None
            else expected_prompt_lengths[i]
            for i, x in enumerate(result)
        ]
        if any(getattr(x, "prompt_tokens", None) is not None for x in result):
            if inference_prompt_lengths != expected_prompt_lengths:
                raise RuntimeError(
                    "Megatron image prompt expansion does not match the training "
                    "processor's input lengths: "
                    f"inference={inference_prompt_lengths}, "
                    f"training={expected_prompt_lengths}."
                )

        max_seq_len = padded_input_length + max_gen_seq_len
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=input_ids.device,
        )

        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        for i in range(batch_size):
            # Take the prompt from the request we submitted rather than from the
            # engine's reply: mcore only echoes prompt_tokens back when
            # SamplingParams.return_prompt_tokens is set, and asking for them would
            # ship the whole prompt over ZMQ for data we already hold.
            prompt_len = input_lengths[i].item()
            generated_tokens = result[i].generated_tokens
            seq_len = prompt_len + len(generated_tokens)
            output_ids_padded[i, :prompt_len] = input_ids[i, :prompt_len]
            output_ids_padded[i, prompt_len:seq_len] = torch.tensor(
                generated_tokens, dtype=torch.long, device=input_ids.device
            )
            generation_lengths[i] = len(generated_tokens)
            unpadded_sequence_lengths[i] = seq_len
            gen_logprobs = result[i].generated_log_probs
            logprobs_padded[i, prompt_len : prompt_len + len(gen_logprobs)] = (
                torch.tensor(
                    gen_logprobs,
                    dtype=torch.float,
                    device=input_ids.device,
                )
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Synchronous batched generation via the mcore data-parallel coordinator.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs with proper padding
                - logprobs: Log probabilities for tokens
                - generation_lengths: Lengths of each response
                - unpadded_sequence_lengths: Lengths of each input + generated sequence
        """
        prompts, multi_modal_data_list, sampling_params = (
            self._prepare_data_for_generation(data, greedy)
        )
        if self.llm is None:
            raise RuntimeError(
                "Inference engine not initialized. Call prepare_for_generation() first."
            )
        result = self.llm.run_sync(
            self._generate_with_persistent_engine(
                prompts,
                multi_modal_data_list,
                sampling_params,
            )
        )

        return self._parse_result_to_batched_data_dict(data, result)

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Streaming generation: yield `(index, batch)` tuples as they complete.

        Args:
            data: BatchedDataDict with input_ids and input_lengths
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec for the single sequence)
        """
        if self.llm is None:
            raise RuntimeError(
                "Inference engine not initialized. Call prepare_for_generation() first."
            )

        async def _generate_single_item(
            index: int,
        ) -> tuple[int, BatchedDataDict[GenerationOutputSpec]]:
            datum = data.get_batch(index, 1)
            prompts, multi_modal_data_list, sampling_params = (
                self._prepare_data_for_generation(datum, greedy)
            )
            result = await self._generate_with_persistent_engine(
                prompts,
                multi_modal_data_list,
                sampling_params,
            )
            output = self._parse_result_to_batched_data_dict(datum, result)
            return (index, output)

        tasks = [
            asyncio.create_task(_generate_single_item(i)) for i in range(data.size)
        ]
        for result in asyncio.as_completed(tasks):
            yield await result

    async def _generate_with_persistent_engine(
        self,
        prompts: list[list[int]],
        multi_modal_data_list: list[Optional[Any]],
        sampling_params: list[SamplingParams],
    ) -> list:
        """Submit one request per sample to the persistent MegatronAsyncLLM (rank 0 only)."""
        dist_rank = torch.distributed.get_rank()
        assert dist_rank == 0, (
            "Only rank 0 submits requests to the inference coordinator"
        )

        print(f"[Rank {dist_rank}] Submitting {len(prompts)} requests to coordinator")

        coros = []
        for prompt, multi_modal_data, request_sampling_params in zip(
            prompts, multi_modal_data_list, sampling_params, strict=True
        ):
            coros.append(
                self.llm.generate(
                    prompt,
                    request_sampling_params,
                    multi_modal_data=multi_modal_data,
                )
            )

        results = await asyncio.gather(*coros)
        print(f"[Rank {dist_rank}] Completed {len(results)} requests")
        return results


class MegatronGenerationRefitMixin:
    """Refit collective, weight transfer, and engine suspend/resume around refits."""

    def init_collective_mcore_generation(
        self,
        ip: str,
        port: int,
        world_size: int,
        rank_offset: int,
        refit_backend: str = "gloo",
    ) -> None:
        """Initialize the refit collective for non-colocated weight transfer.

        Args:
            ip: IP address for the process group rendezvous.
            port: Port for the process group rendezvous.
            world_size: Total world size (train + inference workers).
            rank_offset: Offset for this side's ranks (`train_world_size` for inference).
            refit_backend: Copy-service backend ("gloo", "nccl", or "nvshmem").
        """
        from torch.distributed.distributed_c10d import (
            PrefixStore,
            ProcessGroup,
            ProcessGroupGloo,
            _world,
        )

        local_rank = torch.distributed.get_rank()
        global_rank = local_rank + rank_offset

        # port+1 to avoid collision with the caller's rendezvous on `port`.
        store = torch.distributed.TCPStore(
            host_name=ip,
            port=port + 1,
            world_size=world_size,
            is_master=(global_rank == 0),
        )

        group_name = "refit"
        pg_prefix_store = PrefixStore(f"{group_name}/", store)

        # Training and inference workers run in separate torch.distributed worlds.
        # The public APIs (new_group, init_process_group) assume all ranks belong to one world;
        # new_group validates ranks against the default PG, and init_process_group can only
        # be called once. We construct the PG manually using the same internal pattern as
        # _new_process_group_helper, skipping the single-world assumptions.
        pg = ProcessGroup(pg_prefix_store, global_rank, world_size)
        gloo_store = PrefixStore("cpu/", pg_prefix_store)
        gloo_backend = ProcessGroupGloo(gloo_store, global_rank, world_size)
        gloo_backend._set_sequence_number_for_group()
        pg._register_backend(
            torch.device("cpu"),
            ProcessGroup.BackendType.GLOO,
            gloo_backend,
        )
        pg._set_default_backend(ProcessGroup.BackendType.GLOO)

        # The NCCL copy service moves the actual weight bytes with CUDA-tensor P2P
        # (`torch.distributed.batch_isend_irecv`), which needs an NCCL backend
        # registered for the cuda device on this cross-world PG. GLOO stays the
        # default backend so the object collectives in `prepare_swap_model_weights`
        # (all_gather_object / broadcast_object_list) keep using CPU tensors.
        if refit_backend == "nccl":
            from torch.distributed.distributed_c10d import ProcessGroupNCCL

            # Ensure the NCCL communicator binds to this rank's own GPU.
            torch.cuda.set_device(torch.cuda.current_device())
            nccl_store = PrefixStore("cuda/", pg_prefix_store)
            nccl_options = ProcessGroupNCCL.Options()
            nccl_backend = ProcessGroupNCCL(
                nccl_store, global_rank, world_size, nccl_options
            )
            nccl_backend._set_sequence_number_for_group()
            pg._register_backend(
                torch.device("cuda"),
                ProcessGroup.BackendType.NCCL,
                nccl_backend,
            )

        pg._set_group_name(group_name)

        self.refit_pg = pg

        # Register in torch.distributed's global state so that high-level ops
        # (all_gather_object, broadcast_object_list) work with this PG.
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
        _world.pg_map[pg] = ("gloo", pg_prefix_store)
        _world.pg_names[pg] = group_name

        if refit_backend == "nvshmem":
            from megatron.core.resharding.copy_services.nvshmem_copy_service import (
                NVSHMEMCopyService,
            )

            self.refit_copy_service = NVSHMEMCopyService(group=self.refit_pg)
        elif refit_backend == "nccl":
            from megatron.core.resharding.copy_services.nccl_copy_service import (
                NCCLCopyService,
            )

            self.refit_copy_service = NCCLCopyService(group=self.refit_pg)
        else:
            from megatron.core.resharding.copy_services.gloo_copy_service import (
                GlooCopyService,
            )

            self.refit_copy_service = GlooCopyService(group=self.refit_pg)

        is_source = rank_offset == 0
        # Cache for later refit calls (swap_weights_via_reshard).
        self.refit_dst_rank_offset = (
            torch.distributed.get_world_size() if is_source else rank_offset
        )

        # Build and cache the reshard plan (and any MXFP8 transforms) collectively.
        # All participating ranks (training + generation) call this simultaneously.
        prepare_swap_model_weights(
            src_model=self.model if is_source else None,
            target_model=None if is_source else self.model,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=self.refit_dst_rank_offset,
        )

    def preinit_nvshmem_collective(self) -> None:
        """Initialize NVShmem collectively before any weight transfer.

        Must be called on ALL participating ranks (training + inference) simultaneously,
        after `prepare_for_generation()` has completed and the CG has been recorded.
        The `NVSHMEMCopyService` lazy init can corrupt CUDA graph state.
        """
        if not hasattr(self, "refit_copy_service"):
            return
        if not hasattr(self.refit_copy_service, "_ensure_initialized"):
            return
        self.refit_copy_service._ensure_initialized()

    def swap_weights_via_reshard(self, is_source: bool) -> bool:
        """Transfer weights using Megatron's `swap_model_weights` API.

        Args:
            is_source: True for training workers (senders), False for inference workers (receivers).

        Returns:
            True on success.
        """
        src_model = self.model if is_source else None
        dst_model = None if is_source else self.model

        swap_model_weights(
            src_model,
            dst_model,
            refit_method=self.refit_copy_service,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=self.refit_dst_rank_offset,
        )

        return True

    def _onload_inference_model(self) -> None:
        """Restore the colocated inference weights to GPU before resharding."""
        if not self._inference_model_offloaded:
            return

        resume_inference_weights()
        self._inference_model_offloaded = False

    def _offload_inference_model(self) -> None:
        """Offload the colocated inference weights while training runs."""
        if (
            self.inference_model is None
            or self._inference_model_offloaded
            or not HAVE_TORCH_MEMORY_SAVER
        ):
            return
        pause_inference_weights()
        self._inference_model_offloaded = True

    def _reshard_into_inference_model(self) -> None:
        """Reshard training weights into the colocated inference-layout model."""
        inference_model = self.inference_model
        if inference_model is None:
            return

        self._onload_inference_model()
        self.model = self.move_model(
            self.model, "cuda", move_params=True, move_grads=False
        )
        torch.cuda.synchronize()

        if self.should_disable_forward_pre_hook and self._forward_pre_hook_enabled():
            self._disable_forward_pre_hook_until_next_train_step(param_sync=True)

        if not self._swap_weights_plan_prepared:
            prepare_swap_model_weights(
                src_model=self.model,
                target_model=inference_model,
                group=None,
                src_rank_offset=0,
                dst_rank_offset=0,
            )
            self._swap_weights_plan_prepared = True

        swap_model_weights(
            self.model,
            inference_model,
            refit_method=self.cfg["generation"]["mcore_generation_config"][
                "refit_backend"
            ],
            group=None,
            src_rank_offset=0,
            dst_rank_offset=0,
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=True, move_grads=False
        )
        torch.cuda.synchronize()

    def suspend_for_refit(self) -> None:
        """Pause+suspend the inference engine before a weight refit."""
        if not self._inference_engine_initialized:
            return
        self._sleep()
        torch.cuda.synchronize()

    def resume_after_refit(self) -> None:
        """Resume+unpause the inference engine after a weight refit."""
        if not self._inference_engine_initialized:
            return
        self._wake()
