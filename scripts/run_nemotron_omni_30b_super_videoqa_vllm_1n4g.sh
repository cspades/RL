#!/usr/bin/env bash
set -euo pipefail

# Run directly inside the NeMo-RL container on one 4-GPU node.
# Uses two GPUs for policy training and two for vLLM generation.
# vLLM twin of run_nemotron_omni_30b_super_videoqa_megatron_1n4g.sh: the model,
# manifest, media geometry, batch shape, optimizer offload and timeouts are held
# identical so only the generation backend differs.
#
# run_nemotron_omni_multimodal_single_controller_1n4g.sh is Megatron-only: it
# hardcodes policy.generation.backend=megatron, refit_transport=mcore and a
# block of mcore_generation_config keys, with no GENERATION_BACKEND switch like
# its 8n4g counterpart has. Trailing Hydra args win, so the vLLM settings below
# are appended rather than substituted. The leftover mcore_generation_config
# keys still reach the config, but nothing reads that block once the backend is
# vllm, and the recipe already declares it, so they resolve without error.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"

DATA_ROOT="${DATA_ROOT:-${NEMORL}/workspace/datasets/super-vl-35-videoqa}"
DATA_FILENAME="${DATA_FILENAME:-train_sav_all_tracks_plus_caprl_exclude6215_hsg_mediafixed_9.jsonl}"
DATA_JSONL="${DATA_JSONL:-${DATA_ROOT}/${DATA_FILENAME}}"
MEDIA_ROOT="${MEDIA_ROOT:-/lustre/fs1/portfolios}"
if [[ ! -s "${DATA_JSONL}" ]]; then
  echo "Missing Super VideoQA dataset: ${DATA_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${NEMORL}/3rdparty/Gym-workspace/Gym/resources_servers/sav_tracks/configs/sav_tracks.yaml" ]]; then
  echo "Missing SA-V Gym configuration under ${NEMORL}/3rdparty/Gym-workspace/Gym." >&2
  exit 1
fi

# The existing 30B one-node runner expects VSTAT's train-gym/val-gym names.
# Stage only stable symlinks; the JSONL contents and absolute media paths are
# unchanged. This is the same staging directory the Megatron twin uses, so the
# two runs read byte-identical inputs.
STAGED_DATA_ROOT="${STAGED_DATA_ROOT:-${NEMORL}/workspace/datasets/super-vl-35-videoqa-omni-30b}"
mkdir -p "${STAGED_DATA_ROOT}"
ln -sfn "${DATA_JSONL}" "${STAGED_DATA_ROOT}/train-gym.jsonl"
ln -sfn "${DATA_JSONL}" "${STAGED_DATA_ROOT}/val-gym.jsonl"

export TASK=vstat
export MODEL_NAME="${MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
export DATA_ROOT="${STAGED_DATA_ROOT}"
export PREPARE_VSTAT=false
export NEMO_RL_VIDEO_TRAIN_JSONL="${STAGED_DATA_ROOT}/train-gym.jsonl"
export NEMO_RL_VIDEO_VAL_JSONL="${STAGED_DATA_ROOT}/val-gym.jsonl"
export NEMO_RL_VIDEO_MEDIA_ROOT="${MEDIA_ROOT}"

export NUM_FRAMES="${NUM_FRAMES:-64}"
export TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE:-2}"
export VIDEO_TARGET_PATCHES="${VIDEO_TARGET_PATCHES:-1024}"
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-65536}"
export INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-65536}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export MIN_GENERATION_TOKENS="${MIN_GENERATION_TOKENS:-4096}"

# Keep the first optimizer step small enough for the 2-GPU policy partition.
# Increase these explicitly when stress-testing frontend concurrency.
export NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-2}"
export NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-2}"
export TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-true}"
export MAX_INFLIGHT_PROMPTS="${MAX_INFLIGHT_PROMPTS:-2}"
export MAX_BUFFERED_ROLLOUTS="${MAX_BUFFERED_ROLLOUTS:-4}"
export MAX_STEPS="${MAX_STEPS:-100000}"

# Full optimizer CPU offload, matching the Megatron twin: the 2-GPU train half
# cannot hold master weights and Adam moments alongside the vocab logits for a
# ~37k-token video sequence. Precision-aware bf16 moments stay on so the
# offloaded state costs 6 bytes/param of host RAM instead of 12.
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
export OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-true}"

# MoE-router and MTP settings pinned to the 32n4g Super VL values so this
# smoke run rehearses the real configuration rather than the recipe defaults,
# which leave the router training with load balancing on and MTP using
# repeated layers with detached heads. The 30B checkpoint is not the Super VL
# teacher, so back any of these out individually if it rejects them.
FREEZE_MOE_ROUTER="${FREEZE_MOE_ROUTER:-true}"
MOE_ROUTER_LOAD_BALANCING_TYPE="${MOE_ROUTER_LOAD_BALANCING_TYPE:-none}"
MOE_ROUTER_BIAS_UPDATE_RATE="${MOE_ROUTER_BIAS_UPDATE_RATE:-0.0}"
MTP_USE_REPEATED_LAYER="${MTP_USE_REPEATED_LAYER:-false}"
MTP_DETACH_HEADS="${MTP_DETACH_HEADS:-false}"

export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export MONITOR_GPUS="${MONITOR_GPUS:-true}"
export RESULTS_DIR="${RESULTS_DIR:-${NEMORL}/workspace/results/nemotron-omni-30b-super-videoqa-vllm-1n4g}"

# Async vLLM has no internal DP, so expert parallelism must span exactly the
# tensor-parallel group. Both default to GEN_GPUS=2 in the shared launcher.
INFER_TP="${INFER_TP:-2}"
INFER_EP="${INFER_EP:-${INFER_TP}}"
if (( INFER_EP != INFER_TP )); then
  echo "vLLM generation requires INFER_EP == INFER_TP (got ${INFER_EP} != ${INFER_TP})." >&2
  exit 1
fi
export INFER_TP INFER_EP

# Chunked prefill must be on: this is a Mamba hybrid, and vLLM's 'align' mamba
# cache mode asserts on it ("Chunked prefill is required for mamba cache mode
# 'align'"), which is a VllmConfig validation error that kills the engine before
# it ever reaches GPU init. Keeping the batched-token budget at the full
# sequence length means a 64K prompt still prefills in one chunk, so enabling it
# satisfies the assertion without changing how this recipe actually prefills.
# max_num_seqs covers the 4 concurrent sequences implied by 2 prompts x 2
# generations; most layers are Mamba, so the attention KV footprint is small.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-${MAX_SEQUENCE_LENGTH}}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-true}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
VLLM_CAP_MAX_TOKENS_TO_CONTEXT="${VLLM_CAP_MAX_TOKENS_TO_CONTEXT:-true}"
VLLM_RESET_ENCODER_CACHE_AFTER_WEIGHT_UPDATE="${VLLM_RESET_ENCODER_CACHE_AFTER_WEIGHT_UPDATE:-false}"
VLLM_REFIT_TIMEOUT_S="${VLLM_REFIT_TIMEOUT_S:-300}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}"

# Cached rows in this manifest hold 64 frames tagged _is_video_frame, which the
# Gym adapter collapses into one input_video manifest; the remaining rows are
# ordinary stills carrying up to 16 separate images. The recipe caps images at
# 32, which the un-collapsed worst case of 64 frames-as-images would exceed if
# the processor takes the non-Nemotron path.
#
# NUM_FRAMES has to be restated on vllm_kwargs.media_io_kwargs.video below, not
# just on vllm_cfg.video. materialize_vllm_video_config() is what normally
# copies the one value across tokenizer, data and engine, but it is only called
# from examples/nemo_gym/run_grpo_nemo_gym.py, and both this launcher and the
# 32n one run examples/run_grpo_single_controller.py, which never calls it.
# vLLM's VideoMediaIO then keeps its own default of 32 and rejects the 64-frame
# cached manifest ("Cached Gym video frame count does not match vLLM's
# requested num_frames"), which surfaces as an HTTP 500 on every rollout.
VLLM_LIMIT_MM_IMAGES="${VLLM_LIMIT_MM_IMAGES:-64}"

# Non-colocated vLLM refit packs weights into bounded NCCL chunks. Producer and
# consumers must use the same ratio to agree on chunk boundaries.
export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.005}"

NEMO_GYM_ROLLOUT_TIMEOUT_S="${NEMO_GYM_ROLLOUT_TIMEOUT_S:-720}"
GENERATION_ROUTER_BACKEND_TIMEOUT_S="${GENERATION_ROUTER_BACKEND_TIMEOUT_S:-600}"
STALL_WATCHDOG_TIMEOUT_S="${STALL_WATCHDOG_TIMEOUT_S:-1200}"

echo "Running Omni 30B Super VideoQA reproduction inside the current container"
echo "  GPUs: 2 policy + 2 vLLM generation (TP=${INFER_TP} EP=${INFER_EP})"
echo "  dataset=${DATA_JSONL}"
echo "  prompts/generations=${NUM_PROMPTS_PER_STEP}/${NUM_GENERATIONS_PER_PROMPT}"
echo "  sequence/new_tokens=${MAX_SEQUENCE_LENGTH}/${MAX_NEW_TOKENS}"
echo "  vLLM: mem_util=${VLLM_GPU_MEMORY_UTILIZATION} max_seqs=${VLLM_MAX_NUM_SEQS} batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} images/prompt=${VLLM_LIMIT_MM_IMAGES}"
echo "  Gym/router/watchdog timeouts=${NEMO_GYM_ROLLOUT_TIMEOUT_S}/${GENERATION_ROUTER_BACKEND_TIMEOUT_S}/${STALL_WATCHDOG_TIMEOUT_S}s"

# No mcore_generation_config keys are set here. Those exist on the Megatron twin
# because MCore reimplements the checkpoint's image preprocessing and has to be
# told to match the HF contract. vLLM runs that HF processor directly, so it is
# the reference those flags were chasing and must not be overridden. The
# reasoning parser is likewise left at the recipe's nemotron_v3, which matches
# this checkpoint.
#
# The generation backend still has to be told the video geometry, and
# hf_config_overrides reaches vLLM for anything on the model config: setup.py
# copies it verbatim into vllm_kwargs.hf_overrides.
#
# Keep vLLM and RL preprocessing explicitly aspect-preserving. At 1024 target
# patches a 16:9 frame resolves to a 24x42 patch grid (252 embeddings per
# tubelet), while square preprocessing produces 32x32 (256). Mixing those modes
# causes placeholder/projected-feature alignment failures during training.
exec bash "${SCRIPT_DIR}/run_nemotron_omni_multimodal_single_controller_1n4g.sh" \
  ++policy.megatron_cfg.freeze_moe_router="${FREEZE_MOE_ROUTER}" \
  ++policy.megatron_cfg.moe_router_load_balancing_type="${MOE_ROUTER_LOAD_BALANCING_TYPE}" \
  ++policy.megatron_cfg.moe_router_bias_update_rate="${MOE_ROUTER_BIAS_UPDATE_RATE}" \
  ++policy.megatron_cfg.mtp_use_repeated_layer="${MTP_USE_REPEATED_LAYER}" \
  ++policy.megatron_cfg.mtp_detach_heads="${MTP_DETACH_HEADS}" \
  ++policy.hf_config_overrides.video_temporal_patch_size="${TEMPORAL_PATCH_SIZE}" \
  ++policy.hf_config_overrides.video_target_num_patches="${VIDEO_TARGET_PATCHES}" \
  ++policy.hf_config_overrides.video_maintain_aspect_ratio=true \
  ++policy.generation.backend=vllm \
  ++policy.generation.refit_transport=null \
  ++policy.generation.bad_words="['<image>','<img>','</img>','<so_embedding>','<so_start>','<so_end>']" \
  ++policy.generation.vllm_cfg.async_engine=true \
  ++policy.generation.vllm_cfg.skip_tokenizer_init=false \
  ++policy.generation.vllm_cfg.tensor_parallel_size="${INFER_TP}" \
  ++policy.generation.vllm_cfg.pipeline_parallel_size=1 \
  ++policy.generation.vllm_cfg.expert_parallel_size="${INFER_EP}" \
  ++policy.generation.vllm_cfg.max_model_len="${MAX_SEQUENCE_LENGTH}" \
  ++policy.generation.vllm_cfg.cap_max_tokens_to_context="${VLLM_CAP_MAX_TOKENS_TO_CONTEXT}" \
  ++policy.generation.vllm_cfg.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  ++policy.generation.vllm_cfg.enforce_eager="${VLLM_ENFORCE_EAGER}" \
  ++policy.generation.vllm_cfg.enable_prefix_caching="${VLLM_ENABLE_PREFIX_CACHING}" \
  ++policy.generation.vllm_cfg.logprobs_mode=raw_logprobs \
  ++policy.generation.vllm_cfg.reset_encoder_cache_after_weight_update="${VLLM_RESET_ENCODER_CACHE_AFTER_WEIGHT_UPDATE}" \
  ++policy.generation.vllm_cfg.video.sampling_style=nemotron_vl \
  ++policy.generation.vllm_cfg.video.num_frames="${NUM_FRAMES}" \
  ++policy.generation.vllm_cfg.video.temporal_patch_size="${TEMPORAL_PATCH_SIZE}" \
  ++policy.generation.vllm_cfg.env_vars.NRL_VIDEO_BACKEND=torchcodec \
  ++policy.generation.vllm_cfg.env_vars.NRL_VIDEO_SAMPLING_STYLE=nemotron_vl \
  ++policy.generation.vllm_cfg.env_vars.NRL_VIDEO_TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE}" \
  ++policy.generation.vllm_cfg.env_vars.VLLM_VIDEO_LOADER_BACKEND=nemotron_vl \
  ++policy.generation.vllm_kwargs.limit_mm_per_prompt.image="${VLLM_LIMIT_MM_IMAGES}" \
  ++policy.generation.vllm_kwargs.limit_mm_per_prompt.video.count=1 \
  ++policy.generation.vllm_kwargs.limit_mm_per_prompt.video.num_frames="${NUM_FRAMES}" \
  ++policy.generation.vllm_kwargs.media_io_kwargs.video.num_frames="${NUM_FRAMES}" \
  ++policy.generation.vllm_kwargs.max_num_seqs="${VLLM_MAX_NUM_SEQS}" \
  ++policy.generation.vllm_kwargs.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  ++policy.generation.vllm_kwargs.allowed_local_media_path="${MEDIA_ROOT}" \
  ++policy.generation.vllm_kwargs.mm_processor_cache_gb=0 \
  ++policy.generation.vllm_kwargs.mamba_ssm_cache_dtype=float32 \
  ++policy.generation.vllm_kwargs.skip_mm_profiling=true \
  ++policy.generation.vllm_kwargs.enable_chunked_prefill=true \
  ++policy.generation.vllm_kwargs.disable_custom_all_reduce=true \
  ++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN \
  ++policy.generation.vllm_kwargs.attention_config.use_trtllm_attention=false \
  ++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false \
  ++policy.generation.vllm_kwargs.kernel_config.moe_backend="${MOE_BACKEND}" \
  ++data.shuffle="${DATA_SHUFFLE}" \
  ++data.default.video_maintain_aspect_ratio=true \
  ++env.nemo_gym.config_paths="[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/mcqa/configs/mcqa.yaml,resources_servers/string_match/configs/string_match.yaml,resources_servers/sav_tracks/configs/sav_tracks.yaml]" \
  ++async_rl.rollout_failure.nemo_gym.rollout_timeout_s="${NEMO_GYM_ROLLOUT_TIMEOUT_S}" \
  ++async_rl.generation_router.enabled=true \
  ++async_rl.generation_router.backend_timeout_s="${GENERATION_ROUTER_BACKEND_TIMEOUT_S}" \
  ++async_rl.generation_router.connect_timeout_s=5 \
  ++async_rl.generation_fleet_health.enabled=true \
  ++async_rl.generation_fleet_health.refit_timeout_s="${VLLM_REFIT_TIMEOUT_S}" \
  ++async_rl.stall_watchdog.stall_timeout_s="${STALL_WATCHDOG_TIMEOUT_S}" \
  ++async_rl.stall_watchdog.stall_action=abort \
  "$@"
