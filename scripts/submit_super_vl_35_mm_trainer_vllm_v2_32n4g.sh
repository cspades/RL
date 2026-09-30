#!/usr/bin/env bash
set -euo pipefail

# NeMo-RL V2 / SingleController / vLLM port of:
#   gui-grpo-experiments/configs/four_source_video10k_freezevision_20260922.yaml
#
# This is the vLLM twin of the Megatron-Inference V2 launcher. It retains the
# V1 unified-teacher dataset, policy topology, optimizer, schedule, Gym setup,
# router replay, and one-step asynchronous lookahead.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
CONTAINER_NEMORL="${CONTAINER_NEMORL:-/opt/nemo-rl}"

# Patch only Python model code in the container's vLLM 0.29 worker environment.
export VLLM_RUNTIME_PATCH_SCRIPT="${VLLM_RUNTIME_PATCH_SCRIPT:-${CONTAINER_NEMORL}/scripts/patch_vllm_super_omni_radio_layernorm_0_29.py}"

# Share one cache root; MBridge keys each unique source model into its own subdirectory.
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${NEMORL}/workspace/cache/nemo-rl-omni/megatron-checkpoints-super-vl-35-unified-final-ln-v2}"

MODEL_REL="${MODEL_REL:-workspace/models/super-vl-35-rlvr-v43-falcon-r3-20260905/hf}"
DATA_REL="${DATA_REL:-workspace/datasets/mm-trainer-unified}"
DATA_FILENAME="${DATA_FILENAME:-training.jsonl}"
MODEL_HOST_PATH="${NEMORL}/${MODEL_REL}"
DATA_HOST_PATH="${NEMORL}/${DATA_REL}/${DATA_FILENAME}"

export MM_TRAINER_MODEL_PATH="${MM_TRAINER_MODEL_PATH:-${CONTAINER_NEMORL}/${MODEL_REL}}"
export MM_TRAINER_DATA_PATH="${MM_TRAINER_DATA_PATH:-${CONTAINER_NEMORL}/${DATA_REL}/${DATA_FILENAME}}"
export MM_TRAINER_MEDIA_ROOT="${MM_TRAINER_MEDIA_ROOT:-/lustre}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

if [[ ! -f "${MODEL_HOST_PATH}/config.json" ||
      ! -f "${MODEL_HOST_PATH}/chat_template.jinja" ]]; then
  echo "Missing Super-VL 3.5 HF checkpoint or chat template: ${MODEL_HOST_PATH}" >&2
  exit 1
fi
if [[ ! -s "${DATA_HOST_PATH}" ]]; then
  echo "Missing mixed training manifest: ${DATA_HOST_PATH}" >&2
  exit 1
fi
if [[ ! -d /home/svc-dss/cache ]]; then
  echo "Missing DSS image cache mount source: /home/svc-dss/cache" >&2
  exit 1
fi

GYM_ROOT="${NEMORL}/3rdparty/Gym-workspace/Gym"
GYM_CONFIGS=(
  responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
  resources_servers/gui_coordinate/configs/gui_coordinate.yaml
  resources_servers/math_with_judge/configs/math_with_judge.yaml
  resources_servers/mcqa/configs/mcqa.yaml
  resources_servers/string_match/configs/string_match.yaml
  responses_api_agents/image_tools_agent/configs/image_tools_agent.yaml
  resources_servers/sav_tracks/configs/sav_tracks.yaml
)
for config_path in "${GYM_CONFIGS[@]}"; do
  if [[ ! -f "${GYM_ROOT}/${config_path}" ]]; then
    echo "Missing NeMo-Gym config: ${GYM_ROOT}/${config_path}" >&2
    exit 1
  fi
done

export TASK=vstat
export GENERATION_BACKEND=vllm
export CONFIG="examples/configs/recipes/vlm/super_vl_35_mm_trainer_megatron_v2.yaml"
export MODEL_NAME="${MM_TRAINER_MODEL_PATH}"
export DATA_ROOT="${CONTAINER_NEMORL}/${DATA_REL}"
export NEMO_RL_VIDEO_TRAIN_JSONL="${MM_TRAINER_DATA_PATH}"
export NEMO_RL_VIDEO_VAL_JSONL="${MM_TRAINER_DATA_PATH}"
export NEMO_RL_VIDEO_MEDIA_ROOT="${MM_TRAINER_MEDIA_ROOT}"
export PREPARE_VSTAT=false

export NUM_NODES="${NUM_NODES:-32}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-16}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-8}"

# Match the V1 unified-teacher policy and vLLM parallelism.
export POLICY_TP="${POLICY_TP:-2}"
export POLICY_CP="${POLICY_CP:-2}"
export POLICY_EP="${POLICY_EP:-16}"
export INFER_TP="${INFER_TP:-4}"
export INFER_EP="${INFER_EP:-4}"

export NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-128}"
export NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-16}"
export TRAIN_GBS="${TRAIN_GBS:-2048}"
export MAX_STEPS="${MAX_STEPS:-125}"
export MAX_LOOKAHEAD_VERSIONS="${MAX_LOOKAHEAD_VERSIONS:-1}"
export MAX_INFLIGHT_PROMPTS="${MAX_INFLIGHT_PROMPTS:-128}"
export MAX_BUFFERED_ROLLOUTS="${MAX_BUFFERED_ROLLOUTS:-256}"

export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-65536}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
export TRAIN_MB_TOKENS="${TRAIN_MB_TOKENS:-49152}"
export LOGPROB_MB_TOKENS="${LOGPROB_MB_TOKENS:-65536}"
export NUM_FRAMES="${NUM_FRAMES:-64}"
export TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE:-2}"
export VIDEO_TARGET_PATCHES="${VIDEO_TARGET_PATCHES:-1024}"
export MIN_GENERATION_TOKENS="${MIN_GENERATION_TOKENS:-32768}"

# Match the V1 vLLM serving configuration.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
export VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
export VLLM_CAP_MAX_TOKENS_TO_CONTEXT="${VLLM_CAP_MAX_TOKENS_TO_CONTEXT:-true}"
export VLLM_REFIT_TIMEOUT_S="${VLLM_REFIT_TIMEOUT_S:-300}"
export VLLM_TRITON_FORCE_FIRST_CONFIG="${VLLM_TRITON_FORCE_FIRST_CONFIG:-1}"
export VLLM_LIMIT_MM_IMAGES="${VLLM_LIMIT_MM_IMAGES:-64}"
export MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutlass}"
export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.006}"

# Match the V1 precision-aware optimizer without changing its FP32 state.
export USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-true}"
export EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-float32}"
export EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-float32}"
export STORE_PARAM_REMAINDERS="${STORE_PARAM_REMAINDERS:-false}"
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.0}"
export OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-false}"
export OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-false}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export NEMO_GYM_ROLLOUT_TIMEOUT_S="${NEMO_GYM_ROLLOUT_TIMEOUT_S:-2100}"
export GENERATION_ROUTER_BACKEND_TIMEOUT_S="${GENERATION_ROUTER_BACKEND_TIMEOUT_S:-1800}"
export STALL_WATCHDOG_TIMEOUT_S="${STALL_WATCHDOG_TIMEOUT_S:-12600}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

export CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-true}"
export CHECKPOINT_SAVE_PERIOD="${CHECKPOINT_SAVE_PERIOD:-10}"
export CHECKPOINT_KEEP_TOP_K="${CHECKPOINT_KEEP_TOP_K:-1}"
export CHECKPOINT_ASYNC_SAVE="${CHECKPOINT_ASYNC_SAVE:-false}"
export RESULTS_DIR="${RESULTS_DIR:-${NEMORL}/workspace/results/super-vl-35-mm-trainer-vllm-v2}"
export MM_TRAINER_RESULTS_DIR="${MM_TRAINER_RESULTS_DIR:-${RESULTS_DIR}}"
export MM_TRAINER_GYM_VENV_DIR="${MM_TRAINER_GYM_VENV_DIR:-/opt/gym_venvs}"
export MM_TRAINER_WANDB_ENTITY="${MM_TRAINER_WANDB_ENTITY:-nvidia}"
export MM_TRAINER_WANDB_PROJECT="${MM_TRAINER_WANDB_PROJECT:-mllm-v2-super35vl-unified-teacher}"
export MM_TRAINER_WANDB_NAME="${MM_TRAINER_WANDB_NAME:-super-vl-35-mm-trainer-vllm-v2}"
export MM_TRAINER_WANDB_ID="${MM_TRAINER_WANDB_ID:-${MM_TRAINER_WANDB_NAME}}"
export WANDB_PROJ="${WANDB_PROJ:-${MM_TRAINER_WANDB_PROJECT}}"
export WANDB_NAME="${WANDB_NAME:-${MM_TRAINER_WANDB_NAME}}"
export JOB_NAME="${JOB_NAME:-super-vl-35-mm-trainer-vllm-v2-32n4g}"
export SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/asolergibert/RL/images/nemo-rl-nightly-gym.sqsh}"

# Preserve the DSS cache mount used by the V1 run.
export MOUNTS="${MOUNTS:-/scratch:/scratch,/lustre:/lustre,/home/svc-dss/cache:/home/svc-dss/cache:ro}"

USER_EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
export EXTRA_OVERRIDES="\
++policy.megatron_cfg.freeze_vision_model=true \
++policy.megatron_cfg.freeze_vision_projection=true \
++policy.megatron_cfg.freeze_sound_encoder=true \
++policy.megatron_cfg.freeze_sound_projection=true \
++policy.megatron_cfg.radio_force_cpe_eval_mode=true \
++policy.megatron_cfg.freeze_moe_router=true \
++policy.megatron_cfg.moe_router_load_balancing_type=none \
++policy.megatron_cfg.moe_router_bias_update_rate=0.001 \
++policy.megatron_cfg.mtp_num_layers=0 \
++policy.megatron_cfg.mtp_loss_scaling_factor=0.0 \
++policy.megatron_cfg.mtp_use_repeated_layer=false \
++policy.megatron_cfg.mtp_detach_heads=true \
++policy.megatron_cfg.fp32_lm_head=true \
++policy.megatron_cfg.recompute_granularity=full \
++policy.megatron_cfg.optimizer.lr=3.0e-6 \
++policy.megatron_cfg.optimizer.min_lr=5.0e-7 \
++policy.megatron_cfg.optimizer.adam_beta2=0.99 \
++policy.megatron_cfg.scheduler.lr_decay_iters=60 \
++policy.megatron_cfg.scheduler.lr_decay_style=cosine \
++policy.megatron_cfg.scheduler.lr_warmup_iters=10 \
++policy.megatron_cfg.scheduler.lr_warmup_init=3.0e-8 \
++policy.megatron_cfg.env_vars.NCCL_NVLS_ENABLE=\"'${NCCL_NVLS_ENABLE}'\" \
++policy.router_replay.enabled=true \
++token_capture.enabled=true \
++token_capture.defer_routed_experts_to_policy=false \
++policy.sequence_packing.train_mb_tokens=${TRAIN_MB_TOKENS} \
++policy.sequence_packing.logprob_mb_tokens=${LOGPROB_MB_TOKENS} \
++policy.sequence_packing.microbatch_order=largest_first \
++policy.hf_config_overrides.video_temporal_patch_size=${TEMPORAL_PATCH_SIZE} \
++policy.hf_config_overrides.video_target_num_patches=${VIDEO_TARGET_PATCHES} \
++policy.hf_config_overrides.video_maintain_aspect_ratio=false \
++policy.generation.temperature=1.0 \
++policy.generation.top_p=1.0 \
++policy.generation.bad_words=\"['<image>','<img>','</img>','<so_embedding>','<so_start>','<so_end>']\" \
++policy.generation.vllm_cfg.enable_prefix_caching=${VLLM_ENABLE_PREFIX_CACHING} \
++policy.generation.vllm_cfg.fp32_lm_head=true \
++policy.generation.vllm_cfg.env_vars.VLLM_TRITON_FORCE_FIRST_CONFIG=\"'${VLLM_TRITON_FORCE_FIRST_CONFIG}'\" \
++policy.generation.vllm_kwargs.enable_return_routed_experts=true \
++policy.generation.vllm_kwargs.limit_mm_per_prompt.image=${VLLM_LIMIT_MM_IMAGES} \
++policy.generation.vllm_kwargs.limit_mm_per_prompt.video.count=1 \
++policy.generation.vllm_kwargs.limit_mm_per_prompt.video.num_frames=${NUM_FRAMES} \
++policy.generation.vllm_kwargs.media_io_kwargs.video.num_frames=${NUM_FRAMES} \
++policy.generation.vllm_kwargs.max_num_seqs=${VLLM_MAX_NUM_SEQS} \
++policy.generation.vllm_kwargs.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} \
++policy.generation.vllm_kwargs.enable_chunked_prefill=true \
++policy.generation.vllm_kwargs.allowed_local_media_path=${NEMO_RL_VIDEO_MEDIA_ROOT} \
++policy.generation.vllm_kwargs.attention_backend=FLASHINFER \
++policy.generation.vllm_kwargs.mm_encoder_attn_backend=FLASH_ATTN \
++policy.generation.vllm_kwargs.compilation_config.backend=eager \
++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE \
++data.shuffle=false \
++data.num_workers=1 \
++data.default.video_sampling_style=nemotron_vl \
++data.default.video_maintain_aspect_ratio=false \
++data.validation=null \
++grpo.max_num_epochs=1 \
++grpo.max_num_steps=${MAX_STEPS} \
++grpo.max_rollout_turns=1 \
++grpo.overlong_filtering=true \
++grpo.seq_logprob_error_threshold=2 \
++grpo.skip_reference_policy_logprobs_calculation=true \
++grpo.deduplicate_multimodal_data=false \
++loss_fn.token_level_loss=false \
++loss_fn.sequence_level_importance_ratios=false \
++loss_fn.ratio_clip_min=0.2 \
++loss_fn.ratio_clip_max=0.28 \
++loss_fn.truncated_importance_sampling_type=tis \
++loss_fn.truncated_importance_sampling_ratio=5.0 \
++loss_fn.truncated_importance_sampling_ratio_min=0.2 \
++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=${NEMO_GYM_ROLLOUT_TIMEOUT_S} \
++async_rl.generation_router.enabled=true \
++async_rl.generation_router.backend_timeout_s=${GENERATION_ROUTER_BACKEND_TIMEOUT_S} \
++async_rl.generation_router.connect_timeout_s=5 \
++async_rl.generation_fleet_health.enabled=true \
++async_rl.stall_watchdog.stall_timeout_s=${STALL_WATCHDOG_TIMEOUT_S} \
++async_rl.stall_watchdog.stall_action=abort \
++env.nemo_gym.skip_venv_if_present=true \
++env.nemo_gym.nemo_gym_log_dir=${MM_TRAINER_RESULTS_DIR}/logs/nemo_gym \
++checkpointing.checkpoint_dir=${MM_TRAINER_RESULTS_DIR}/checkpoints \
++checkpointing.save_period=${CHECKPOINT_SAVE_PERIOD} \
++checkpointing.keep_top_k=${CHECKPOINT_KEEP_TOP_K} \
++checkpointing.save_optimizer=true \
++checkpointing.save_data_plane=true \
++policy.megatron_cfg.checkpoint.async_save=${CHECKPOINT_ASYNC_SAVE} \
++logger.log_dir=${MM_TRAINER_RESULTS_DIR}/logs \
++logger.tensorboard_enabled=true \
++logger.wandb.entity=${MM_TRAINER_WANDB_ENTITY} \
++logger.wandb.project=${MM_TRAINER_WANDB_PROJECT} \
++logger.wandb.name=${MM_TRAINER_WANDB_NAME}-\${NRL_SLURM_JOB_ID} \
++logger.wandb.id=${MM_TRAINER_WANDB_ID}-\${NRL_SLURM_JOB_ID} \
++logger.wandb.resume=never \
${USER_EXTRA_OVERRIDES}"

exec bash "${SCRIPT_DIR}/submit_nemotron_omni_multimodal_single_controller_8n4g.sh" "$@"
