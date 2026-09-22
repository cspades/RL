#!/usr/bin/env bash
set -euo pipefail

# V2 SingleController/Megatron-inference launcher for the validated Super 3.5
# video-teacher run recorded in super_vl_35_videoqa_recipe.yaml.
#
# Copy the checkpoint and JSONL into the host paths below before submitting.
# The whole repository is mounted at /opt/nemo-rl, and /lustre is mounted at
# the same absolute path inside the container. Media may therefore remain
# anywhere under /lustre when the JSONL contains its absolute /lustre paths.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
CONTAINER_NEMORL="${CONTAINER_NEMORL:-/opt/nemo-rl}"

MODEL_REL="${MODEL_REL:-workspace/models/super-vl-35-video-teacher-step-120/hf}"
DATA_REL="${DATA_REL:-workspace/datasets/super-vl-35-videoqa}"
DATA_FILENAME="${DATA_FILENAME:-train_sav_all_tracks_plus_caprl_exclude6215_hsg_mediafixed_9.jsonl}"
MODEL_HOST_PATH="${NEMORL}/${MODEL_REL}"
DATA_HOST_ROOT="${NEMORL}/${DATA_REL}"

export VIDEO_TEACHER_MODEL_PATH="${VIDEO_TEACHER_MODEL_PATH:-${CONTAINER_NEMORL}/${MODEL_REL}}"
export VIDEO_TEACHER_DATA_PATH="${VIDEO_TEACHER_DATA_PATH:-${CONTAINER_NEMORL}/${DATA_REL}/${DATA_FILENAME}}"
# The manifest contains absolute paths from multiple shared media trees
# (including llmservice and nemotron); this is their mounted common allowlist.
export VIDEO_TEACHER_MEDIA_ROOT="${VIDEO_TEACHER_MEDIA_ROOT:-/lustre/fs1/portfolios}"

if [[ ! -f "${MODEL_HOST_PATH}/config.json" ]]; then
  echo "Missing copied HF checkpoint: ${MODEL_HOST_PATH}" >&2
  exit 1
fi
if [[ ! -s "${DATA_HOST_ROOT}/${DATA_FILENAME}" ]]; then
  echo "Missing copied training manifest: ${DATA_HOST_ROOT}/${DATA_FILENAME}" >&2
  exit 1
fi
if [[ ! -f "${NEMORL}/3rdparty/Gym-workspace/Gym/resources_servers/sav_tracks/app.py" ]]; then
  echo "Missing SA-V verifier: ${NEMORL}/3rdparty/Gym-workspace/Gym/resources_servers/sav_tracks" >&2
  exit 1
fi

export TASK=vstat
export GENERATION_BACKEND=megatron
export CONFIG="examples/configs/recipes/vlm/super_vl_35_videoqa_recipe.yaml"
export MODEL_NAME="${VIDEO_TEACHER_MODEL_PATH}"
export DATA_ROOT="${CONTAINER_NEMORL}/${DATA_REL}"
export NEMO_RL_VIDEO_TRAIN_JSONL="${VIDEO_TEACHER_DATA_PATH}"
export NEMO_RL_VIDEO_VAL_JSONL="${VIDEO_TEACHER_DATA_PATH}"
export NEMO_RL_VIDEO_MEDIA_ROOT="${VIDEO_TEACHER_MEDIA_ROOT}"
export PREPARE_VSTAT=false

# Match mtpenalty223151: 32x4 GPUs, split evenly between policy and generation.
export NUM_NODES="${NUM_NODES:-32}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-16}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-8}"
export NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-128}"
export NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-16}"
export TRAIN_GBS="${TRAIN_GBS:-2048}"
export MAX_STEPS="${MAX_STEPS:-1000000}"

# Rollout-pump concurrency, counted in prompt groups rather than requests.
# MAX_BUFFERED_ROLLOUTS is how many finished-but-untrained groups may sit in the
# DataPlane. The pump takes a buffer permit before the inflight one, making it
# the outer backpressure bound that inflight can never exceed.
export MAX_INFLIGHT_PROMPTS="${MAX_INFLIGHT_PROMPTS:-32}"
export MAX_BUFFERED_ROLLOUTS="${MAX_BUFFERED_ROLLOUTS:-256}"

# Frame manifests keep frontend work small, and each ASGI replica can await many
# generations concurrently. Four replicas per backend provide CPU parallelism
# without creating 512 forked HTTP processes and ZMQ clients across the fleet.
export HTTP_SERVER_NUM_REPLICAS="${HTTP_SERVER_NUM_REPLICAS:-4}"

# Timeouts
export NEMO_GYM_ROLLOUT_TIMEOUT_S="${NEMO_GYM_ROLLOUT_TIMEOUT_S:-2100}"
export GENERATION_ROUTER_BACKEND_TIMEOUT_S="${GENERATION_ROUTER_BACKEND_TIMEOUT_S:-1800}"
export STALL_WATCHDOG_TIMEOUT_S="${STALL_WATCHDOG_TIMEOUT_S:-12600}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

# Match the reference policy and its vLLM replica topology using MCore replicas.
export POLICY_TP="${POLICY_TP:-2}"
export POLICY_EP="${POLICY_EP:-16}"
export POLICY_CP="${POLICY_CP:-1}"
export INFER_TP="${INFER_TP:-4}"
export INFER_EP="${INFER_EP:-4}"
export REFIT_TRANSPORT="${REFIT_TRANSPORT:-mcore}"
export REFIT_BACKEND="${REFIT_BACKEND:-nccl}"

export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-65536}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
export INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-65536}"
# Match this checkpoint's HF raw-image processor without changing MCore defaults.
export IMAGE_DYNAMIC_RESOLUTION_ROUNDING_MODE="${IMAGE_DYNAMIC_RESOLUTION_ROUNDING_MODE:-round_plus_half}"
export IMAGE_DYNAMIC_RESOLUTION_RESIZE_MODE="${IMAGE_DYNAMIC_RESOLUTION_RESIZE_MODE:-torch_bicubic_antialias}"
export TRAIN_MB_TOKENS="${TRAIN_MB_TOKENS:-49152}"
export LOGPROB_MB_TOKENS="${LOGPROB_MB_TOKENS:-65536}"
export NUM_FRAMES="${NUM_FRAMES:-64}"
export TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE:-2}"
export VIDEO_TARGET_PATCHES="${VIDEO_TARGET_PATCHES:-1024}"
export MIN_GENERATION_TOKENS="${MIN_GENERATION_TOKENS:-16384}"

export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.0}"
export OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-false}"
export OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-false}"
export USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-false}"
export OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-false}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-false}"
export EMPTY_UNUSED_MEMORY_LEVEL="${EMPTY_UNUSED_MEMORY_LEVEL:-2}"
export BUFFER_SIZE_GB="${BUFFER_SIZE_GB:-16}"
export PREFIX_CACHING_MAMBA_GB="${PREFIX_CACHING_MAMBA_GB:-32}"
export VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-17179869184}"

export CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-false}"
export RESULTS_DIR="${RESULTS_DIR:-${NEMORL}/workspace/results/super-vl-35-videoqa-megatron-v2}"
export VIDEO_TEACHER_RESULTS_DIR="${VIDEO_TEACHER_RESULTS_DIR:-${RESULTS_DIR}}"
export VIDEO_TEACHER_GYM_VENV_DIR="${VIDEO_TEACHER_GYM_VENV_DIR:-${CONTAINER_NEMORL}/workspace/gym_venvs/super-vl-35-videoqa}"
export VIDEO_TEACHER_WANDB_PROJECT="${VIDEO_TEACHER_WANDB_PROJECT:-mllm-rl-dev}"
export VIDEO_TEACHER_WANDB_NAME="${VIDEO_TEACHER_WANDB_NAME:-super-vl-35-videoqa-megatron-v2}"
export VIDEO_TEACHER_WANDB_ID="${VIDEO_TEACHER_WANDB_ID:-${VIDEO_TEACHER_WANDB_NAME}}"
export WANDB_PROJ="${WANDB_PROJ:-${VIDEO_TEACHER_WANDB_PROJECT}}"
export WANDB_NAME="${WANDB_NAME:-${VIDEO_TEACHER_WANDB_NAME}}"
export JOB_NAME="${JOB_NAME:-super-vl-35-videoqa-megatron-v2-32n4g}"
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/asolergibert/RL/images/nemo-rl-nightly-gym.sqsh}"

# TASK=vstat supplies generic Nano-Omni defaults. Restore the validated Super
# run's frozen modules, media geometry, parser, and output paths
# after those defaults are applied. Router replay is deliberately disabled:
# this branch only implements replay for vLLM generation and rejects it with
# the Megatron backend during configuration validation.
USER_EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
export EXTRA_OVERRIDES="\
++policy.megatron_cfg.freeze_vision_model=true \
++policy.megatron_cfg.freeze_vision_projection=true \
++policy.megatron_cfg.freeze_moe_router=true \
++policy.megatron_cfg.moe_router_load_balancing_type=none \
++policy.megatron_cfg.moe_router_bias_update_rate=0.0 \
++policy.megatron_cfg.mtp_use_repeated_layer=false \
++policy.megatron_cfg.mtp_detach_heads=false \
++policy.router_replay.enabled=false \
++policy.hf_config_overrides.video_temporal_patch_size=${TEMPORAL_PATCH_SIZE} \
++policy.hf_config_overrides.video_target_num_patches=${VIDEO_TARGET_PATCHES} \
++policy.generation.mcore_generation_config.http_server_num_replicas=${HTTP_SERVER_NUM_REPLICAS} \
++policy.generation.mcore_generation_config.image_dynamic_resolution=true \
++policy.generation.mcore_generation_config.image_dynamic_resolution_rounding_mode=${IMAGE_DYNAMIC_RESOLUTION_ROUNDING_MODE} \
++policy.generation.mcore_generation_config.image_dynamic_resolution_resize_mode=${IMAGE_DYNAMIC_RESOLUTION_RESIZE_MODE} \
++policy.generation.mcore_generation_config.megatron_inference_wrapper=megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper.NemotronOmniInferenceWrapper \
++policy.generation.mcore_generation_config.parsers=[deepseek-r1-reasoning] \
++policy.generation.mcore_generation_config.video_maintain_aspect_ratio=false \
++policy.generation.ignore_eos=false \
++policy.generation.bad_words=\"['<image>','<img>','</img>','<so_embedding>','<so_start>','<so_end>']\" \
++data.default.video_sampling_style=nemotron_vl \
++data.default.video_maintain_aspect_ratio=false \
++grpo.deduplicate_multimodal_data=false \
++async_rl.rollout_failure.nemo_gym.rollout_timeout_s=${NEMO_GYM_ROLLOUT_TIMEOUT_S} \
++async_rl.generation_router.enabled=true \
++async_rl.generation_router.backend_timeout_s=${GENERATION_ROUTER_BACKEND_TIMEOUT_S} \
++async_rl.generation_router.connect_timeout_s=5 \
++async_rl.generation_fleet_health.enabled=false \
++async_rl.stall_watchdog.stall_timeout_s=${STALL_WATCHDOG_TIMEOUT_S} \
++async_rl.stall_watchdog.stall_action=abort \
++checkpointing.checkpoint_dir=${VIDEO_TEACHER_RESULTS_DIR}/checkpoints \
++checkpointing.save_data_plane=true \
++logger.log_dir=${VIDEO_TEACHER_RESULTS_DIR}/logs \
++logger.wandb.project=${VIDEO_TEACHER_WANDB_PROJECT} \
++logger.wandb.name=${VIDEO_TEACHER_WANDB_NAME}-\${NRL_SLURM_JOB_ID} \
++logger.wandb.id=${VIDEO_TEACHER_WANDB_ID}-\${NRL_SLURM_JOB_ID} \
${USER_EXTRA_OVERRIDES}"

exec bash "${SCRIPT_DIR}/submit_nemotron_omni_multimodal_single_controller_8n4g.sh" "$@"
