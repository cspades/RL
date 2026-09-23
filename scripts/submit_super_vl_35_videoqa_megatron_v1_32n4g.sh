#!/usr/bin/env bash
set -euo pipefail

# NeMo-RL V1 twin of submit_super_vl_35_videoqa_megatron_v2_32n4g.sh.
# Model, data, topology, optimization, sampling, and media geometry are kept
# identical; only the V1 GRPO execution path replaces SingleController.

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

export CONFIG="examples/configs/recipes/vlm/super_vl_35_videoqa_recipe.yaml"
export ENTRYPOINT="examples/nemo_gym/run_grpo_nemo_gym.py"
export GENERATION_BACKEND=megatron
export MODEL_NAME="${VIDEO_TEACHER_MODEL_PATH}"
export DATA_ROOT="${CONTAINER_NEMORL}/${DATA_REL}"
export NEMO_RL_VIDEO_TRAIN_JSONL="${VIDEO_TEACHER_DATA_PATH}"
export NEMO_RL_VIDEO_VAL_JSONL="${VIDEO_TEACHER_DATA_PATH}"
export NEMO_RL_VIDEO_MEDIA_ROOT="${VIDEO_TEACHER_MEDIA_ROOT}"
export PREPARE_VSTAT=false

export NUM_NODES="${NUM_NODES:-32}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-16}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-8}"
export NUM_PROMPTS="${NUM_PROMPTS:-128}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
export TRAIN_GBS="${TRAIN_GBS:-2048}"
export MAX_STEPS="${MAX_STEPS:-1000000}"
export ASYNC_GRPO="${ASYNC_GRPO:-true}"
export MAX_TRAJECTORY_AGE_STEPS="${MAX_TRAJECTORY_AGE_STEPS:-1}"
export IN_FLIGHT_WEIGHT_UPDATES="${IN_FLIGHT_WEIGHT_UPDATES:-true}"
export VAL_PERIOD="${VAL_PERIOD:-0}"
export VAL_AT_START="${VAL_AT_START:-false}"
export VAL_AT_END="${VAL_AT_END:-false}"
export VAL_NUM_GENERATIONS="${VAL_NUM_GENERATIONS:-4}"

export POLICY_TP="${POLICY_TP:-2}"
export POLICY_EP="${POLICY_EP:-16}"
export POLICY_CP="${POLICY_CP:-1}"
export INFER_TP="${INFER_TP:-4}"
export INFER_EP="${INFER_EP:-4}"
export REFIT_BACKEND="${REFIT_BACKEND:-nccl}"

export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-65536}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
export INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-65536}"
export TRAIN_MB_TOKENS="${TRAIN_MB_TOKENS:-49152}"
export LOGPROB_MB_TOKENS="${LOGPROB_MB_TOKENS:-65536}"
export NUM_FRAMES="${NUM_FRAMES:-64}"
export TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE:-2}"
export VIDEO_TARGET_PATCHES="${VIDEO_TARGET_PATCHES:-1024}"
export MIN_GENERATION_TOKENS="${MIN_GENERATION_TOKENS:-16384}"
export IMAGE_DYNAMIC_RESOLUTION_ROUNDING_MODE="${IMAGE_DYNAMIC_RESOLUTION_ROUNDING_MODE:-round_plus_half}"
export IMAGE_DYNAMIC_RESOLUTION_RESIZE_MODE="${IMAGE_DYNAMIC_RESOLUTION_RESIZE_MODE:-torch_bicubic_antialias}"

export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.0}"
export OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-false}"
export OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-false}"
export USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-false}"
export EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-float32}"
export EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-float32}"
export STORE_PARAM_REMAINDERS="${STORE_PARAM_REMAINDERS:-false}"
export OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-false}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-false}"
export BUFFER_SIZE_GB="${BUFFER_SIZE_GB:-16}"
export PREFIX_CACHING_MAMBA_GB="${PREFIX_CACHING_MAMBA_GB:-32}"
export VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-17179869184}"

export CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-false}"
export RESULTS_DIR="${RESULTS_DIR:-${NEMORL}/workspace/results/super-vl-35-videoqa-megatron-v1}"
export VIDEO_TEACHER_RESULTS_DIR="${VIDEO_TEACHER_RESULTS_DIR:-${RESULTS_DIR}}"
export VIDEO_TEACHER_GYM_VENV_DIR="${VIDEO_TEACHER_GYM_VENV_DIR:-${CONTAINER_NEMORL}/workspace/gym_venvs/super-vl-35-videoqa}"
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-${VIDEO_TEACHER_GYM_VENV_DIR}}"
export VIDEO_TEACHER_WANDB_PROJECT="${VIDEO_TEACHER_WANDB_PROJECT:-mllm-rl-dev}"
export VIDEO_TEACHER_WANDB_NAME="${VIDEO_TEACHER_WANDB_NAME:-super-vl-35-videoqa-megatron-v1}"
export VIDEO_TEACHER_WANDB_ID="${VIDEO_TEACHER_WANDB_ID:-${VIDEO_TEACHER_WANDB_NAME}}"
export WANDB_PROJ="${WANDB_PROJ:-${VIDEO_TEACHER_WANDB_PROJECT}}"
export WANDB_NAME="${WANDB_NAME:-${VIDEO_TEACHER_WANDB_NAME}}"
export WANDB_GROUP="${WANDB_GROUP:-adlr}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export JOB_NAME="${JOB_NAME:-super-vl-35-videoqa-megatron-v1-32n4g}"
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
export CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/asolergibert/RL/images/nemo-rl-nightly-gym.sqsh}"

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
++policy.sequence_packing.train_mb_tokens=${TRAIN_MB_TOKENS} \
++policy.sequence_packing.logprob_mb_tokens=${LOGPROB_MB_TOKENS} \
++policy.megatron_cfg.optimizer.overlap_cpu_optimizer_d2h_h2d=${OVERLAP_CPU_OPTIMIZER_D2H_H2D} \
++policy.hf_config_overrides.video_temporal_patch_size=${TEMPORAL_PATCH_SIZE} \
++policy.hf_config_overrides.video_target_num_patches=${VIDEO_TARGET_PATCHES} \
++policy.generation.temperature=1.0 \
++policy.generation.top_p=1.0 \
++policy.generation.ignore_eos=false \
++policy.generation.bad_words=\"['<image>','<img>','</img>','<so_embedding>','<so_start>','<so_end>']\" \
++policy.generation.mcore_generation_config.http_server_num_replicas=4 \
++policy.generation.mcore_generation_config.image_dynamic_resolution=true \
++policy.generation.mcore_generation_config.image_dynamic_resolution_rounding_mode=${IMAGE_DYNAMIC_RESOLUTION_ROUNDING_MODE} \
++policy.generation.mcore_generation_config.image_dynamic_resolution_resize_mode=${IMAGE_DYNAMIC_RESOLUTION_RESIZE_MODE} \
++policy.generation.mcore_generation_config.megatron_inference_wrapper=megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper.NemotronOmniInferenceWrapper \
++policy.generation.mcore_generation_config.parsers=[deepseek-r1-reasoning] \
++policy.generation.mcore_generation_config.video_maintain_aspect_ratio=false \
++data.default.video_maintain_aspect_ratio=false \
++grpo.deduplicate_multimodal_data=true \
++checkpointing.checkpoint_dir=${VIDEO_TEACHER_RESULTS_DIR}/checkpoints \
++logger.log_dir=${VIDEO_TEACHER_RESULTS_DIR}/logs \
++logger.wandb.name=${VIDEO_TEACHER_WANDB_NAME}-\${NRL_SLURM_JOB_ID} \
++logger.wandb.project=${VIDEO_TEACHER_WANDB_PROJECT} \
++logger.wandb.id=${VIDEO_TEACHER_WANDB_ID}-\${NRL_SLURM_JOB_ID} \
${USER_EXTRA_OVERRIDES}"

exec bash "${SCRIPT_DIR}/submit_nemotron_omni_vstat_megatron_8n4g.sh" "$@"
