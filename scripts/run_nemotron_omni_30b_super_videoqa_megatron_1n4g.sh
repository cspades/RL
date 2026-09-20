#!/usr/bin/env bash
set -euo pipefail

# Run directly inside the NeMo-RL container on one 4-GPU node.
# Uses two GPUs for policy training and two for Megatron generation.
# This reproduces the Super VideoQA frontend/timeout path with the smaller
# Nemotron-3 Nano Omni 30B-A3B model; it is not a throughput comparison.

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
# unchanged.
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
export MAX_INFLIGHT_PROMPTS="${MAX_INFLIGHT_PROMPTS:-2}"
export MAX_BUFFERED_ROLLOUTS="${MAX_BUFFERED_ROLLOUTS:-4}"
export MAX_STEPS="${MAX_STEPS:-100000}"

# Full optimizer CPU offload, as the 67B vision 1n4g runner does: the 2-GPU
# train half cannot hold master weights and Adam moments alongside the vocab
# logits for a ~37k-token video sequence. Precision-aware bf16 moments stay on
# so the offloaded state costs 6 bytes/param of host RAM instead of 12.
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
export OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-true}"

export ASYNC_RL_DIAGNOSTICS="${ASYNC_RL_DIAGNOSTICS:-true}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export MONITOR_GPUS="${MONITOR_GPUS:-true}"
export MEGATRON_TRANSFORMER_IMPL="${MEGATRON_TRANSFORMER_IMPL:-inference_optimized}"
export MEGATRON_CUDA_GRAPH_IMPL="${MEGATRON_CUDA_GRAPH_IMPL:-local}"
export VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-8589934592}"
export PREFIX_CACHING_MAMBA_GB="${PREFIX_CACHING_MAMBA_GB:-8}"
export RESULTS_DIR="${RESULTS_DIR:-${NEMORL}/workspace/results/nemotron-omni-30b-super-videoqa-megatron-1n4g}"

NEMO_GYM_ROLLOUT_TIMEOUT_S="${NEMO_GYM_ROLLOUT_TIMEOUT_S:-720}"
GENERATION_ROUTER_BACKEND_TIMEOUT_S="${GENERATION_ROUTER_BACKEND_TIMEOUT_S:-600}"
STALL_WATCHDOG_TIMEOUT_S="${STALL_WATCHDOG_TIMEOUT_S:-1200}"
HTTP_SERVER_NUM_REPLICAS="${HTTP_SERVER_NUM_REPLICAS:-4}"
MAX_CONCURRENT_GYM_ROWS="${MAX_CONCURRENT_GYM_ROWS:-4}"
GENERATION_ROUTER_MAX_INFLIGHT="${GENERATION_ROUTER_MAX_INFLIGHT:-4}"
GENERATION_ROUTER_MAX_INFLIGHT_PER_BACKEND="${GENERATION_ROUTER_MAX_INFLIGHT_PER_BACKEND:-4}"
GENERATION_ROUTER_MAX_INFLIGHT_BYTES="${GENERATION_ROUTER_MAX_INFLIGHT_BYTES:-4294967296}"

echo "Running Omni 30B Super VideoQA reproduction inside the current container"
echo "  GPUs: 2 policy + 2 Megatron generation"
echo "  dataset=${DATA_JSONL}"
echo "  prompts/generations=${NUM_PROMPTS_PER_STEP}/${NUM_GENERATIONS_PER_PROMPT}"
echo "  sequence/new_tokens=${MAX_SEQUENCE_LENGTH}/${MAX_NEW_TOKENS}"
echo "  Gym/router/watchdog timeouts=${NEMO_GYM_ROLLOUT_TIMEOUT_S}/${GENERATION_ROUTER_BACKEND_TIMEOUT_S}/${STALL_WATCHDOG_TIMEOUT_S}s"

exec bash "${SCRIPT_DIR}/run_nemotron_omni_multimodal_single_controller_1n4g.sh" \
  ++policy.generation.bad_words="['<image>','<img>','</img>','<so_embedding>','<so_start>','<so_end>']" \
  ++policy.generation.mcore_generation_config.http_server_num_replicas="${HTTP_SERVER_NUM_REPLICAS}" \
  ++policy.generation.mcore_generation_config.image_dynamic_resolution=true \
  ++policy.generation.mcore_generation_config.video_maintain_aspect_ratio=false \
  ++data.shuffle=true \
  ++data.default.video_maintain_aspect_ratio=false \
  ++env.nemo_gym.config_paths="[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/mcqa/configs/mcqa.yaml,resources_servers/string_match/configs/string_match.yaml,resources_servers/sav_tracks/configs/sav_tracks.yaml]" \
  ++async_rl.rollout_failure.nemo_gym.rollout_timeout_s="${NEMO_GYM_ROLLOUT_TIMEOUT_S}" \
  ++async_rl.rollout_failure.nemo_gym.max_concurrent_rows="${MAX_CONCURRENT_GYM_ROWS}" \
  ++async_rl.sampler.name=ready_first \
  ++async_rl.sampler.max_staleness_versions=1 \
  ++async_rl.generation_router.enabled=true \
  ++async_rl.generation_router.backend_timeout_s="${GENERATION_ROUTER_BACKEND_TIMEOUT_S}" \
  ++async_rl.generation_router.connect_timeout_s=5 \
  ++async_rl.generation_router.diagnostics_interval_s=30 \
  ++async_rl.generation_router.admission_enabled=true \
  ++async_rl.generation_router.max_inflight_requests="${GENERATION_ROUTER_MAX_INFLIGHT}" \
  ++async_rl.generation_router.max_inflight_requests_per_backend="${GENERATION_ROUTER_MAX_INFLIGHT_PER_BACKEND}" \
  ++async_rl.generation_router.max_inflight_request_bytes="${GENERATION_ROUTER_MAX_INFLIGHT_BYTES}" \
  ++async_rl.generation_router.unknown_request_bytes=67108864 \
  ++async_rl.generation_router.request_body_timeout_s=120 \
  ++async_rl.generation_fleet_health.enabled=false \
  ++async_rl.stall_watchdog.stall_timeout_s="${STALL_WATCHDOG_TIMEOUT_S}" \
  ++async_rl.stall_watchdog.stall_action=abort \
  ++async_rl.stall_watchdog.gym_subprocess_check=false \
  ++env.nemo_gym.initial_global_config_dict.global_aiohttp_client_request_debug=true \
  "$@"
