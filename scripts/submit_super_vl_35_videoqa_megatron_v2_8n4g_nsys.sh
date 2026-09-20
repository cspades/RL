#!/usr/bin/env bash
set -euo pipefail

# Reduced-scale Super 3.5 video-QA run for reaching the optimizer quickly and
# capturing one training step with Nsight Systems. The model topology is the
# same as the 32-node recipe; only fleet size, rollout volume, and generation
# length are reduced. Every value remains overrideable from the environment.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"

# Four training nodes (16 GPUs) satisfy TP=2, EP=16. Four generation nodes
# provide four TP=4 Megatron inference replicas.
export NUM_NODES="${NUM_NODES:-8}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-4}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-4}"

# Four prompt groups x four generations produce a 16-row training batch.
# Training DP is 8, so TRAIN_GBS=16 is divisible by the training DP size.
export NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-4}"
export NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-4}"
export TRAIN_GBS="${TRAIN_GBS:-16}"
export MAX_INFLIGHT_PROMPTS="${MAX_INFLIGHT_PROMPTS:-4}"
export MAX_CONCURRENT_GYM_ROWS="${MAX_CONCURRENT_GYM_ROWS:-16}"
export MAX_BUFFERED_ROLLOUTS="${MAX_BUFFERED_ROLLOUTS:-8}"
export HTTP_SERVER_NUM_REPLICAS="${HTTP_SERVER_NUM_REPLICAS:-8}"

# Bound stragglers for this profiling run. This is intentionally not a
# benchmark-equivalent 16K rollout recipe.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export MIN_GENERATION_TOKENS="${MIN_GENERATION_TOKENS:-4096}"

# Capture the first optimizer iteration and stop profiling before the second.
export ENABLE_NSYS="${ENABLE_NSYS:-true}"
export NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-1:2}"
export NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-*policy*,*megatron*}"
export MAX_STEPS="${MAX_STEPS:-2}"

export NEMO_GYM_ROLLOUT_TIMEOUT_S="${NEMO_GYM_ROLLOUT_TIMEOUT_S:-1800}"
export GENERATION_ROUTER_BACKEND_TIMEOUT_S="${GENERATION_ROUTER_BACKEND_TIMEOUT_S:-1500}"
export STALL_WATCHDOG_TIMEOUT_S="${STALL_WATCHDOG_TIMEOUT_S:-2400}"

export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export MONITOR_GPUS="${MONITOR_GPUS:-false}"
export CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-false}"
export RESULTS_DIR="${RESULTS_DIR:-${NEMORL}/workspace/results/super-vl-35-videoqa-megatron-v2-8n4g-nsys}"
export VIDEO_TEACHER_RESULTS_DIR="${VIDEO_TEACHER_RESULTS_DIR:-${RESULTS_DIR}}"
export VIDEO_TEACHER_WANDB_NAME="${VIDEO_TEACHER_WANDB_NAME:-super-vl-35-videoqa-megatron-v2-8n4g-nsys}"
export VIDEO_TEACHER_WANDB_ID="${VIDEO_TEACHER_WANDB_ID:-${VIDEO_TEACHER_WANDB_NAME}}"
export JOB_NAME="${JOB_NAME:-super-vl-35-videoqa-megatron-v2-8n4g-nsys}"
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"

exec bash "${SCRIPT_DIR}/submit_super_vl_35_videoqa_megatron_v2_32n4g.sh" "$@"
