#!/usr/bin/env bash
set -euo pipefail

# NeMo-RL GRPO integration for draft Megatron-LM PR #7367.
# Ray assigns two GPUs to the colocated MIMO policy and two GPUs to the
# dedicated ordinary Hybrid+RADIO LLaVA DynamicInferenceEngine destination.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
CONFIG="${CONFIG:-${NEMO_RL_ROOT}/examples/configs/recipes/vlm/vlm_grpo-nemotron-mock-mimo-1n4g-megatron_generation.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

cd "${NEMO_RL_ROOT}"
exec uv run --no-sync python examples/run_vlm_grpo.py --config "${CONFIG}" "$@"
