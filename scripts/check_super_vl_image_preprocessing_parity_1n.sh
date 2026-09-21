#!/usr/bin/env bash
set -euo pipefail

# Run inside the NeMo-RL container on one node. This is CPU-only and does not
# launch Ray, Megatron workers, Gym, or a training job.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"

MODEL_PATH="${MODEL_PATH:-${NEMORL}/workspace/models/super-vl-35-video-teacher-step-120/hf}"
DATA_ROOT="${DATA_ROOT:-${NEMORL}/workspace/datasets/super-vl-35-videoqa}"
DATA_FILENAME="${DATA_FILENAME:-train_sav_all_tracks_plus_caprl_exclude6215_hsg_mediafixed_9.jsonl}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/${DATA_FILENAME}}"

MODEL_LENGTH="${MODEL_LENGTH:-}"
ROUNDING_MODE="${ROUNDING_MODE:-round_plus_half}"
RESIZE_MODE="${RESIZE_MODE:-torch_bicubic_antialias}"
PIXEL_ATOL="${PIXEL_ATOL:-0}"
START_ROW="${START_ROW:-0}"
MAX_ROWS="${MAX_ROWS:-}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
REPORT_LIMIT="${REPORT_LIMIT:-20}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing HF checkpoint: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -s "${DATASET_PATH}" ]]; then
  echo "Missing dataset: ${DATASET_PATH}" >&2
  exit 2
fi

ARGS=(
  --model "${MODEL_PATH}"
  --dataset "${DATASET_PATH}"
  --rounding-mode "${ROUNDING_MODE}"
  --resize-mode "${RESIZE_MODE}"
  --pixel-atol "${PIXEL_ATOL}"
  --start-row "${START_ROW}"
  --progress-every "${PROGRESS_EVERY}"
  --report-limit "${REPORT_LIMIT}"
)
if [[ -n "${MODEL_LENGTH}" ]]; then
  ARGS+=(--model-length "${MODEL_LENGTH}")
fi
if [[ -n "${MAX_ROWS}" ]]; then
  ARGS+=(--max-rows "${MAX_ROWS}")
fi

cd "${NEMORL}"
exec uv run --no-sync python \
  "${SCRIPT_DIR}/check_super_vl_image_preprocessing_parity.py" \
  "${ARGS[@]}" \
  "$@"
