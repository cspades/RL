#!/usr/bin/env bash
# Bring nemo-lens up to the revision the mounted checkout needs, in the driver
# venv and in every cached Ray worker venv. Older nightly containers ship
# older nemo-lens revisions, whose nemo.lens.groups has no SpanRegistry.
#
# This is the NEMO_LENS_RUNTIME_SETUP block from
# submit_nemotron_omni_multimodal_single_controller_8n4g.sh, runnable by hand.
# Keep the default revision in sync with that script and with the nemo-lens
# source in pyproject.toml.
#
# Run it inside the container: it needs uv and the container interpreters.
set -euo pipefail

NEMO_LENS_RUNTIME_REV="${NEMO_LENS_RUNTIME_REV:-b0f977d414b2f89938604a0b7eaa78ee08bc8700}"
NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
DRIVER_PYTHON="${DRIVER_PYTHON:-/opt/nemo_rl_venv/bin/python}"
LENS_SPEC="nemo-lens[sdk,aiohttp] @ git+https://github.com/NVIDIA-NeMo/Lens.git@${NEMO_LENS_RUNTIME_REV}"
PROBE='from nemo.lens.groups import SpanRegistry; from nemo.lens.instruments import MetricSpec, register_metric_group'

usage() {
  cat <<'EOF'
Usage: update_nemo_lens.sh [-n] [PYTHON...]

With no PYTHON arguments, updates the driver venv and every cached worker venv
under NEMO_RL_VENV_DIR. Given explicit interpreter paths, updates only those.

  -n, --dry-run   Report what each interpreter needs, install nothing.
  -h, --help      This message.

Environment:
  NEMO_LENS_RUNTIME_REV   Revision to install (default: the pinned SHA).
  NEMO_RL_VENV_DIR        Where cached worker venvs live (default /opt/ray_venvs).
  DRIVER_PYTHON           Driver interpreter (default /opt/nemo_rl_venv/bin/python).
EOF
}

DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -n|--dry-run) DRY_RUN=true; shift ;;
    --) shift; break ;;
    -*) echo "update_nemo_lens: unknown option $1" >&2; usage >&2; exit 2 ;;
    *) break ;;
  esac
done

if [[ "${DRY_RUN}" == false ]] && ! command -v uv >/dev/null 2>&1; then
  echo "update_nemo_lens: uv is not on PATH; run this inside the container." >&2
  exit 1
fi

# Counted rather than fatal: one unwritable venv should not stop the rest, but
# the exit status still has to report that the job was not finished.
failed=0

ensure_nemo_lens_runtime() {
  local python=$1
  if [[ ! -x "${python}" ]]; then
    echo "[nemo-lens] skip ${python} (no such interpreter)"
    return
  fi
  if "${python}" -c "${PROBE}" >/dev/null 2>&1; then
    echo "[nemo-lens] ok ${python}"
    return
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[nemo-lens] WOULD UPDATE ${python} to ${NEMO_LENS_RUNTIME_REV}"
    return
  fi
  echo "[nemo-lens] updating ${python} to ${NEMO_LENS_RUNTIME_REV}"
  if ! uv pip install --python "${python}" "${LENS_SPEC}"; then
    echo "[nemo-lens] FAILED to install into ${python}" >&2
    failed=$((failed + 1))
    return
  fi
  # Installing is not the goal; importing is. A resolved-but-broken install has
  # to fail here rather than at Ray actor startup an hour later.
  if ! "${python}" -c "${PROBE}"; then
    echo "[nemo-lens] FAILED: required Lens APIs still missing in ${python}" >&2
    failed=$((failed + 1))
  fi
}

pythons=()
if [[ $# -gt 0 ]]; then
  pythons=("$@")
else
  pythons=("${DRIVER_PYTHON}")
  # A fresh container has no cached worker venvs yet; without nullglob the
  # unmatched pattern would be handed on as a literal path.
  shopt -s nullglob
  workers=("${NEMO_RL_VENV_DIR}"/*/bin/python)
  shopt -u nullglob
  if (( ${#workers[@]} == 0 )); then
    echo "[nemo-lens] no worker venvs under ${NEMO_RL_VENV_DIR}"
  fi
  pythons+=("${workers[@]}")
fi

for python in "${pythons[@]}"; do
  ensure_nemo_lens_runtime "${python}"
done

if (( failed > 0 )); then
  echo "update_nemo_lens: ${failed} interpreter(s) failed" >&2
  exit 1
fi
