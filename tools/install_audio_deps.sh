#!/bin/bash
# Install audio/video dependencies that are NOT shipped in the NeMo-RL container.
#
# Run this script before using audio/video features or running audio/VLM tests:
#
#   bash tools/install_audio_deps.sh
#
# Safe to call multiple times.
set -euo pipefail

AUDIO_DEPS_MAX_ATTEMPTS="${AUDIO_DEPS_MAX_ATTEMPTS:-3}"
AUDIO_DEPS_COMMAND_TIMEOUT_S="${AUDIO_DEPS_COMMAND_TIMEOUT_S:-300}"
AUDIO_DEPS_STAGGER_MAX_S="${AUDIO_DEPS_STAGGER_MAX_S:-0}"

retry_with_timeout() {
    local description="$1"
    shift
    local attempt status=1

    for ((attempt = 1; attempt <= AUDIO_DEPS_MAX_ATTEMPTS; attempt++)); do
        echo "[audio-deps] ${description} (attempt ${attempt}/${AUDIO_DEPS_MAX_ATTEMPTS})..."
        if timeout --signal=TERM --kill-after=30s \
            "${AUDIO_DEPS_COMMAND_TIMEOUT_S}s" "$@"; then
            return 0
        fi
        status=$?
        if (( attempt < AUDIO_DEPS_MAX_ATTEMPTS )); then
            sleep $((attempt * 5 + RANDOM % 6))
        fi
    done

    echo "[audio-deps] ERROR: ${description} failed after ${AUDIO_DEPS_MAX_ATTEMPTS} attempts." >&2
    return "$status"
}

if (( AUDIO_DEPS_STAGGER_MAX_S > 0 )); then
    # Avoid having every node hit the package mirrors simultaneously.
    sleep $((RANDOM % (AUDIO_DEPS_STAGGER_MAX_S + 1)))
fi

if ! python -c "import torchcodec" 2>/dev/null; then
    # Install system FFmpeg — torchcodec dlopens libavcodec.so.* at runtime.
    echo "[audio-deps] Installing system FFmpeg..."
    apt_options=(
        -o Acquire::Retries=5
        -o Acquire::http::Timeout=30
        -o Acquire::https::Timeout=30
        -o DPkg::Lock::Timeout=120
    )
    retry_with_timeout "Updating APT package lists" \
        env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" update
    retry_with_timeout "Installing system FFmpeg" \
        env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" \
        install -y --no-install-recommends ffmpeg

    # torchaudio 2.11+ routes torchaudio.load through torchcodec, so both are needed.
    # --no-config prevents the project's [tool.uv] overrides from interfering.
    echo "[audio-deps] Installing torchaudio==2.11.0 and torchcodec..."
    retry_with_timeout "Installing torchaudio and torchcodec" \
        env UV_HTTP_RETRIES=5 UV_HTTP_TIMEOUT=60 \
        uv pip install --no-config \
            --index-url https://download.pytorch.org/whl/cu130 \
            --extra-index-url https://pypi.org/simple \
            --reinstall-package torchaudio \
            "torchaudio==2.11.0" \
            "torchcodec==0.11.1"
fi

# PyAV is intentionally absent from the base image (pyproject excludes it via
# `av; sys_platform == 'never'` because it bundles CVE-carrying codec libs), so it
# must be installed after the fact into the isolated Megatron policy worker
# environment that imports it. `--no-config` bypasses that exclusion; the version
# floor is therefore restated here to keep pyproject's CVE-2026-40962 constraint.
#
# The worker venv is created lazily at worker start, so run this AFTER the
# Megatron worker has been created at least once (or point RAY_MEGATRON_PYTHON at
# an existing venv).
RAY_MEGATRON_PYTHON="${RAY_MEGATRON_PYTHON:-/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}"
if [[ ! -x "$RAY_MEGATRON_PYTHON" ]]; then
    echo "[audio-deps] ERROR: Megatron worker environment not found: $RAY_MEGATRON_PYTHON" >&2
    echo "[audio-deps] It is created on first worker start. Run this script after that," >&2
    echo "[audio-deps] or set RAY_MEGATRON_PYTHON to an existing worker interpreter." >&2
    exit 1
fi
if ! "$RAY_MEGATRON_PYTHON" -c "import av" 2>/dev/null; then
    # `uv pip install --python` targets the venv directly; these venvs are built
    # by `uv venv` without `--seed`, so they have no pip to invoke.
    echo "[audio-deps] Installing PyAV in the Megatron worker environment..."
    retry_with_timeout "Installing PyAV" \
        env UV_HTTP_RETRIES=5 UV_HTTP_TIMEOUT=60 \
        uv pip install --no-config --python "$RAY_MEGATRON_PYTHON" "av>=17.1.0"
fi

echo "[audio-deps] Done."
