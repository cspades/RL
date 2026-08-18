# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM video loader registration."""

import os
from typing import Any

import numpy as np

from nemo_rl.data import video_utils as _shared

_CACHED_VIDEO_FRAME_MANIFEST_MAGIC = _shared._CACHED_VIDEO_FRAME_MANIFEST_MAGIC
_TORCHCODEC_END_OF_STREAM_ERROR = _shared._TORCHCODEC_END_OF_STREAM_ERROR
_compute_video_timestamps = _shared._compute_video_timestamps
_find_torchcodec_decodable_frame_count = (
    _shared._find_torchcodec_decodable_frame_count
)
_get_positive_int_env = _shared.get_positive_int_env
_get_video_sampling_style = _shared._get_video_sampling_style
_is_torchcodec_end_of_stream_error = _shared._is_torchcodec_end_of_stream_error
_load_cached_video_frame_manifest = _shared.load_cached_video_frame_manifest
_load_video_frames_decord_with_metadata = (
    _shared._load_video_frames_decord_with_metadata
)
_load_video_frames_pyav_with_metadata = _shared._load_video_frames_pyav_with_metadata
_load_video_frames_torchcodec_with_metadata = (
    _shared._load_video_frames_torchcodec_with_metadata
)
_load_video_frames_vllm_with_metadata = (
    _shared._load_video_frames_vllm_with_metadata
)
_round_video_frame_count = _shared._round_video_frame_count
_select_video_frame_count = _shared._select_video_frame_count
_timestamp_to_video_frame_index = _shared._timestamp_to_video_frame_index
_torchcodec_sample_indices = _shared._torchcodec_sample_indices

build_cached_video_frame_data_url = _shared.build_cached_video_frame_data_url
load_video_frames = _shared.load_video_frames


def load_video_frames_with_metadata(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility dispatcher backed by shared decoder functions."""
    backend = os.environ.get("NRL_VIDEO_BACKEND", "torchcodec").strip().lower()
    if backend == "torchcodec":
        return _load_video_frames_torchcodec_with_metadata(
            video_path, num_frames, temporal_patch_size
        )
    if backend == "vllm":
        return _load_video_frames_vllm_with_metadata(
            video_path, num_frames, temporal_patch_size
        )
    if backend == "decord":
        return _load_video_frames_decord_with_metadata(
            video_path, num_frames, temporal_patch_size
        )
    if backend != "pyav":
        raise ValueError(
            f"Unsupported NRL_VIDEO_BACKEND={backend!r}; "
            "supported: 'torchcodec', 'pyav', 'decord', 'vllm'"
        )
    return _load_video_frames_pyav_with_metadata(
        video_path, num_frames, temporal_patch_size
    )


def register_torchcodec_vllm_video_loader() -> bool:
    """Register shared TorchCodec decoding with vLLM's Nemotron video loader."""
    video_backend = os.environ.get("NRL_VIDEO_BACKEND", "torchcodec").strip().lower()
    vllm_loader = os.environ.get("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    if video_backend != "torchcodec" or vllm_loader != "nemotron_vl":
        return False

    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

    class TorchCodecNemotronVLVideoBackend:
        @classmethod
        def load_bytes(
            cls,
            data: bytes,
            num_frames: int = -1,
            fps: int = -1,
            max_duration: int = 300,
            frame_recovery: bool = False,
            **kwargs: Any,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            del cls, max_duration, kwargs
            if frame_recovery:
                raise ValueError(
                    "frame_recovery is not supported by the TorchCodec video loader"
                )

            cached_video = _shared.load_cached_video_frame_manifest(
                data, num_frames=int(num_frames)
            )
            if cached_video is not None:
                return cached_video

            try:
                from torchcodec.decoders import VideoDecoder
            except ImportError as exc:
                raise ImportError(
                    "Gym video generation requires the optional video dependencies. "
                    "Run `bash tools/install_audio_deps.sh` before training."
                ) from exc

            decoder = VideoDecoder(
                data,
                dimension_order="NHWC",
                num_ffmpeg_threads=0,
                device="cpu",
                seek_mode="exact",
            )
            total_frames = int(decoder.metadata.num_frames or 0)
            source_fps = float(decoder.metadata.average_fps or 0.0)
            if total_frames <= 0:
                raise ValueError("Video has no frames")
            if source_fps <= 0:
                raise ValueError(f"Video has invalid fps ({source_fps})")

            requested_num_frames = (
                total_frames if int(num_frames) < 0 else int(num_frames)
            )
            if fps > 0:
                duration_limited_frames = max(
                    1, int((total_frames / source_fps) * float(fps))
                )
                requested_num_frames = min(
                    requested_num_frames, duration_limited_frames
                )

            frames, source_fps, total_frames, sampled_indices = (
                _shared.decode_torchcodec_video(
                    data,
                    requested_num_frames=requested_num_frames,
                    temporal_patch_size=_shared.get_positive_int_env(
                        _shared.VIDEO_TEMPORAL_PATCH_SIZE_ENV, 1
                    ),
                    source_description="in-memory video",
                    initial_decoder=decoder,
                )
            )
            metadata = _shared.build_video_metadata(
                fps=source_fps,
                total_frames=total_frames,
                sampled_indices=sampled_indices,
                backend="torchcodec_nemotron_vl",
            )
            metadata["original_video_bytes"] = data
            return frames, metadata

    VIDEO_LOADER_REGISTRY.register("nemotron_vl")(TorchCodecNemotronVLVideoBackend)
    return True
