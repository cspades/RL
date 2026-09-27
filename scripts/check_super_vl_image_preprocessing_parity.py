#!/usr/bin/env python3
"""Compare NeMo-RL and Megatron image/video-frame preprocessing."""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
MCORE_ROOT = (
    REPO_ROOT
    / "3rdparty"
    / "Megatron-Bridge-workspace"
    / "Megatron-Bridge"
    / "3rdparty"
    / "Megatron-LM"
)
if str(MCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(MCORE_ROOT))

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.image_preprocessing import (  # noqa: E402
    preprocess_image_bytes_list,
    preprocess_video_bytes_list,
)
from nemo_rl.data.multimodal_utils import (  # noqa: E402
    CACHED_VIDEO_FRAME_MANIFEST_MAGIC,
    PackedTensor,
    attach_image_model_inputs_to_message,
    extract_multimodal_model_inputs,
    resolve_to_image,
)
from nemo_rl.environments.nemo_gym_multimodal import (  # noqa: E402
    _extract_static_video_messages,
)
from nemo_rl.environments.nemo_gym_request import (  # noqa: E402
    _chat_template_kwargs_for_processor,
)
from nemo_rl.environments.nemotron_utils import (  # noqa: E402
    process_nemotron_video_frames,
)
from nemo_rl.models.generation.vllm.video_utils import (  # noqa: E402
    build_cached_video_frame_metadata,
)
from nemo_rl.models.generation.megatron.utils import (  # noqa: E402
    build_image_preprocessing_config,
    build_video_preprocessing_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan still-image and cached-video-frame rows and require exact parity "
            "between NeMo-RL training and Megatron generation preprocessing."
        )
    )
    parser.add_argument("--model", required=True, help="HF checkpoint directory")
    parser.add_argument("--dataset", required=True, help="Input JSONL dataset")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum JSONL rows to inspect after --start-row (default: all)",
    )
    parser.add_argument(
        "--model-length",
        type=int,
        default=None,
        help="Override preprocessor_config.max_model_len",
    )
    parser.add_argument(
        "--rounding-mode",
        choices=("ceil", "round_plus_half"),
        default="round_plus_half",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("pil", "torch_bicubic_antialias"),
        default="torch_bicubic_antialias",
    )
    parser.add_argument(
        "--pixel-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for packed pixel tensors",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--report-limit", type=int, default=20)
    parser.add_argument(
        "--geometry-cache",
        action="store_true",
        help=(
            "Check only the first row for each ordered image-size layout. "
            "Disabled by default so the parity proof covers every image."
        ),
    )
    parser.add_argument(
        "--pad-dynamic-image-shapes",
        action="store_true",
        help=(
            "Exercise NeMo-RL's ragged multi-image materialization path. Set this "
            "iff env.nemo_gym.pad_dynamic_image_shapes is enabled in the target run."
        ),
    )
    parser.add_argument("--video-num-frames", type=int, default=64)
    parser.add_argument("--video-temporal-patch-size", type=int, default=2)
    parser.add_argument("--video-target-num-patches", type=int, default=1024)
    parser.add_argument("--video-maintain-aspect-ratio", action="store_true")
    return parser.parse_args()


def _content_part_source(part: dict[str, Any]) -> str | None:
    for key in ("image_url", "image", "url"):
        value = part.get(key)
        if isinstance(value, dict):
            value = value.get("url") or value.get("path")
        if isinstance(value, str) and value:
            return value
    return None


def _image_sources(row: dict[str, Any]) -> tuple[list[str], bool]:
    """Return image sources and whether they are cached video frames."""
    sources: list[str] = []
    has_still_image = False
    has_video_frame = False
    for item in row.get("responses_create_params", {}).get("input", []):
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in ("input_video", "video", "video_url"):
                return [], True
            if part_type not in ("input_image", "image", "image_url"):
                continue
            if part.get("_is_video_frame"):
                has_video_frame = True
            else:
                has_still_image = True
            source = _content_part_source(part)
            if source is None:
                raise ValueError(f"{part_type} content part has no image source.")
            sources.append(source)
    if has_still_image and has_video_frame:
        raise ValueError("A row cannot mix still images and cached video frames.")
    return sources, has_video_frame


def _load_rgb_image(source: str) -> Image.Image:
    with closing(resolve_to_image(source)) as image:
        return image.convert("RGB").copy()


def _image_size(source: str) -> tuple[int, int]:
    with closing(resolve_to_image(source)) as image:
        return image.size


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _materialize_training_field(message: dict[str, Any], key: str) -> torch.Tensor:
    value = message.get(key)
    if not isinstance(value, PackedTensor):
        raise TypeError(
            f"NeMo-RL attachment did not produce PackedTensor field {key!r}; "
            f"got {type(value).__name__}."
        )
    return value.as_tensor()


def _first_nonfinite(tensor: torch.Tensor) -> int | None:
    locations = (~torch.isfinite(tensor)).nonzero()
    return None if locations.numel() == 0 else int(locations[0].flatten()[0])


def _compare_row(
    *,
    row_index: int,
    sources: list[str],
    processor: Any,
    megatron_config: Any,
    pixel_atol: float,
    pad_dynamic_image_shapes: bool,
) -> str | None:
    images = [_load_rgb_image(source) for source in sources]
    try:
        # --- NeMo-RL training side ---
        # This is the production RL path used to attach policy-training media
        # after a Gym rollout. PackedTensor.as_tensor() performs the deferred
        # patchification used immediately before the training-model forward.
        training_message: dict[str, Any] = {
            "role": "user",
            "content": "",
            "token_ids": torch.empty(0, dtype=torch.long),
        }
        attach_image_model_inputs_to_message(
            training_message,
            images=images,
            processor=processor,
            pad_dynamic_image_shapes=pad_dynamic_image_shapes,
        )
        rl_patches = _materialize_training_field(training_message, "pixel_values")
        rl_sizes = _materialize_training_field(training_message, "imgs_sizes")
        rl_num_frames = _materialize_training_field(training_message, "num_frames")

        # --- MCore generation side ---
        # This is the raw-image preprocessing called by Megatron's inference
        # request path before the generation-model vision encoder.
        megatron_output = preprocess_image_bytes_list(
            [_encode_png(image) for image in images],
            megatron_config,
        )
        megatron_sizes_tensor = megatron_output["imgs_sizes"]
        megatron_patches = megatron_output["imgs"].squeeze(0)

        # --- RL versus MCore parity assertions ---
        if rl_sizes.ndim != 2 or rl_sizes.shape[1] != 2:
            return f"row={row_index} invalid RL imgs_sizes shape={tuple(rl_sizes.shape)}"
        if rl_num_frames.reshape(-1).tolist() != [1] * len(images):
            return (
                f"row={row_index} RL ordinary-image num_frames mismatch "
                f"value={rl_num_frames.reshape(-1).tolist()} expected={[1] * len(images)}"
            )
        if rl_sizes.dtype != torch.int32:
            return (
                f"row={row_index} RL imgs_sizes dtype mismatch "
                f"value={rl_sizes.dtype} expected=torch.int32"
            )
        if megatron_sizes_tensor.dtype != torch.int32:
            return (
                f"row={row_index} Megatron imgs_sizes dtype mismatch "
                f"value={megatron_sizes_tensor.dtype} expected=torch.int32"
            )
        rl_sizes_list = rl_sizes.tolist()
        megatron_sizes = megatron_sizes_tensor.tolist()
        merge_area = megatron_config.spatial_merge_size**2
        rl_tokens = [
            (height // megatron_config.patch_dim)
            * (width // megatron_config.patch_dim)
            // merge_area
            for height, width in rl_sizes_list
        ]
        megatron_tokens = [
            (height // megatron_config.patch_dim)
            * (width // megatron_config.patch_dim)
            // merge_area
            for height, width in megatron_sizes
        ]

        if len(rl_sizes_list) != len(images):
            return (
                f"row={row_index} RL image-count mismatch sizes={len(rl_sizes_list)} "
                f"sources={len(images)}"
            )
        if rl_sizes_list != megatron_sizes:
            return (
                f"row={row_index} size mismatch rl={rl_sizes_list} "
                f"megatron={megatron_sizes} sources={sources}"
            )
        if rl_tokens != megatron_tokens:
            return (
                f"row={row_index} token mismatch rl={rl_tokens} "
                f"megatron={megatron_tokens} sources={sources}"
            )
        if rl_patches.shape != megatron_output["imgs"].shape:
            return (
                f"row={row_index} patch-shape mismatch "
                f"rl={tuple(rl_patches.shape)} "
                f"megatron={tuple(megatron_output['imgs'].shape)} sources={sources}"
            )
        rl_patches = rl_patches.squeeze(0)
        if rl_patches.dtype != megatron_patches.dtype:
            return (
                f"row={row_index} patch-dtype mismatch rl={rl_patches.dtype} "
                f"megatron={megatron_patches.dtype}"
            )
        rl_nonfinite = _first_nonfinite(rl_patches)
        megatron_nonfinite = _first_nonfinite(megatron_patches)
        if rl_nonfinite is not None or megatron_nonfinite is not None:
            return (
                f"row={row_index} non-finite packed pixels "
                f"rl_first={rl_nonfinite} megatron_first={megatron_nonfinite}"
            )
        if not torch.allclose(
            rl_patches,
            megatron_patches,
            rtol=0.0,
            atol=pixel_atol,
            equal_nan=False,
        ):
            max_error = (rl_patches - megatron_patches).abs().max().item()
            patch_counts = [
                (height // megatron_config.patch_dim)
                * (width // megatron_config.patch_dim)
                for height, width in rl_sizes_list
            ]
            first_bad_image = None
            offset = 0
            for image_index, patch_count in enumerate(patch_counts):
                if not torch.allclose(
                    rl_patches[offset : offset + patch_count],
                    megatron_patches[offset : offset + patch_count],
                    rtol=0.0,
                    atol=pixel_atol,
                ):
                    first_bad_image = image_index
                    break
                offset += patch_count
            return (
                f"row={row_index} pixel mismatch max_abs_error={max_error:.9g} "
                f"first_bad_image={first_bad_image} sizes={rl_sizes_list} "
                f"sources={sources}"
            )
        return None
    finally:
        for image in images:
            image.close()


def _close_message_images(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("image"), Image.Image):
                part["image"].close()


def _compare_video_row(
    *,
    row_index: int,
    row: dict[str, Any],
    sources: list[str],
    processor: Any,
    image_config: Any,
    video_num_frames: int,
    temporal_patch_size: int,
    target_num_patches: int,
    maintain_aspect_ratio: bool,
    pixel_atol: float,
) -> str | None:
    if len(sources) != video_num_frames:
        return (
            f"row={row_index} cached-video frame-count mismatch "
            f"sources={len(sources)} configured={video_num_frames}"
        )

    extracted = _extract_static_video_messages(row)
    if extracted is None:
        return f"row={row_index} cached-video row was not recognized by NeMo-RL"
    hf_messages, video_path = extracted
    if video_path is not None:
        _close_message_images(hf_messages)
        return f"row={row_index} expected cached frames, got native video={video_path}"

    try:
        # --- NeMo-RL training side ---
        # Exact cached-video training preprocessing: video-specific resize and
        # normalization followed by PackedTensor patchification at the
        # training-model boundary.
        processed = process_nemotron_video_frames(
            processor,
            hf_messages,
            template_kwargs=_chat_template_kwargs_for_processor(row),
            temporal_patch_size=temporal_patch_size,
            target_num_patches=target_num_patches,
            maintain_aspect_ratio=maintain_aspect_ratio,
            prompt_expansion_mode="temporal_patch",
        )
        processed["num_frames"] = torch.tensor([len(sources)], dtype=torch.int32)
        model_inputs = extract_multimodal_model_inputs(processor, processed)
        training_message = {
            key: value
            for key, value in model_inputs.items()
            if isinstance(value, PackedTensor)
        }
        rl_patches = _materialize_training_field(training_message, "pixel_values")
        rl_sizes = _materialize_training_field(training_message, "imgs_sizes")
        rl_num_frames = _materialize_training_field(training_message, "num_frames")

        # --- MCore generation side ---
        # Reproduce the lossless cached-frame manifest sent to Megatron, then
        # run the exact inference video preprocessor that prepares inputs for
        # the generation-model vision encoder.
        metadata = build_cached_video_frame_metadata(len(sources))
        manifest = {
            "frame_paths": sources,
            "metadata": metadata,
        }
        payload = CACHED_VIDEO_FRAME_MANIFEST_MAGIC + json.dumps(
            manifest, separators=(",", ":")
        ).encode()
        video_config = build_video_preprocessing_config(
            image_config,
            {
                "video_num_frames": video_num_frames,
                "video_temporal_patch_size": temporal_patch_size,
                "video_target_num_patches": target_num_patches,
                "video_maintain_aspect_ratio": maintain_aspect_ratio,
            },
            frame_manifest_magic=CACHED_VIDEO_FRAME_MANIFEST_MAGIC,
        )
        if video_config is None:
            return f"row={row_index} failed to build Megatron video config"
        megatron_output = preprocess_video_bytes_list([payload], video_config)
        megatron_patches = megatron_output["imgs"]
        megatron_sizes = megatron_output["imgs_sizes"]
        megatron_num_frames = megatron_output["num_frames"]

        # --- RL versus MCore parity assertions ---
        if not torch.equal(rl_sizes, megatron_sizes):
            return (
                f"row={row_index} video size mismatch "
                f"rl={rl_sizes.tolist()} megatron={megatron_sizes.tolist()}"
            )
        if not torch.equal(rl_num_frames, megatron_num_frames):
            return (
                f"row={row_index} video num_frames mismatch "
                f"rl={rl_num_frames.tolist()} megatron={megatron_num_frames.tolist()}"
            )
        if rl_patches.shape != megatron_patches.shape:
            return (
                f"row={row_index} video patch-shape mismatch "
                f"rl={tuple(rl_patches.shape)} megatron={tuple(megatron_patches.shape)}"
            )
        if rl_patches.dtype != megatron_patches.dtype:
            return (
                f"row={row_index} video patch-dtype mismatch "
                f"rl={rl_patches.dtype} megatron={megatron_patches.dtype}"
            )
        rl_nonfinite = _first_nonfinite(rl_patches)
        megatron_nonfinite = _first_nonfinite(megatron_patches)
        if rl_nonfinite is not None or megatron_nonfinite is not None:
            return (
                f"row={row_index} non-finite packed video pixels "
                f"rl_first={rl_nonfinite} megatron_first={megatron_nonfinite}"
            )
        if not torch.allclose(
            rl_patches,
            megatron_patches,
            rtol=0.0,
            atol=pixel_atol,
            equal_nan=False,
        ):
            max_error = (rl_patches - megatron_patches).abs().max().item()
            return (
                f"row={row_index} video pixel mismatch "
                f"max_abs_error={max_error:.9g} sources={sources}"
            )
        return None
    finally:
        _close_message_images(hf_messages)


def main() -> int:
    args = _parse_args()
    if args.start_row < 0:
        raise ValueError("--start-row must be non-negative.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative.")
    if args.report_limit < 0:
        raise ValueError("--report-limit must be non-negative.")
    if args.pixel_atol < 0:
        raise ValueError("--pixel-atol must be non-negative.")
    if args.video_num_frames <= 0:
        raise ValueError("--video-num-frames must be positive.")
    if args.video_temporal_patch_size <= 0:
        raise ValueError("--video-temporal-patch-size must be positive.")
    if args.video_target_num_patches <= 0:
        raise ValueError("--video-target-num-patches must be positive.")

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    image_processor = processor.image_processor
    megatron_config = build_image_preprocessing_config(
        image_processor,
        dynamic_resolution=True,
        dynamic_resolution_model_length=args.model_length,
        dynamic_resolution_rounding_mode=args.rounding_mode,
        dynamic_resolution_resize_mode=args.resize_mode,
    )

    rows_read = 0
    rows_checked = 0
    image_rows_checked = 0
    video_rows_checked = 0
    images_checked = 0
    geometries_checked = 0
    checked_geometries: set[tuple[tuple[int, int], ...]] = set()
    mismatches: list[str] = []
    dataset_path = Path(args.dataset)
    with dataset_path.open(encoding="utf-8") as dataset:
        for row_index, line in enumerate(dataset):
            if row_index < args.start_row:
                continue
            if args.max_rows is not None and rows_read >= args.max_rows:
                break
            rows_read += 1
            row = json.loads(line)
            sources, is_video_frames = _image_sources(row)
            if not sources:
                continue
            rows_checked += 1
            if is_video_frames:
                video_rows_checked += 1
            else:
                image_rows_checked += 1
            images_checked += len(sources)
            geometry = tuple(_image_size(source) for source in sources)
            if not args.geometry_cache or geometry not in checked_geometries:
                checked_geometries.add(geometry)
                geometries_checked += 1
                if is_video_frames:
                    mismatch = _compare_video_row(
                        row_index=row_index,
                        row=row,
                        sources=sources,
                        processor=processor,
                        image_config=megatron_config,
                        video_num_frames=args.video_num_frames,
                        temporal_patch_size=args.video_temporal_patch_size,
                        target_num_patches=args.video_target_num_patches,
                        maintain_aspect_ratio=args.video_maintain_aspect_ratio,
                        pixel_atol=args.pixel_atol,
                    )
                else:
                    mismatch = _compare_row(
                        row_index=row_index,
                        sources=sources,
                        processor=processor,
                        megatron_config=megatron_config,
                        pixel_atol=args.pixel_atol,
                        pad_dynamic_image_shapes=args.pad_dynamic_image_shapes,
                    )
                if mismatch is not None:
                    mismatches.append(mismatch)
                    if len(mismatches) <= args.report_limit:
                        print(f"MISMATCH: {mismatch}", flush=True)
            if args.progress_every and rows_checked % args.progress_every == 0:
                print(
                    f"progress rows_read={rows_read} rows_checked={rows_checked} "
                    f"image_rows_checked={image_rows_checked} "
                    f"video_rows_checked={video_rows_checked} "
                    f"images_checked={images_checked} "
                    f"geometries_checked={geometries_checked} "
                    f"mismatches={len(mismatches)}",
                    flush=True,
                )

    print(
        f"complete rows_read={rows_read} rows_checked={rows_checked} "
        f"image_rows_checked={image_rows_checked} "
        f"video_rows_checked={video_rows_checked} "
        f"images_checked={images_checked} geometries_checked={geometries_checked} "
        f"mismatches={len(mismatches)}",
        flush=True,
    )
    if not rows_checked:
        print("ERROR: no still-image or cached-video-frame rows were found.", file=sys.stderr)
        return 2
    if mismatches:
        if len(mismatches) > args.report_limit:
            print(
                f"{len(mismatches) - args.report_limit} additional mismatches omitted.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
