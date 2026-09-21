#!/usr/bin/env python3
"""Compare policy and Megatron raw-image preprocessing over a JSONL dataset."""

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
)
from nemo_rl.data.multimodal_utils import resolve_to_image  # noqa: E402
from nemo_rl.models.generation.megatron.utils import (  # noqa: E402
    build_image_preprocessing_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan ordinary input_image rows and require exact parity between the "
            "checkpoint HF processor and Megatron raw-image preprocessing."
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
        "--no-geometry-cache",
        action="store_true",
        help="Reprocess every row instead of checking each ordered image-size layout once",
    )
    return parser.parse_args()


def _content_part_source(part: dict[str, Any]) -> str | None:
    for key in ("image_url", "image", "url"):
        value = part.get(key)
        if isinstance(value, dict):
            value = value.get("url") or value.get("path")
        if isinstance(value, str) and value:
            return value
    return None


def _ordinary_image_sources(row: dict[str, Any]) -> list[str]:
    """Return initial still-image sources, excluding specialized video rows."""
    sources: list[str] = []
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
                return []
            if part_type not in ("input_image", "image", "image_url"):
                continue
            if part.get("_is_video_frame"):
                return []
            source = _content_part_source(part)
            if source is None:
                raise ValueError(f"{part_type} content part has no image source.")
            sources.append(source)
    return sources


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


def _as_image_list(pixel_values: Any) -> list[torch.Tensor]:
    if isinstance(pixel_values, torch.Tensor):
        if pixel_values.ndim != 4:
            raise ValueError(
                f"HF pixel_values must be NCHW, got {tuple(pixel_values.shape)}."
            )
        return [image for image in pixel_values]
    if isinstance(pixel_values, list):
        tensors = [torch.as_tensor(image) for image in pixel_values]
        if any(image.ndim != 3 for image in tensors):
            raise ValueError("HF ragged pixel_values must contain CHW tensors.")
        return tensors
    raise TypeError(
        "HF processor returned unsupported pixel_values type "
        f"{type(pixel_values).__name__}."
    )


def _pack_hf_images(
    images: list[torch.Tensor], patch_dim: int
) -> tuple[torch.Tensor, list[list[int]]]:
    packed: list[torch.Tensor] = []
    sizes: list[list[int]] = []
    for image in images:
        channels, height, width = image.shape
        if height % patch_dim or width % patch_dim:
            raise ValueError(
                f"HF image shape {(height, width)} is not divisible by {patch_dim}."
            )
        patch_rows, patch_cols = height // patch_dim, width // patch_dim
        patches = image.reshape(
            channels,
            patch_rows,
            patch_dim,
            patch_cols,
            patch_dim,
        )
        patches = patches.permute(1, 3, 0, 2, 4).contiguous()
        packed.append(patches.reshape(patch_rows * patch_cols, -1))
        sizes.append([height, width])
    return torch.cat(packed, dim=0), sizes


def _compare_row(
    *,
    row_index: int,
    sources: list[str],
    image_processor: Any,
    megatron_config: Any,
    pixel_atol: float,
) -> str | None:
    images = [_load_rgb_image(source) for source in sources]
    try:
        hf_output = dict(image_processor(images=images, return_tensors=None))
        hf_images = _as_image_list(hf_output["pixel_values"])
        hf_patches, hf_sizes = _pack_hf_images(hf_images, megatron_config.patch_dim)
        hf_tokens = [int(value) for value in hf_output["num_tokens"]]

        megatron_output = preprocess_image_bytes_list(
            [_encode_png(image) for image in images],
            megatron_config,
        )
        megatron_sizes = megatron_output["imgs_sizes"].tolist()
        megatron_patches = megatron_output["imgs"].squeeze(0)
        merge_area = megatron_config.spatial_merge_size**2
        megatron_tokens = [
            (height // megatron_config.patch_dim)
            * (width // megatron_config.patch_dim)
            // merge_area
            for height, width in megatron_sizes
        ]

        if hf_sizes != megatron_sizes:
            return (
                f"row={row_index} size mismatch hf={hf_sizes} "
                f"megatron={megatron_sizes} sources={sources}"
            )
        if hf_tokens != megatron_tokens:
            return (
                f"row={row_index} token mismatch hf={hf_tokens} "
                f"megatron={megatron_tokens} sources={sources}"
            )
        if hf_patches.shape != megatron_patches.shape:
            return (
                f"row={row_index} patch-shape mismatch hf={tuple(hf_patches.shape)} "
                f"megatron={tuple(megatron_patches.shape)} sources={sources}"
            )
        if not torch.allclose(
            hf_patches,
            megatron_patches,
            rtol=0.0,
            atol=pixel_atol,
            equal_nan=True,
        ):
            max_error = (hf_patches - megatron_patches).abs().max().item()
            return (
                f"row={row_index} pixel mismatch max_abs_error={max_error:.9g} "
                f"sizes={hf_sizes} sources={sources}"
            )
        return None
    finally:
        for image in images:
            image.close()


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
            sources = _ordinary_image_sources(row)
            if not sources:
                continue
            rows_checked += 1
            images_checked += len(sources)
            geometry = tuple(_image_size(source) for source in sources)
            if args.no_geometry_cache or geometry not in checked_geometries:
                checked_geometries.add(geometry)
                geometries_checked += 1
                mismatch = _compare_row(
                    row_index=row_index,
                    sources=sources,
                    image_processor=image_processor,
                    megatron_config=megatron_config,
                    pixel_atol=args.pixel_atol,
                )
                if mismatch is not None:
                    mismatches.append(mismatch)
                    if len(mismatches) <= args.report_limit:
                        print(f"MISMATCH: {mismatch}", flush=True)
            if args.progress_every and rows_checked % args.progress_every == 0:
                print(
                    f"progress rows_read={rows_read} rows_checked={rows_checked} "
                    f"images_checked={images_checked} "
                    f"geometries_checked={geometries_checked} "
                    f"mismatches={len(mismatches)}",
                    flush=True,
                )

    print(
        f"complete rows_read={rows_read} rows_checked={rows_checked} "
        f"images_checked={images_checked} geometries_checked={geometries_checked} "
        f"mismatches={len(mismatches)}",
        flush=True,
    )
    if not rows_checked:
        print("ERROR: no ordinary-image rows were found.", file=sys.stderr)
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
