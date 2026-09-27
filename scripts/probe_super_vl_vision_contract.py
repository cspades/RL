#!/usr/bin/env python3
"""Numerically isolate the Super-VL post-RADIO vision contract.

This probe intentionally loads only the checkpoint's RADIO tower and vision
projector.  It runs one still image and one cached-video tubelet through the
real checkpoint weights, then compares the two runtime contracts used here:

* patched vLLM/HF: RADIO -> checkpoint LayerNorm -> shuffle -> projector
* current MCore bridge commit 4d2cf452: RADIO -> shuffle -> projector

The JSON output records tensor fingerprints and the first stage where these
contracts differ.  It complements, rather than replaces, the raw-pixel parity
checker in check_super_vl_image_preprocessing_parity.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nemo_rl.data.multimodal_utils import (  # noqa: E402
    PackedTensor,
    attach_image_model_inputs_to_message,
    resolve_to_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def image_sources(row: dict[str, Any]) -> tuple[list[str], bool]:
    sources: list[str] = []
    has_still = False
    has_video = False
    for message in row.get("responses_create_params", {}).get("input", []):
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("input_image", "image", "image_url"):
                continue
            has_video |= bool(part.get("_is_video_frame"))
            has_still |= not bool(part.get("_is_video_frame"))
            source: Any = None
            for key in ("image_url", "image", "url"):
                source = part.get(key)
                if isinstance(source, dict):
                    source = source.get("url") or source.get("path")
                if isinstance(source, str) and source:
                    break
            if not isinstance(source, str) or not source:
                raise ValueError("Image content part has no source")
            sources.append(source)
    if has_still and has_video:
        raise ValueError("A row cannot mix stills and cached video frames")
    return sources, has_video


def load_rgb_image(source: str) -> Image.Image:
    with closing(resolve_to_image(source)) as image:
        return image.convert("RGB").copy()


def load_index(model_dir: Path) -> dict[str, str]:
    path = model_dir / "model.safetensors.index.json"
    return json.loads(path.read_text())["weight_map"]


def load_selected(
    model_dir: Path, weight_map: dict[str, str], names: list[str]
) -> dict[str, torch.Tensor]:
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(weight_map[name], []).append(name)
    result: dict[str, torch.Tensor] = {}
    for filename, file_names in by_file.items():
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for name in file_names:
                result[name] = handle.get_tensor(name)
    return result


def convert_radio_state_dict(
    source: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply the checkpoint's registered RADIO renaming/qkv split explicitly."""
    converted: dict[str, torch.Tensor] = {}
    direct_prefixes = {
        "radio_model.model.patch_generator.video_embedder": "embeddings.video_patch_projection",
        "radio_model.model.patch_generator.embedder": "embeddings.patch_projection",
        "radio_model.model.patch_generator.pos_embed": "embeddings.position_embedding",
        "radio_model.model.patch_generator.cls_token.token": "embeddings.cls_register_token",
    }
    for name, value in source.items():
        if name.startswith("radio_model.input_conditioner."):
            continue
        renamed = None
        for old_prefix, new_prefix in direct_prefixes.items():
            if name.startswith(old_prefix):
                renamed = new_prefix + name.removeprefix(old_prefix)
                break
        if renamed is not None:
            if renamed == "embeddings.cls_register_token" and value.ndim == 3:
                value = value.squeeze(0)
            converted[renamed] = value
            continue
        blocks_prefix = "radio_model.model.blocks."
        if not name.startswith(blocks_prefix):
            continue
        remainder = name.removeprefix(blocks_prefix)
        layer_index, suffix = remainder.split(".", 1)
        target_prefix = f"encoder.layer.{layer_index}."
        if suffix.startswith("attn.qkv."):
            qkv_suffix = suffix.removeprefix("attn.qkv.")
            query, key, val = value.chunk(3, dim=0)
            converted[target_prefix + f"attention.attention.query.{qkv_suffix}"] = query
            converted[target_prefix + f"attention.attention.key.{qkv_suffix}"] = key
            converted[target_prefix + f"attention.attention.value.{qkv_suffix}"] = val
            continue
        suffix = suffix.replace("attn.proj.", "attention.output.dense.")
        converted[target_prefix + suffix] = value
    return converted


def fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu().contiguous()
    raw = value.numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256_fp32": hashlib.sha256(raw).hexdigest(),
        "min": value.min().item(),
        "max": value.max().item(),
        "mean": value.mean().item(),
        "std": value.std().item(),
        "l2": torch.linalg.vector_norm(value).item(),
    }


def comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_fp32 = left.detach().float()
    right_fp32 = right.detach().float()
    delta = left_fp32 - right_fp32
    cosine = torch.nn.functional.cosine_similarity(
        left_fp32.reshape(1, -1), right_fp32.reshape(1, -1)
    ).item()
    return {
        "equal": torch.equal(left, right),
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(right_fp32).clamp_min(1.0e-30)
        ).item(),
        "cosine": cosine,
        "first_mismatch_flat_index": (
            None
            if torch.equal(left, right)
            else int((left_fp32 != right_fp32).flatten().nonzero()[0].item())
        ),
    }


def materialize_pixels(
    row: dict[str, Any], processor: Any
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    sources, is_video = image_sources(row)
    images = [load_rgb_image(source) for source in sources]
    message: dict[str, Any] = {
        "role": "user",
        "content": "",
        "token_ids": torch.empty(0, dtype=torch.long),
    }
    attach_image_model_inputs_to_message(
        message,
        images=images,
        processor=processor,
        pad_dynamic_image_shapes=False,
    )
    packed = message["pixel_values"]
    sizes = message["imgs_sizes"]
    if not isinstance(packed, PackedTensor) or not isinstance(sizes, PackedTensor):
        raise TypeError("Expected deferred PackedTensor media fields")
    return packed.as_tensor().squeeze(0), sizes.as_tensor(), is_video


def unpatchify(
    patches: torch.Tensor, sizes: torch.Tensor, patch_size: int
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    offset = 0
    patch_width = 3 * patch_size * patch_size
    if patches.shape[-1] != patch_width:
        raise ValueError(
            f"Expected patch width {patch_width}, got {patches.shape[-1]}"
        )
    for height_tensor, width_tensor in sizes:
        height, width = int(height_tensor), int(width_tensor)
        rows, columns = height // patch_size, width // patch_size
        count = rows * columns
        image_patches = patches[offset : offset + count]
        offset += count
        image = (
            image_patches.reshape(rows, columns, 3, patch_size, patch_size)
            .permute(2, 0, 3, 1, 4)
            .reshape(3, height, width)
        )
        outputs.append(image)
    if offset != patches.shape[0]:
        raise ValueError(f"Consumed {offset} patches, received {patches.shape[0]}")
    return outputs


def load_vision_modules(model_dir: Path, device: torch.device):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    radio_class = get_class_from_dynamic_module(
        "modeling_radio.RadioModel", model_dir
    )
    projector_class = get_class_from_dynamic_module(
        "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3VisionProjector",
        model_dir,
    )

    vision = radio_class(config.vision_config)
    vision.make_preprocessor_external()
    projector = projector_class(config)

    weight_map = load_index(model_dir)
    vision_names = [
        name for name in weight_map if name.startswith("vision_model.")
    ]
    loaded_vision = load_selected(model_dir, weight_map, vision_names)
    released_vision_state = {
        name.removeprefix("vision_model."): value
        for name, value in loaded_vision.items()
    }
    vision_state = convert_radio_state_dict(released_vision_state)
    missing, unexpected = vision.load_state_dict(vision_state, strict=False)
    missing_without_identity_layerscale = [
        name
        for name in missing
        if ".layer_scale" not in name and name != "summary_idxs"
    ]
    if missing_without_identity_layerscale or unexpected:
        raise RuntimeError(
            "RADIO weight mismatch: "
            f"missing={missing_without_identity_layerscale}, unexpected={unexpected}"
        )

    projector_names = [
        "mlp1.0.weight",
        "mlp1.1.weight",
        "mlp1.3.weight",
        "vision_projector.vision_final_layernorm.weight",
        "vision_projector.vision_final_layernorm.bias",
    ]
    loaded_projector = load_selected(model_dir, weight_map, projector_names)
    projector_state = {
        "mlp1.norm.weight": loaded_projector["mlp1.0.weight"],
        "mlp1.linear1.weight": loaded_projector["mlp1.1.weight"],
        "mlp1.linear2.weight": loaded_projector["mlp1.3.weight"],
        "vision_final_layernorm.weight": loaded_projector[
            "vision_projector.vision_final_layernorm.weight"
        ],
        "vision_final_layernorm.bias": loaded_projector[
            "vision_projector.vision_final_layernorm.bias"
        ],
    }
    missing, unexpected = projector.load_state_dict(projector_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Projector weight mismatch: missing={missing}, unexpected={unexpected}"
        )

    dtype = torch.bfloat16
    vision = vision.to(device=device, dtype=dtype).eval()
    projector = projector.to(device=device, dtype=dtype).eval()
    # The vLLM runtime patch deliberately keeps this checkpoint-backed norm in
    # fp32, then casts its output back to the RADIO activation dtype.
    projector.vision_final_layernorm.float()
    return config, vision, projector


@torch.inference_mode()
def run_case(
    *,
    name: str,
    images: list[torch.Tensor],
    is_video: bool,
    vision: torch.nn.Module,
    projector: torch.nn.Module,
    patch_size: int,
    temporal_patch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if is_video:
        selected = images[:temporal_patch_size]
        if len(selected) < temporal_patch_size:
            selected.extend([selected[-1]] * (temporal_patch_size - len(selected)))
        pixels = torch.stack(selected).to(device=device, dtype=torch.bfloat16)
        num_groups = math.ceil(pixels.shape[0] / temporal_patch_size)
        packed_video = pixels.reshape(
            num_groups,
            temporal_patch_size * pixels.shape[1],
            pixels.shape[2],
            pixels.shape[3],
        )
        pre_ln = vision(
            packed_video, use_video_patch_projection=True
        ).features
        height, width = pixels.shape[-2:]
    else:
        pixels = images[0].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        pre_ln = vision(pixels).features
        height, width = pixels.shape[-2:]

    # vLLM's runtime patch and HF apply this checkpoint-backed norm.  Current
    # MCore commit 4d2cf452 intentionally removed it from the Super provider.
    vllm_post_ln = projector.vision_final_layernorm(pre_ln.float()).to(
        pre_ln.dtype
    )
    mcore_post_radio = pre_ln

    grid_h, grid_w = height // patch_size, width // patch_size
    vllm_shuffled = projector.pixel_shuffle(
        vllm_post_ln.reshape(vllm_post_ln.shape[0], grid_h, grid_w, -1),
        scale_factor=projector.downsample_ratio,
    ).reshape(vllm_post_ln.shape[0], -1, vllm_post_ln.shape[-1] * 4)
    mcore_shuffled = projector.pixel_shuffle(
        mcore_post_radio.reshape(
            mcore_post_radio.shape[0], grid_h, grid_w, -1
        ),
        scale_factor=projector.downsample_ratio,
    ).reshape(
        mcore_post_radio.shape[0], -1, mcore_post_radio.shape[-1] * 4
    )
    vllm_projected = projector.mlp1(vllm_shuffled)
    mcore_projected = projector.mlp1(mcore_shuffled)

    return {
        "case": name,
        "input": fingerprint(pixels),
        "radio_output_shared_probe": fingerprint(pre_ln),
        "first_divergent_stage": "post_radio_final_layernorm",
        "vllm_post_radio": fingerprint(vllm_post_ln),
        "mcore_post_radio": fingerprint(mcore_post_radio),
        "post_radio_comparison": comparison(vllm_post_ln, mcore_post_radio),
        "pixel_shuffle_comparison": comparison(vllm_shuffled, mcore_shuffled),
        "projector_comparison": comparison(vllm_projected, mcore_projected),
        "vllm_projected": fingerprint(vllm_projected),
        "mcore_projected": fingerprint(mcore_projected),
    }


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    device = torch.device("cuda")

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    rows = [
        json.loads(line)
        for line in dataset.read_text().splitlines()
        if line.strip()
    ]
    still_row = next(row for row in rows if not image_sources(row)[1])
    video_row = next(row for row in rows if image_sources(row)[1])

    config, vision, projector = load_vision_modules(model_dir, device)
    results = []
    for name, row in (("multi_image_first_image", still_row), ("cached_video_first_tubelet", video_row)):
        patches, sizes, is_video = materialize_pixels(row, processor)
        images = unpatchify(patches, sizes, config.patch_size)
        results.append(
            run_case(
                name=name,
                images=images,
                is_video=is_video,
                vision=vision,
                projector=projector,
                patch_size=config.patch_size,
                temporal_patch_size=config.video_temporal_patch_size,
                device=device,
            )
        )

    layernorm = projector.vision_final_layernorm
    report = {
        "model": str(model_dir),
        "dataset": str(dataset),
        "contracts": {
            "vllm": "RADIO -> checkpoint final LayerNorm -> pixel shuffle -> MLP projector",
            "mcore_commit_4d2cf452": "RADIO -> pixel shuffle -> MLP projector",
        },
        "layernorm_weight": fingerprint(layernorm.weight),
        "layernorm_bias": fingerprint(layernorm.bias),
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
