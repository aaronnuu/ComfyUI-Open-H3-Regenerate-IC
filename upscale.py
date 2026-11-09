"""Author upscaler, loaded without the upstream sys.modules shims."""
from __future__ import annotations

import torch
from safetensors.torch import load_file

from .memory import chunks


def _windows(length, tile, overlap):
    if tile <= 0 or length <= tile:
        return [(0, length)]
    stride = tile - 2 * overlap
    if stride <= 0:
        raise ValueError("Upscaler tile must exceed twice its spatial overlap")
    starts = list(range(0, max(1, length - tile + 1), stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return [(s, min(length, s + tile)) for s in starts]


def _ramp(length, left, right, overlap):
    weight = torch.ones(length)
    n = min(overlap, length // 2)
    if n and left:
        weight[:n] = torch.arange(1, n + 1) / (n + 1)
    if n and right:
        weight[-n:] = torch.arange(n, 0, -1) / (n + 1)
    return weight


@torch.inference_mode()
def tiled_pass(model, native, shape, settings, spatial_tile, device, check=lambda: None):
    from .vendor.upscaler_model import _make_norm_tensors
    _, frames, src_h, src_w = native.shape
    _, _, dst_h, dst_w = shape
    if (src_h, src_w) == (dst_h, dst_w):
        return native.float().cpu()
    dtype = next(model.parameters()).dtype
    mean, std = _make_norm_tensors(device, dtype)
    out = torch.zeros(shape, dtype=torch.float32, device="cpu")
    scale = (dst_h / src_h + dst_w / src_w) / 2
    time_chunk = settings.upscale_frames or frames
    spatial_overlap = min(8, max(0, spatial_tile // 4))
    for ts, te in chunks(frames, time_chunk):
        check()
        lo = max(0, ts - settings.upscale_overlap)
        hi = min(frames, te + settings.upscale_overlap)
        weights = torch.zeros((dst_h, dst_w))
        for ys, ye in _windows(src_h, spatial_tile, spatial_overlap):
            for xs, xe in _windows(src_w, spatial_tile, spatial_overlap):
                check()
                dy, ey = round(ys * dst_h / src_h), round(ye * dst_h / src_h)
                dx, ex = round(xs * dst_w / src_w), round(xe * dst_w / src_w)
                tile = native[:, lo:hi, ys:ye, xs:xe][None].to(device, dtype)
                prediction = model._forward_seg((tile - mean) / std, scale, (hi - lo, ey - dy, ex - dx))
                prediction = (prediction * std + mean)[0, :, ts - lo:te - lo].float().cpu()
                wy = _ramp(ey - dy, ys > 0, ye < src_h, round(spatial_overlap * dst_h / src_h))
                wx = _ramp(ex - dx, xs > 0, xe < src_w, round(spatial_overlap * dst_w / src_w))
                weight = wy[:, None] * wx[None, :]
                out[:, ts:te, dy:ey, dx:ex] += prediction * weight
                weights[dy:ey, dx:ex] += weight
                del tile, prediction
        if not (weights > 0).all():
            raise RuntimeError("Upscaler tiling left uncovered pixels")
        out[:, ts:te] /= weights
    return out


@torch.inference_mode()
def upscale(path, native, shape, settings, spatial_tile, device, check=lambda: None):
    # Importing the author module is safe in ComfyUI. The standalone upstream
    # upscale.load() is deliberately unused: it overwrites global Comfy modules.
    from .vendor.upscaler_model import LatentResizer3D, _detect_arch, _extract_upscaler_sd
    state = _extract_upscaler_sd(load_file(str(path), device="cpu"))
    model = LatentResizer3D(**_detect_arch(state))
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device, torch.bfloat16).eval().requires_grad_(False)
    try:
        # Match the official two-pass route for resolutions above 2K.
        if shape[2] > 90:
            native = tiled_pass(model, native, [*shape[:2], 90, 160], settings, spatial_tile, device, check)
        return tiled_pass(model, native, shape, settings, spatial_tile, device, check)
    finally:
        del model
