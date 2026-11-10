"""Native ComfyUI conditioning, tiled VAE and sampling."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import gc
import math

import torch
import torch.nn.functional as F

from .memory import MemorySettings, apply_memory

TASK = ("Enhance <Video 1> into a high-quality version of the same video. "
        "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
        "Restore fine details and remove degradation. Do not continue or change the scene.")
RESOLUTIONS = {"768p": (768, 1344, 768), "1080p": (1088, 1920, 1080),
               "2k": (1440, 2560, 1440), "4k": (2176, 3840, 2160), "4k-dci": (2304, 4096, 2304)}


def prepare_draft(images, input_fps, resolution, lq="576"):
    if images.ndim != 4 or images.shape[-1] < 3 or len(images) == 0:
        raise ValueError("Connect video frames (IMAGE [frames,height,width,channels])")
    if not math.isfinite(input_fps) or input_fps <= 0:
        raise ValueError("input_fps must be positive")
    wanted = [int(i * input_fps / 24) for i in range(124)]
    real = sum(i < len(images) for i in wanted)
    if real < 113:
        raise ValueError(f"The input supplies {real} frames at 24 fps; IC needs at least 113 (~4.7 seconds)")
    height, width, keep = RESOLUTIONS[resolution]
    dh, dw = (int(lq), {"384": 672, "512": 896, "576": 1024}[lq]) if resolution == "768p" else (768, 1344)
    output = torch.empty((124, dh, dw, 3), dtype=torch.float32, device="cpu")
    for start in range(0, 124, 4):
        idx = torch.tensor([min(i, len(images) - 1) for i in wanted[start:start + 4]], device=images.device)
        x = images.index_select(0, idx)[..., :3].movedim(-1, 1).float()
        h, w = x.shape[-2:]
        scale = max(dh / h, dw / w)
        sh, sw = max(dh, round(h * scale)), max(dw, round(w * scale))
        if (sh, sw) != (h, w):
            x = F.interpolate(x, (sh, sw), mode="bilinear", align_corners=False, antialias=scale < 1)
        y, z = (sh - dh) // 2, (sw - dw) // 2
        output[start:start + len(x)] = x[:, :, y:y + dh, z:z + dw].movedim(1, -1).clamp(0, 1).cpu()
    return output, real, (height, width, keep)


@contextmanager
def tiled_vae(vae, tile, overlap, decode=False):
    native = vae.first_stage_model
    if not hasattr(native, "clip_length") or not hasattr(native, "latents_mean"):
        raise ValueError("Select the MiniMax H3 video VAE with a stock tiled encoder/decoder")
    if decode and getattr(native, "noncausal", False):
        raise ValueError("The Sol noncausal decoder cannot tile. Use the standard H3 video VAE for bounded VRAM")
    if overlap >= tile or overlap < 0 or tile < 128 or tile % 16 or overlap % 16:
        raise ValueError("VAE tile/overlap must be multiples of 16, tile >=128, and 0 <= overlap < tile")
    # Core's H3 ignores generic tiling kwargs; use its semantic tile sizes.
    old = native.tiling, native.tile_size, native.tile_overlap_min
    native.tiling, native.tile_size, native.tile_overlap_min = True, tile, overlap
    try:
        yield
    finally:
        native.tiling, native.tile_size, native.tile_overlap_min = old


def silence_latent(path=None):
    from safetensors.torch import load_file
    path = Path(path or Path(__file__).parent / "assets" / "silence_audio_124f.safetensors")
    if not path.is_file():
        raise FileNotFoundError("Missing official silence asset. Run scripts/download_models.py --assets-only")
    sd = load_file(str(path))
    if len(sd) != 1:
        raise ValueError("Unexpected silence asset tensors")
    z = next(iter(sd.values()))
    if z.ndim == 4 and z.shape[0] == 1:
        z = z[0]
    if tuple(z.shape) != (2, 32, 207):
        raise ValueError(f"Unexpected silence latent shape: {tuple(z.shape)}")
    return z.permute(1, 0, 2).unsqueeze(0).float().contiguous()


@torch.inference_mode()
def encode_draft(clip, vae, images, input_fps, caption, resolution, lq, upscaler_path,
                 vae_tile=256, vae_overlap=64, upscale_frames=5, upscale_overlap=5, upscale_tile=64, audio=None):
    import comfy.model_management as mm
    from comfy.nested_tensor import NestedTensor
    import node_helpers
    from .upscale import upscale
    frames, real, (height, width, keep) = prepare_draft(images, input_fps, resolution, lq)
    with tiled_vae(vae, vae_tile, vae_overlap):
        z = vae.encode_tiled(frames).float().cpu()
    if tuple(z.shape) != (1, 24, 37, frames.shape[1] // 16, frames.shape[2] // 16):
        raise ValueError(f"Unexpected H3 VAE latent shape {tuple(z.shape)}")
    items = [{"type": "video", "data": frames[::12], "timestamps": [i / 2 for i in range(len(frames[::12]))]}]
    tokens = clip.tokenize(TASK + "\n\nScene description: " + caption, minimax_ref_items=items)
    positive = clip.encode_from_tokens_scheduled(tokens)
    refs = [{"kind": "video", "latent_t": 37, "latent_h": z.shape[3], "latent_w": z.shape[4],
             "ref_audio_t": 0, "latent": z, "audio_latent": None}]
    positive = node_helpers.conditioning_set_values(positive, {"minimax_refs": refs})
    del frames, tokens
    mm.unload_all_models()
    mm.soft_empty_cache()
    settings = MemorySettings(upscale_frames=upscale_frames, upscale_overlap=upscale_overlap)
    video = upscale(upscaler_path, z[0], (24, 37, height // 16, width // 16),
                    settings, upscale_tile, mm.get_torch_device(), mm.throw_exception_if_processing_interrupted)[None]
    silence = silence_latent()
    latent = {"samples": NestedTensor((video, silence)),
              "noise_mask": NestedTensor((torch.ones_like(video), torch.zeros_like(silence))),
              "open_h3_ic": {"real_frames": real, "height": keep, "width": width, "audio": audio}}
    return positive, latent


@torch.inference_mode()
def sample(model, positive, latent, sigmas, seed, oom_retries):
    import comfy.model_management as mm
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    from comfy.nested_tensor import NestedTensor
    recipe = model.model_options.get("open_h3_ic")
    if recipe is None or "minimax_refs" not in positive[0][1] or "open_h3_ic" not in latent:
        raise ValueError("Use the IC LoRA Loader and IC Encode outputs with this sampler")
    source = latent["samples"]
    video, audio = source.unbind()
    if recipe["sampler"] == "hf8":
        source = NestedTensor((torch.zeros_like(video), audio))
    noise = comfy.sample.prepare_noise(source, seed)
    settings = model.model_options["open_h3_ic_memory"]
    current = model
    for attempt in range(oom_retries + 1):
        try:
            progress = comfy.utils.ProgressBar(len(sigmas) - 1)
            result = comfy.sample.sample_custom(current, noise, 1.0, comfy.samplers.ksampler("euler"), sigmas,
                positive, positive, source, noise_mask=latent["noise_mask"], seed=seed,
                callback=lambda step, x0, x, total: progress.update_absolute(step + 1, total))
            out = latent.copy()
            out["samples"] = result
            return out
        except torch.cuda.OutOfMemoryError:
            if attempt >= oom_retries:
                raise
        # Leave the exception scope before GC. Restart from identical noise.
        import logging
        settings = settings.reduced()
        logging.warning("IC CUDA OOM: restarting sampling with token=%d, heads=%d, keys=%d, online attention",
                        settings.token_chunk, settings.head_chunk, settings.key_chunk)
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        current = apply_memory(model, settings)


@torch.inference_mode()
def decode(vae, latent, tile=256, overlap=64):
    meta = latent.get("open_h3_ic")
    if meta is None:
        raise ValueError("Connect the output of IC Sampler")
    video = latent["samples"].unbind()[0]
    with tiled_vae(vae, tile, overlap, decode=True):
        frames = vae.decode_tiled(video)
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    h, w = meta["height"], meta["width"]
    y, x = (frames.shape[1] - h) // 2, (frames.shape[2] - w) // 2
    frames = frames[:meta["real_frames"], y:y + h, x:x + w, :3]
    audio = meta.get("audio")
    rate = int(audio["sample_rate"]) if audio is not None else 32000
    length = round(len(frames) / 24 * rate)
    wave = audio["waveform"][..., :length] if audio is not None else torch.zeros(1, 2, length)
    wave = F.pad(wave, (0, max(0, length - wave.shape[-1])))
    return frames, 24.0, {"waveform": wave, "sample_rate": rate}
