"""Stage-separated IC pipeline. Never retains models in ComfyUI output caches."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
import gc
import hashlib
import json
import logging
import math

import torch

from .memory import MemorySettings, install_streaming

LOG = logging.getLogger("OpenH3IC")
RESOLUTIONS = {"1080p": (768, 1344, 1088, 1920, 1080), "2k": (768, 1344, 1440, 2560, 1440),
               "4k": (768, 1344, 2176, 3840, 2160), "4k-dci": (768, 1344, 2304, 4096, 2304)}
LOW_SIZES = {"384": (384, 672), "512": (512, 896), "576": (576, 1024)}


@dataclass(frozen=True)
class PipelineConfig:
    weights: Path
    memory: MemorySettings
    offload_dir: Path
    upscale_tile: int = 64

    @property
    def base(self):
        return self.weights / "base_model"

    def manifest(self):
        return json.loads((self.weights / "config.json").read_text(encoding="utf-8"))


def resolution_spec(resolution, lq):
    if resolution == "768p":
        return (*LOW_SIZES[lq], 768, 1344, 768)
    return RESOLUTIONS[resolution]


def validate_weights(config, sharp=1., decoder="standard", sampler="up3"):
    if not (config.weights / "config.json").is_file():
        raise FileNotFoundError(f"IC weights/config.json not found: {config.weights}. "
                                "Run scripts/download_models.py --out <this folder>; see README.md.")
    manifest = config.manifest()
    if manifest.get("base_component") != "transformer_ref":
        raise ValueError("These nodes require Open-H3-Regenerate-2K-IC weights (transformer_ref), not the older refiner.")
    components = manifest["components"]
    required = ["lora", "hyperflow", "silence_audio"]
    if sampler != "hf8":
        required.append("latent_upsampler")
    if sharp:
        required += ["sharp_lora", "sharp_curve"]
    if decoder.startswith("fast"):
        required.append("vae_fast")
    missing = [str(config.weights / components[k]) for k in required
               if not (config.weights / components[k]).is_file()]
    for name in ("transformer_ref", "vae", "text_encoder", "tokenizer", "processor"):
        if not (config.base / name).is_dir():
            missing.append(str(config.base / name))
    if missing:
        raise FileNotFoundError("Missing IC components:\n" + "\n".join(missing))
    return manifest


def dependency_check():
    try:
        from diffusers import MiniMaxH3Transformer3DModel, AutoencoderKLMiniMaxH3
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
        from transformers import Qwen3VLForConditionalGeneration
        import accelerate
        return MiniMaxH3Transformer3DModel, AutoencoderKLMiniMaxH3, MiniMaxH3Ref2VATextEncoderStep
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("IC needs diffusers>=0.40 with MiniMax-H3 modular pipelines, "
                           "transformers>=5.6, and accelerate. See requirements.txt. "
                           f"Dependency import failed: {exc}") from exc


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execution_device():
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()
    device = mm.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError("Open-H3 IC requires an NVIDIA CUDA GPU.")
    mm.unload_all_models()
    mm.soft_empty_cache()
    return device


def interrupt():
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()


def _offload(model, config, device, stage, extra=""):
    mode = config.memory.weight_offload
    if mode == "cpu":
        from accelerate import cpu_offload
        cpu_offload(model, execution_device=device, offload_buffers=True)
    elif mode == "disk":
        from accelerate import disk_offload
        # The adapters, strength and weight mtimes are part of the cache identity.
        # A cache can therefore never silently substitute a different LoRA mix.
        files = [config.weights / "config.json"]
        curve_file = config.weights / config.manifest()["components"]["sharp_curve"]
        if curve_file.is_file():
            files.append(curve_file)
        files += sorted((config.weights / "lora").glob("*.safetensors"))
        files += sorted((config.base / ("text_encoder" if stage == "text" else "transformer_ref")).glob("*.safetensors"))
        fingerprint = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
        key = hashlib.sha256((repr(fingerprint) + extra + torch.__version__).encode()).hexdigest()[:16]
        destination = config.offload_dir / f"{stage}-{key}"
        destination.mkdir(parents=True, exist_ok=True)
        disk_offload(model, destination, execution_device=device, offload_buffers=True)
    elif mode == "gpu":
        model.to(device)
    else:
        raise ValueError(f"Unknown offload mode: {mode}")


def prepare_frames(images, fps, height, width):
    from .vendor.ref2va_ic.media import FRAMES, MIN_REAL, _cover
    if images.ndim != 4 or images.shape[-1] < 3 or not len(images):
        raise ValueError("images must be a nonempty IMAGE batch [frames,height,width,3]")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Input fps must be positive")
    indices = [int(i * fps / 24.) for i in range(FRAMES)]
    valid = sum(index < len(images) for index in indices)
    if valid < MIN_REAL:
        raise ValueError(f"Only {valid} frames at 24 fps; IC requires at least 113 (about 4.7 seconds). "
                         "Load a longer video or check input_fps.")
    pixels = torch.empty((FRAMES, 3, height, width), dtype=torch.float32, device="cpu")
    for i, index in enumerate(indices[:valid]):
        frame = images[index, ..., :3].detach().permute(2, 0, 1)[None].cpu().float()
        pixels[i] = _cover(frame, height, width)[0].clamp(0, 1)
    pixels[valid:] = pixels[valid - 1].clone()
    return pixels, valid


def install_vae_cpu_stitching(vae, device, check=interrupt):
    """Only the current VAE clip/tile and its activations are kept on the GPU."""
    for name in ("_encode_clip", "_decode_clip"):
        original = getattr(vae, name)
        def clip(self, x, _original=original):
            check()
            return _original(x.to(device)).cpu()
        setattr(vae, name, MethodType(clip, vae))
    vae.decoder.register_forward_hook(lambda module, args, result: result.cpu())
    vae.quant_conv.register_forward_hook(lambda module, args, result: result.cpu())
    return vae


def _vae(config, device):
    _, cls, _ = dependency_check()
    vae = cls.from_pretrained(str(config.base), subfolder="vae", torch_dtype=torch.float32,
                             local_files_only=True, low_cpu_mem_usage=True).eval().requires_grad_(False).to(device)
    vae.enable_tiling(tile_sample_min_height=config.memory.vae_tile,
                      tile_sample_min_width=config.memory.vae_tile,
                      tile_sample_min_overlap_height=config.memory.vae_overlap,
                      tile_sample_min_overlap_width=config.memory.vae_overlap)
    vae.set_attention_backend("native")
    # Stitch spatial and temporal outputs in RAM. The released decoder's native
    # overlap/temporal chunk geometry is preserved (never arbitrary frame cuts).
    return install_vae_cpu_stitching(vae, device)


@torch.inference_mode()
def encode_draft(config, images, fps, caption, resolution, lq, seed, progress):
    from .vendor.ref2va_ic import encode
    dependency_check()
    h, w, _, _, keep = resolution_spec(resolution, lq)
    pixels, real = prepare_frames(images, fps, h, w)
    device = execution_device()
    progress(0, 3)
    vae = text = None
    try:
        vae = _vae(config, device)
        mean, std, pm, ps = encode._stats(vae, "cpu")
        z = vae.encode((pixels.permute(1, 0, 2, 3)[None] - pm) / ps, return_dict=False)[0]
        latent = ((z.sample(generator=torch.Generator().manual_seed(seed % (1 << 62))) - mean) / std)[0].bfloat16()
        del z, vae
        vae = None
        cleanup()
        progress(1, 3)
        _, _, cls = dependency_check()
        text = cls().init_pipeline(str(config.base))
        text.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(config.base), local_files_only=True)
        text.text_encoder.eval().requires_grad_(False)
        _offload(text.text_encoder, config, device, "text")
        prompt = encode.encode_prompt(text, pixels, caption)
        progress(3, 3)
        if not torch.isfinite(latent).all():
            raise RuntimeError("VAE returned nonfinite latents")
        return {"latent": latent.cpu(), "prompt": prompt, "frames": real,
                "resolution": resolution, "lq": lq, "keep_height": keep,
                "weights": str(config.weights), "fps": 24.}
    finally:
        del vae, text, pixels
        cleanup()


def sigma_plan(meta, sampler):
    from .vendor.ref2va_ic import hyperflow
    grid = hyperflow.sigmas(meta, 12.)
    if sampler == "hf8":
        return grid
    if sampler in ("up3", "up4"):
        return grid[8 - int(sampler[2:]):]
    raise ValueError(f"Unknown sampler: {sampler}")


@torch.inference_mode()
def sample(config, draft, sampler, sharp, seed, progress):
    from safetensors.torch import load_file
    from .vendor.ref2va_ic import hyperflow, layout, lora, sharp as sharp_lib
    from .upscale import upscale
    cls, _, _ = dependency_check()
    manifest = validate_weights(config, sharp, sampler=sampler)
    device = execution_device()
    comp = manifest["components"]
    _, _, h, w, _ = resolution_spec(draft["resolution"], draft["lq"])
    shape = [24, draft["latent"].shape[1], h // 16, w // 16]
    model = None
    try:
        # Upscale before loading the large transformer; the two never share VRAM.
        up = None
        if sampler != "hf8":
            up = layout.patch_rows(upscale(config.weights / comp["latent_upsampler"], draft["latent"], shape,
                                          config.memory, config.upscale_tile, device, interrupt))
            cleanup()
        LOG.info("Loading IC Ref2VA transformer with %s weight offload", config.memory.weight_offload)
        model = cls.from_pretrained(str(config.base), subfolder="transformer_ref", torch_dtype=torch.bfloat16,
                                    local_files_only=True, low_cpu_mem_usage=True).eval().requires_grad_(False)
        meta = hyperflow.install(model, config.weights / comp["hyperflow"], "cpu")
        if sharp:
            sharp_lib.install(model, config.weights / comp["sharp_lora"], config.weights / comp["sharp_curve"], sharp, "cpu")
        lora.install(model, config.weights / comp["lora"])
        model.set_attention_backend("native")
        install_streaming(model, config.memory, device, interrupt)
        _offload(model, config, device, "transformer", f"sharp={sharp};ic-v1")
        audio = load_file(str(config.weights / comp["silence_audio"]))["audio_latents"]
        ref, aud, prompt, packed, n = layout.pack(draft["latent"], shape, draft["prompt"], audio, "cpu")
        sigmas = sigma_plan(meta, sampler)
        # CUDA RNG order matches the original: anchor noise, then target noise.
        rng = torch.Generator(device=device).manual_seed(seed % (1 << 62))
        anchor = .999 * ref.float() + .001 * torch.randn(ref.shape, generator=rng, device=device).cpu()
        rows = shape[1] * (shape[2] // 2) * (shape[3] // 2)
        x = torch.randn((1, rows, 96), generator=rng, device=device, dtype=torch.float32).cpu()
        if up is not None:
            x = (1 - sigmas[0]) * up + sigmas[0] * x
            del up
        steps = len(sigmas) - 1
        for step, (sigma, nxt) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            interrupt()
            times, indices = layout.row_timesteps(packed, sigma)
            model.time_embedder.endpoint_timesteps = hyperflow.endpoints(times, layout.canonical_sigma(sigma), nxt)
            model._ic_block_progress = lambda block, count: progress(step * count + block, steps * count)
            retries = 0
            while True:
                failed = False
                try:
                    velocity = model(hidden_states=torch.cat([anchor, x], 1), audio_hidden_states=aud,
                                     encoder_hidden_states=prompt, timestep=times, timestep_indices=indices,
                                     token_tags=packed.token_tags, position_ids=packed.position_ids,
                                     video_indices=packed.video_indices, audio_indices=packed.audio_indices,
                                     text_indices=packed.text_indices, return_dict=False)[0]
                except torch.cuda.OutOfMemoryError:
                    failed = True
                if not failed:
                    break
                cleanup()
                if retries >= config.memory.oom_retries:
                    raise RuntimeError("IC ran out of VRAM after reducing chunks. Try 768p/1080p, token_chunk=256, "
                                       "head_chunk=1, attention=online and weight_offload=disk.") from None
                retries += 1
                model._ic_memory = model._ic_memory.reduced()
                LOG.warning("VRAM exhausted; retrying unchanged step %d with %s", step + 1, model._ic_memory)
            # The reference rows remain anchored and never enter the Euler update.
            x.add_(velocity[:, n:].float(), alpha=sigma - nxt)
            del velocity
        if not torch.isfinite(x).all():
            raise RuntimeError("IC sampler returned nonfinite latents")
        return {"samples": layout.unpatch_rows(x, shape)[None].bfloat16(),
                "open_h3_ic": {k: draft[k] for k in ("frames", "keep_height", "resolution", "fps")}}
    finally:
        del model
        cleanup()


@torch.inference_mode()
def decode(config, latent, decoder, progress):
    from .vendor.ref2va_ic import encode
    samples, info = latent["samples"], latent.get("open_h3_ic")
    if info is None or samples.ndim != 5 or samples.shape[:2] != (1, 24):
        raise ValueError("Connect the LATENT output of Open H3 IC Sampler to this node.")
    device = execution_device()
    vae = fast = None
    pixels = None
    try:
        progress(0, 2)
        if decoder != "standard":
            from .vendor.ref2va_ic.fast_vae import FastVAE
            engine = "ref" if decoder == "fast_ref" else "auto"
            path = config.weights / config.manifest()["components"]["vae_fast"]
            if path.is_file():
                fast = FastVAE(path, config.base, engine=engine, device=str(device))
                if fast.fits(samples[0]):
                    # A real OOM still falls back to the tiled released VAE.
                    try:
                        pixels = fast.decode_frames(samples[0]).cpu().float().div_(255)
                    except torch.cuda.OutOfMemoryError:
                        pass
                del fast
                fast = None
                cleanup()
            if pixels is None:
                LOG.warning("Fast VAE unavailable or insufficient VRAM; using tiled standard VAE")
        if pixels is None:
            vae = _vae(config, device)
            mean, std, pm, ps = encode._stats(vae, "cpu")
            raw = vae.decode(samples.float().cpu() * std + mean, return_dict=False)[0][0]
            # Normalize in small temporal slices, keeping the final IMAGE on CPU.
            pixels = torch.empty((min(info["frames"], raw.shape[1]), raw.shape[2], raw.shape[3], 3))
            for t in range(0, len(pixels), 4):
                pixels[t:t + 4] = (raw[:, t:min(t + 4, len(pixels))] * ps[0] + pm[0]).clamp(0, 1).permute(1, 2, 3, 0)
            del raw
        pixels = pixels[:info["frames"]]
        top = (pixels.shape[1] - info["keep_height"]) // 2
        progress(2, 2)
        return pixels[:, top:top + info["keep_height"]], 24.
    finally:
        del vae, fast
        cleanup()
