"""Native H3 full-context attention with CPU Q/K/V storage and chunked projections/FFN.

Modified by aaronnuu (zoetrope), 2026-11-10: attention "native" leaves every attention forward to
ComfyUI, so its optimized backend runs (int8 with --use-ck-attention: 181 s a step against about
660 s for the SDPA path, 1080p on an RTX 3090) with Q/K/V on the GPU; only the FFN is chunked by
token_chunk. It needs about 12.5 GB of VRAM free at 1080p.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MemorySettings:
    token_chunk: int = 1024
    head_chunk: int = 4
    key_chunk: int = 2048
    attention: str = "sdpa"
    vae_tile: int = 256
    vae_overlap: int = 64
    upscale_frames: int = 5
    upscale_overlap: int = 5
    oom_retries: int = 2

    def reduced(self):
        return replace(self, token_chunk=max(64, self.token_chunk // 2),
                       head_chunk=max(1, self.head_chunk // 2),
                       key_chunk=max(128, self.key_chunk // 2), attention="online")


def chunks(length, size):
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for start in range(0, length, size):
        yield start, min(length, start + size)


def online_attention(q, k, v, key_chunk, device, check=lambda: None):
    """Stable exact softmax across ALL K/V, streamed from CPU. [B,H,N,D].

    Scores are FP32. No N x N attention matrix is created. Maxima and sums are
    combined with the online softmax recurrence, never per-chunk softmax averages.
    """
    q = q.to(device).float()
    maximum = torch.full((*q.shape[:-1], 1), -torch.inf, device=device)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros((*q.shape[:-1], v.shape[-1]), device=device)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for start, end in chunks(k.shape[-2], key_chunk):
        check()
        kk = k[..., start:end, :].to(device).float()
        vv = v[..., start:end, :].to(device).float()
        scores = torch.matmul(q, kk.transpose(-1, -2)) * scale
        new_max = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
        correction = torch.exp(maximum - new_max)
        weights = torch.exp(scores - new_max)
        numerator.mul_(correction).add_(weights @ vv)
        denominator.mul_(correction).add_(weights.sum(dim=-1, keepdim=True))
        maximum = new_max
    return numerator / denominator


def full_context_attention(q, k, v, settings, device, check=lambda: None):
    """CPU [B,N,H,D] in/out, all keys visible to every query."""
    result = torch.empty_like(q, device="cpu")
    for hs, he in chunks(q.shape[2], settings.head_chunk):
        check()
        # Transpose before transfer; head groups stay contiguous on the GPU.
        kk = k[:, :, hs:he].permute(0, 2, 1, 3)
        vv = v[:, :, hs:he].permute(0, 2, 1, 3)
        if settings.attention == "sdpa":
            kk = kk.contiguous().to(device)
            vv = vv.contiguous().to(device)
        for start, end in chunks(q.shape[1], settings.token_chunk):
            check()
            qq = q[:, start:end, hs:he].permute(0, 2, 1, 3).contiguous().to(device)
            if settings.attention == "online":
                out = online_attention(qq, kk, vv, settings.key_chunk, device, check).to(q.dtype)
            elif device.type == "cuda":
                # Never silently fall back to a quadratic-memory math kernel.
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    out = F.scaled_dot_product_attention(qq, kk, vv)
            else:
                out = F.scaled_dot_product_attention(qq, kk, vv)
            result[:, start:end, hs:he] = out.permute(0, 2, 1, 3).cpu()
        del kk, vv
    return result


def attention_forward(attn, settings):
    def run(x, rope_freqs=None, transformer_options=None):
        import comfy.model_management as mm
        import comfy.quant_ops
        check = mm.throw_exception_if_processing_interrupted
        n, h, d = len(x), attn.heads, attn.head_dim
        q, k, v = [torch.empty((1, n, h, d), dtype=x.dtype, device="cpu") for _ in range(3)]
        for start, end in chunks(n, settings.token_chunk):
            check()
            qq, kk, vv = attn.qkv_proj(x[start:end]).split(h * d, dim=-1)
            qq, kk = qq.reshape(1, end - start, h, d), kk.reshape(1, end - start, h, d)
            if rope_freqs is None:
                qq, kk = attn.q_norm(qq), attn.k_norm(kk)
            else:
                freqs = rope_freqs[:, start:end]
                comfy.quant_ops.ck.rms_rope_split_half_(
                    qq, kk, freqs, mm.cast_to(attn.q_norm.weight, device=x.device),
                    mm.cast_to(attn.k_norm.weight, device=x.device),
                    epsilon=attn.q_norm.eps, rot_dim=freqs.shape[-3] * 2)
            q[:, start:end], k[:, start:end] = qq.cpu(), kk.cpu()
            v[:, start:end] = vv.reshape(1, end - start, h, d).cpu()
            del qq, kk, vv
        result = full_context_attention(q, k, v, settings, x.device, check).reshape(n, h * d)
        del q, k, v
        out = torch.empty_like(x)
        for start, end in chunks(n, settings.token_chunk):
            check()
            out[start:end] = attn.out_proj(result[start:end].to(x.device))
        return out
    return run


def ffn_forward(mlp, settings):
    forward = type(mlp).forward

    def run(x):
        import comfy.model_management as mm
        out = torch.empty_like(x)
        for start, end in chunks(len(x), settings.token_chunk):
            mm.throw_exception_if_processing_interrupted()
            out[start:end] = forward(mlp, x[start:end])
        return out
    return run


def apply_memory(model, settings):
    """Only this ModelPatcher clone changes. Native quantization and offload remain active."""
    out = model.clone()
    dm = out.get_model_object("diffusion_model")
    for prefix, blocks in (("blocks", dm.blocks), ("token_refiner.blocks", dm.token_refiner.blocks)):
        for i, block in enumerate(blocks):
            root = f"diffusion_model.{prefix}.{i}"
            if settings.attention != "native":
                out.add_object_patch(root + ".attn.forward", attention_forward(block.attn, settings))
            out.add_object_patch(root + ".mlp.forward", ffn_forward(block.mlp, settings))
    out.model_options["open_h3_ic_memory"] = settings
    return out
