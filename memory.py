"""Exact full-context H3 forward with CPU activation storage.

Only row chunks, a group of attention heads, and the currently executing weights
need to fit on the GPU. This avoids allocating full-sequence FFN intermediates,
Q/K/V, AdaLN broadcasts, and residual copies on the GPU. Arithmetic follows the
Diffusers MiniMaxH3Transformer3DModel; it does not split the attention document.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MethodType
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MemorySettings:
    token_chunk: int = 1024
    head_chunk: int = 4
    key_chunk: int = 2048
    attention: str = "sdpa"
    weight_offload: str = "cpu"
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


def _dtype(module):
    from diffusers.models.modeling_utils import get_parameter_dtype
    return get_parameter_dtype(module)


def _project_to_cpu(module, x, size, device, check):
    out = None
    for start, end in chunks(x.shape[1], size):
        check()
        y = module(x[:, start:end].to(device=device, dtype=_dtype(module)))
        if out is None:
            out = torch.empty((x.shape[0], x.shape[1], y.shape[-1]), dtype=y.dtype, device="cpu")
        out[:, start:end] = y.cpu()
    return out


def _modulated_norm(block, x, indices, table, which, device):
    shift, scale = table[which:which + 2]
    norm = block.norm1 if which == 0 else block.norm2
    return norm(x.to(device)) * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def streamed_forward(self, hidden_states, audio_hidden_states, encoder_hidden_states,
                     timestep, timestep_indices, token_tags, position_ids,
                     video_indices, audio_indices, text_indices,
                     attention_kwargs=None, return_dict=True):
    """Drop-in forward; weights may be resident, CPU-offloaded or disk-offloaded."""
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3TransformerOutput, _apply_rotary_emb,
    )
    settings, device, check = self._ic_memory, self._ic_device, self._ic_check
    check()
    video_indices, audio_indices, text_indices = [
        i.cpu() for i in (video_indices, audio_indices, text_indices)]
    position_ids, timestep_indices, token_tags = [
        i.cpu() for i in (position_ids, timestep_indices, token_tags)]
    timestep = timestep.to(device)
    # The token-refiner sequence is small (Qwen vision/text context only).
    text = self.context_embedder(encoder_hidden_states.to(device, _dtype(self.context_embedder)))
    text = self.token_refiner(text)
    x = torch.zeros((text.shape[0], len(position_ids), text.shape[-1]), dtype=text.dtype, device="cpu")
    x.index_copy_(1, text_indices, text.cpu())
    del text
    for projection, rows, indices in ((self.proj_in, hidden_states, video_indices),
                                      (self.audio_proj_in, audio_hidden_states, audio_indices)):
        projected = _project_to_cpu(projection, rows, settings.token_chunk, device, check)
        x.index_copy_(1, indices, projected.to(x.dtype))
        del projected
    temb = self.time_embedder(self.time_proj(timestep).to(_dtype(self.time_embedder)))
    adaln = timestep_indices * 3 + token_tags
    for block_index, block in enumerate(self.transformer_blocks):
        check()
        table = block.adaln_proj(temb)
        q = k = v = None
        # Store full Q/K/V on CPU; GPU projections and RoPE see only a row chunk.
        for start, end in chunks(x.shape[1], settings.token_chunk):
            check()
            idx = adaln[start:end].to(device)
            norm = _modulated_norm(block, x[:, start:end], idx, table, 0, device)
            rotary = self.rope(position_ids[start:end].to(device))
            qq = block.attn.norm_q(block.attn.to_q(norm).unflatten(-1, (block.attn.heads, -1)))
            kk = block.attn.norm_k(block.attn.to_k(norm).unflatten(-1, (block.attn.heads, -1)))
            vv = block.attn.to_v(norm).unflatten(-1, (block.attn.heads, -1))
            qq, kk = _apply_rotary_emb(qq, *rotary), _apply_rotary_emb(kk, *rotary)
            if q is None:
                shape = (x.shape[0], x.shape[1], *qq.shape[2:])
                q, k, v = [torch.empty(shape, dtype=qq.dtype, device="cpu") for _ in range(3)]
            q[:, start:end], k[:, start:end], v[:, start:end] = qq.cpu(), kk.cpu(), vv.cpu()
            del norm, qq, kk, vv, rotary
        attn = full_context_attention(q, k, v, settings, device, check).flatten(2, 3)
        del q, k, v
        # AdaLN, residual and FFN intermediates are chunked as well. No broadcast
        # of a full sequence's six AdaLN tables is ever materialized on the GPU.
        for start, end in chunks(x.shape[1], settings.token_chunk):
            check()
            idx = adaln[start:end].to(device)
            residual = x[:, start:end].to(device)
            out = block.attn.to_out[1](block.attn.to_out[0](attn[:, start:end].to(device)))
            state = residual + table[2].index_select(0, idx) * out
            norm = _modulated_norm(block, state, idx, table, 3, device)
            state = state + table[5].index_select(0, idx) * block.ff(norm)
            x[:, start:end] = state.cpu()
            del residual, out, state, norm
        del attn, table
        if self._ic_block_progress is not None:
            self._ic_block_progress(block_index + 1, len(self.transformer_blocks))
    outputs = []
    for projection, indices in ((self.proj_out, video_indices), (self.audio_proj_out, audio_indices)):
        result = torch.empty((x.shape[0], len(indices), projection.out_features), dtype=_dtype(projection), device="cpu")
        for start, end in chunks(len(indices), settings.token_chunk):
            check()
            selected = indices[start:end]
            norm = self.norm_out(x.index_select(1, selected).to(device), temb,
                                 timestep_indices[selected].to(device))
            result[:, start:end] = projection(norm.to(_dtype(projection))).cpu()
        outputs.append(result)
    if not return_dict:
        return tuple(outputs)
    return MiniMaxH3TransformerOutput(sample=outputs[0], audio_sample=outputs[1])


def install_streaming(model, settings, device, check=lambda: None, block_progress=None):
    model._ic_memory, model._ic_device = settings, torch.device(device)
    model._ic_check, model._ic_block_progress = check, block_progress
    model.forward = MethodType(streamed_forward, model)
    return model
