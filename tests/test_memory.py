import copy

import pytest
import torch
import torch.nn.functional as F

from openh3ic_test.memory import MemorySettings, full_context_attention, install_streaming, online_attention


@pytest.mark.parametrize("key_chunk", [1, 7, 64])
def test_online_softmax_uses_all_keys(key_chunk):
    torch.manual_seed(10)
    q, k, v = torch.randn(2, 3, 11, 8), torch.randn(2, 3, 29, 8), torch.randn(2, 3, 29, 8)
    expected = F.scaled_dot_product_attention(q, k, v)
    actual = online_attention(q, k, v, key_chunk, torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("mode", ["online", "sdpa"])
def test_head_and_query_chunks_preserve_full_context(mode):
    torch.manual_seed(11)
    q, k, v = [torch.randn(1, 31, 5, 12) for _ in range(3)]
    expected = F.scaled_dot_product_attention(*[x.permute(0, 2, 1, 3) for x in (q, k, v)]).permute(0, 2, 1, 3)
    settings = MemorySettings(token_chunk=7, head_chunk=2, key_chunk=11, attention=mode)
    actual = full_context_attention(q, k, v, settings, torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def tiny_case():
    from diffusers import MiniMaxH3Transformer3DModel
    from openh3ic_test.vendor.ref2va_ic import layout, hyperflow, lora
    torch.manual_seed(42)
    model = MiniMaxH3Transformer3DModel(num_attention_heads=4, attention_head_dim=12, hidden_size=32,
                                      num_layers=2, num_refiner_layers=1, ffn_dim=64, freq_dim=8,
                                      time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=2).eval()
    model.set_attention_backend("native")
    base = model.time_embedder
    endpoint = copy.deepcopy(base)
    with torch.no_grad():
        endpoint.linear_2.weight.add_(.05)
    model.time_embedder = hyperflow.TwoTimeEmbedder(base, endpoint, copy.deepcopy(model.time_proj), .7)
    # Exercise unmerged IC factors and a Sharp curve branch as well.
    layer = model.transformer_blocks[0].attn.to_q
    model.transformer_blocks[0].attn.to_q = lora.LoRALinear(layer, torch.randn(3, 32) * .01,
                                                         torch.randn(48, 3) * .01, 1.)
    from openh3ic_test.vendor.ref2va_ic.sharp import CurveContext, CurveLinear
    curve = CurveContext.__new__(CurveContext)
    curve.table, curve.strength, curve.coords = torch.randn(1025, 8), .3, None
    layer = model.transformer_blocks[0].adaln_proj.linear
    model.transformer_blocks[0].adaln_proj.linear = CurveLinear(
        layer, torch.randn(2, 8) * .01, torch.randn(layer.out_features, 2) * .01, curve)
    model.register_forward_pre_hook(curve.pre, with_kwargs=True)
    prompt = {"prompt_embeds": torch.randn(4, 5120), "text_token_tags": torch.tensor([0, 1, 1, 1])}
    ref, audio, prompt_emb, packed, _ = layout.pack(torch.randn(24, 2, 4, 4), [24, 2, 6, 6],
                                                   prompt, torch.randn(2, 32, 3), "cpu")
    times, indices = layout.row_timesteps(packed, .834942)
    model.time_embedder.endpoint_timesteps = hyperflow.endpoints(times, .834942, .696852)
    args = dict(hidden_states=torch.randn(1, len(packed.video_indices), 96), audio_hidden_states=audio,
                encoder_hidden_states=prompt_emb, timestep=times, timestep_indices=indices,
                token_tags=packed.token_tags, position_ids=packed.position_ids,
                video_indices=packed.video_indices, audio_indices=packed.audio_indices,
                text_indices=packed.text_indices, return_dict=False)
    return model, args


@pytest.mark.parametrize("mode", ["online", "sdpa"])
@torch.inference_mode()
def test_streamed_transformer_matches_diffusers_with_ic_and_hyperflow(mode):
    model, args = tiny_case()
    expected = model(**args)
    install_streaming(model, MemorySettings(token_chunk=7, head_chunk=3, key_chunk=11, attention=mode), "cpu")
    actual = model(**args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=8e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("offload", ["cpu", "disk"])
@torch.inference_mode()
def test_cuda_streaming_with_real_accelerate_offload(offload, tmp_path):
    from accelerate import cpu_offload, disk_offload
    model, args = tiny_case()
    device = torch.device("cuda", 0)
    model.to(device)
    expected = tuple(x.cpu() for x in model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in args.items()}))
    model.cpu()
    install_streaming(model, MemorySettings(token_chunk=7, head_chunk=2, attention="online", key_chunk=11), device)
    if offload == "cpu":
        cpu_offload(model, execution_device=device, offload_buffers=True)
    else:
        disk_offload(model, tmp_path, execution_device=device, offload_buffers=True)
    actual = model(**args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, rtol=2e-4, atol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@torch.inference_mode()
def test_cuda_fused_attention_bfloat16():
    torch.manual_seed(22)
    q, k, v = [torch.randn(1, 71, 5, 32, dtype=torch.bfloat16) for _ in range(3)]
    expected = F.scaled_dot_product_attention(*[x.permute(0, 2, 1, 3).cuda() for x in (q, k, v)]).permute(0, 2, 1, 3).cpu()
    actual = full_context_attention(q, k, v, MemorySettings(token_chunk=17, head_chunk=2, attention="sdpa"), torch.device("cuda"))
    torch.testing.assert_close(actual, expected, rtol=.015, atol=.004)
