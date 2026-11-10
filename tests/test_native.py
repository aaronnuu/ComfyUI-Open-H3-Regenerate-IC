"""Numerical checks on real ComfyUI H3 modules, not mocked inference."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from openh3ic_test.memory import MemorySettings, apply_memory, attention_forward
from openh3ic_test.native.adapters import apply_ic, apply_recipe, apply_sharp, CurveContext, curve_forward
from openh3ic_test.runtime import tiled_vae, silence_latent

CORE = bool(os.environ.get("COMFYUI_PATH"))
needs_core = pytest.mark.skipif(not CORE, reason="Set COMFYUI_PATH to test the native H3 integration")


def recipe_files(tmp_path, inner=128):
    from .support import hyperflow_state_dict, write_hyperflow_file, tiny_model
    hf = hyperflow_state_dict(inner=inner)
    hyperflow = write_hyperflow_file(str(tmp_path), sd={k: v.clone() for k, v in hf.items()})
    ic = {k: v.clone() for k, v in hf.items() if "time_embedder" not in k}
    ic_path = tmp_path / "ic.safetensors"
    save_file(ic, str(ic_path))
    base = tiny_model(num_attention_heads=inner // 128)
    sharp = {}
    gen = torch.Generator().manual_seed(65)
    for name in ("blocks.0.attn.qkv_proj", "final_layer.adaln_proj.linear"):
        w = base.model.state_dict()["diffusion_model." + name + ".weight"]
        sharp[f"diffusion_model.{name}.lora_A.weight"] = .02 * torch.randn(3, w.shape[1], generator=gen)
        sharp[f"diffusion_model.{name}.lora_B.weight"] = .02 * torch.randn(w.shape[0], 3, generator=gen)
    for i, block in enumerate(base.model.diffusion_model.blocks):
        root = f"diffusion_model.blocks.{i}.adaln_proj.linear"
        sharp[root + ".lora_A.weight"] = .02 * torch.randn(3, 8, generator=gen)
        sharp[root + ".lora_B.weight"] = .02 * torch.randn(block.adaln_proj.linear.out_features, 3, generator=gen)
    sharp_path = tmp_path / "sharp.safetensors"
    save_file(sharp, str(sharp_path))
    return base, (ic_path, hyperflow, sharp_path), ic


@needs_core
@torch.inference_mode()
def test_official_ic_mapping_uses_attention_inner_width_and_swaps_swiglu(tmp_path):
    base, paths, sd = recipe_files(tmp_path, inner=256)  # deliberately != hidden_size 128
    before = {k: v.clone() for k, v in base.model.state_dict().items()}
    out = apply_ic(base, paths[0], dtype=torch.float32)
    out.patch_model(torch.device("cpu"))
    dm = out.model.diffusion_model
    for prefix, native in (("transformer_blocks.0", "blocks.0"), ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0")):
        def delta(leaf):
            root = "transformer." + prefix + "." + leaf
            return sd[root + ".lora_B.weight"] @ sd[root + ".lora_A.weight"]
        root = "diffusion_model." + native
        expected = before[root + ".attn.qkv_proj.weight"] + torch.cat([delta("attn." + q) for q in ("to_q", "to_k", "to_v")])
        torch.testing.assert_close(dm.get_submodule(native).attn.qkv_proj.weight, expected)
        value, gate = delta("ff.net.0.proj").chunk(2, 0)
        torch.testing.assert_close(dm.get_submodule(native).mlp.fc1.weight,
            before[root + ".mlp.fc1.weight"] + torch.cat([gate, value]))
    out.unpatch_model()
    for k, v in base.model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


@needs_core
@pytest.mark.parametrize("mode", ["sdpa", "online"])
@pytest.mark.parametrize("sampler", ["up3", "hf8"])
@torch.inference_mode()
def test_composed_hyperflow_sharp_ic_full_forward_matches_chunks(tmp_path, mode, sampler):
    base, paths, _ = recipe_files(tmp_path)
    # Use the shipped, hash-checked curve; download script also has CI coverage.
    patched, sigmas = apply_recipe(base, *paths, dtype=torch.float32, sampler=sampler)
    chunked = apply_memory(patched, MemorySettings(token_chunk=7, head_chunk=1, key_chunk=11, attention=mode))
    dm = base.model.diffusion_model
    video, audio, ctx = torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3), torch.randn(1, 5, 48)
    ref = torch.randn_like(video)
    payload = {"seed": 7, "refs": [{"kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 4, "ref_audio_t": 0}],
               "cond_video_latents": [ref], "text_token_tags": torch.tensor([0, 0, 1, 1, 1])}
    def run(m, sigma):
        m.patch_model(torch.device("cpu"))
        times = []
        hook = dm.time_embedder.register_forward_pre_hook(lambda mod, args: times.append(args[0].clone()))
        try:
            result = [v.clone() for v in dm([video, audio], sigma.reshape(1) * 1000, ctx,
                minimax_payload=payload, audio_denoise_mask=torch.zeros_like(audio),
                transformer_options={"wrappers": m.wrappers, "sample_sigmas": sigmas})]
            if float(sigma) == 1.0:
                assert float(times[0].min()) == 0.0
            return result
        finally:
            hook.remove()
            m.unpatch_model()
    for sigma in sigmas[:-1]:
        want = run(patched, sigma)
        got = run(chunked, sigma)
        for g, w in zip(got, want):
            torch.testing.assert_close(g, w, rtol=2e-4, atol=3e-5)
    assert not hasattr(base.model.diffusion_model.blocks[0].attn.forward, "_ic_memory")


@needs_core
@torch.inference_mode()
def test_native_sampler_preserves_silence_and_reference(tmp_path):
    from comfy.nested_tensor import NestedTensor
    from openh3ic_test.runtime import sample
    base, paths, _ = recipe_files(tmp_path)
    model, sigmas = apply_recipe(base, *paths, dtype=torch.float32)
    model = apply_memory(model, MemorySettings(token_chunk=7, head_chunk=1, attention="online", key_chunk=11))
    video, audio = torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)
    ref = torch.randn_like(video)
    original = ref.clone()
    positive = [[torch.randn(1, 5, 48), {"minimax_token_tags": torch.tensor([0, 0, 1, 1, 1]),
        "minimax_refs": [{"kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 4,
                          "ref_audio_t": 0, "latent": ref}]}]]
    latent = {"samples": NestedTensor((video, audio)), "noise_mask": NestedTensor((torch.ones_like(video), torch.zeros_like(audio))),
              "open_h3_ic": {"real_frames": 5}}
    out = sample(model, positive, latent, sigmas, 42, 0)
    v, a = out["samples"].unbind()
    assert v.shape == video.shape and torch.isfinite(v).all()
    torch.testing.assert_close(a, audio)
    torch.testing.assert_close(ref, original, rtol=0, atol=0)


@needs_core
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_native_attention_projection_rope_chunks(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from .support import tiny_model
    model = tiny_model(num_attention_heads=2)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    dm = model.model.diffusion_model.to(device=device, dtype=dtype)
    attn = dm.blocks[0].attn
    x = torch.randn(43, 128, device=device, dtype=dtype)
    from comfy.ldm.minimax.model import rope_rotation_table
    rope = rope_rotation_table(dm.rope_freqs(torch.randn(43, 3), x.device), x.dtype)
    want = attn(x, rope_freqs=rope)
    got = attention_forward(attn, MemorySettings(token_chunk=7, head_chunk=1, key_chunk=11, attention="online"))(x, rope)
    torch.testing.assert_close(got, want, rtol=.015 if device == "cuda" else 2e-4,
                               atol=.006 if device == "cuda" else 2e-5)


def test_sharp_curve_uses_recorded_t_not_hyperflow_embedding():
    table = torch.randn(1025, 8)
    context = CurveContext(table)
    layer = torch.nn.Linear(9, 12)
    weight = torch.randn(12, 8)
    wrapped = curve_forward(layer, weight, context, .7)
    t = torch.tensor([.125, .75, 1.])
    embed = context.record(lambda t: torch.randn(len(t), 9))(t)
    coords = table[torch.tensor([128, 768, 1024])]
    torch.testing.assert_close(wrapped(embed), layer(embed) + .7 * coords @ weight.t())


def test_vae_semantic_tiles_restore_on_failure():
    native = SimpleNamespace(clip_length=17, latents_mean=torch.zeros(24), tiling=False, tile_size=512, tile_overlap_min=128)
    vae = SimpleNamespace(first_stage_model=native)
    with pytest.raises(RuntimeError):
        with tiled_vae(vae, 256, 64):
            assert (native.tiling, native.tile_size, native.tile_overlap_min) == (True, 256, 64)
            raise RuntimeError("interrupted")
    assert (native.tiling, native.tile_size, native.tile_overlap_min) == (False, 512, 128)


@needs_core
def test_pruned_model_rejected_before_reading_large_loras():
    from .support import tiny_model
    model = tiny_model(adaln_curve_grid=1025, time_embed_dim=8)
    with pytest.raises(ValueError, match="pruned"):
        apply_recipe(model, "absent", "absent", "absent")


def test_official_silence_asset_layout():
    if not (Path(__file__).parents[1] / "assets/silence_audio_124f.safetensors").is_file():
        pytest.skip("Run download_models.py --assets-only")
    assert silence_latent().shape == (1, 32, 2, 207)


@needs_core
@pytest.mark.parametrize("sampler_name", ["up3", "hf8"])
def test_oom_restarts_from_identical_noise_with_reduced_chunks(tmp_path, monkeypatch, sampler_name):
    import comfy.sample
    from comfy.nested_tensor import NestedTensor
    from openh3ic_test.runtime import sample
    base, paths, _ = recipe_files(tmp_path)
    model, sigmas = apply_recipe(base, *paths, dtype=torch.float32, sampler=sampler_name)
    model = apply_memory(model, MemorySettings(token_chunk=512, head_chunk=4, key_chunk=1024))
    video, audio = torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)
    latent = {"samples": NestedTensor((video, audio)), "noise_mask": NestedTensor((torch.ones_like(video), torch.zeros_like(audio))), "open_h3_ic": {}}
    calls = []
    def fake(m, noise, cfg, sampler, grid, pos, neg, source, **kwargs):
        calls.append((m.model_options["open_h3_ic_memory"], noise.unbind()[0].clone(), source.unbind()[0].clone()))
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic CUDA OOM")
        return source
    monkeypatch.setattr(comfy.sample, "sample_custom", fake)
    out = sample(model, [[torch.zeros(1), {"minimax_refs": []}]], latent, sigmas, 72, 1)
    assert calls[1][0].token_chunk == 256 and calls[1][0].head_chunk == 2
    assert calls[1][0].attention == "online"
    torch.testing.assert_close(calls[0][1], calls[1][1], rtol=0, atol=0)
    torch.testing.assert_close(calls[0][2], calls[1][2], rtol=0, atol=0)
    torch.testing.assert_close(calls[0][2], torch.zeros_like(video) if sampler_name == "hf8" else video)
    assert out["open_h3_ic"] == latent["open_h3_ic"]
