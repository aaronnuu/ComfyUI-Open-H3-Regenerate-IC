import pytest
import torch

from openh3ic_test.runtime import install_vae_cpu_stitching


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@torch.inference_mode()
def test_released_vae_cpu_stitching_preserves_temporal_overlap():
    from diffusers import AutoencoderKLMiniMaxH3
    torch.manual_seed(30)
    vae = AutoencoderKLMiniMaxH3(block_out_channels=(32, 32), layers_per_block=1,
                                spatial_downsample_factors=(2, 2), temporal_downsample_factors=(2, 2),
                                decoder_num_layers=1, decoder_num_attention_heads=2, decoder_attention_head_dim=24,
                                decoder_num_register_tokens=2).cuda().eval()
    vae.enable_tiling(tile_sample_min_height=32, tile_sample_min_width=32,
                      tile_sample_min_overlap_height=8, tile_sample_min_overlap_width=8)
    pixels = torch.randn(1, 3, 39, 40, 48)
    latent = torch.randn(1, 24, 12, 10, 12)
    expected_encode = vae.encode(pixels.cuda()).latent_dist.mean.cpu()
    expected_decode = vae.decode(latent.cuda()).sample.cpu()
    install_vae_cpu_stitching(vae, torch.device("cuda"), check=lambda: None)
    actual_encode = vae.encode(pixels).latent_dist.mean
    actual_decode = vae.decode(latent).sample
    assert actual_encode.device.type == actual_decode.device.type == "cpu"
    torch.testing.assert_close(actual_encode, expected_encode, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_decode, expected_decode, rtol=1e-5, atol=1e-5)
