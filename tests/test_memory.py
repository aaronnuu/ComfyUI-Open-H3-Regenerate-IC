import pytest
import torch
import torch.nn.functional as F

from openh3ic_test.memory import MemorySettings, full_context_attention, online_attention


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@torch.inference_mode()
def test_cuda_fused_attention_bfloat16():
    torch.manual_seed(22)
    q, k, v = [torch.randn(1, 71, 5, 32, dtype=torch.bfloat16) for _ in range(3)]
    expected = F.scaled_dot_product_attention(*[x.permute(0, 2, 1, 3).cuda() for x in (q, k, v)]).permute(0, 2, 1, 3).cpu()
    actual = full_context_attention(q, k, v, MemorySettings(token_chunk=17, head_chunk=2, attention="sdpa"), torch.device("cuda"))
    torch.testing.assert_close(actual, expected, rtol=.015, atol=.004)
