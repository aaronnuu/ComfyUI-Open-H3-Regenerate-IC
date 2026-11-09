from pathlib import Path

import pytest
import torch

from openh3ic_test.runtime import prepare_frames, resolution_spec, sigma_plan
from openh3ic_test.memory import MemorySettings
from openh3ic_test.upscale import tiled_pass
from openh3ic_test.vendor.ref2va_ic import layout


def test_patch_roundtrip_and_row_timesteps():
    reference = torch.randn(24, 2, 4, 4)
    torch.testing.assert_close(layout.unpatch_rows(layout.patch_rows(reference), list(reference.shape)), reference)
    prompt = {"prompt_embeds": torch.randn(4, 5120), "text_token_tags": torch.tensor([0, 1, 1, 1])}
    _, _, _, packed, count = layout.pack(reference, [24, 2, 6, 6], prompt, torch.zeros(2, 32, 3), "cpu")
    times, indices = layout.row_timesteps(packed, .834942)
    rows = times[indices]
    torch.testing.assert_close(rows[packed.video_indices[:count]], torch.full((count,), .999))
    torch.testing.assert_close(rows[packed.video_indices[count:]], torch.full((18,), 1 - layout.canonical_sigma(.834942)))
    torch.testing.assert_close(rows[packed.audio_indices], torch.ones(6))


def test_frame_resampling_padding_and_short_video_rejection():
    images = torch.arange(120).float().view(120, 1, 1, 1).expand(120, 4, 6, 3) / 120
    pixels, real = prepare_frames(images, 24., 4, 6)
    assert real == 120 and pixels.shape == (124, 3, 4, 6)
    torch.testing.assert_close(pixels[-1], pixels[119])
    with pytest.raises(ValueError, match="at least 113"):
        prepare_frames(images[:100], 24., 4, 6)
    _, real = prepare_frames(images.repeat_interleave(2, dim=0), 48., 4, 6)
    assert real == 120


def test_resolution_crop_geometry():
    assert resolution_spec("4k", "576") == (768, 1344, 2176, 3840, 2160)
    assert resolution_spec("768p", "384") == (384, 672, 768, 1344, 768)
    for resolution in ["1080p", "2k", "4k", "4k-dci"]:
        _, _, h, w, _ = resolution_spec(resolution, "576")
        assert h % 32 == w % 32 == 0


class IdentityUpscaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def _forward_seg(self, x, scale, size):
        return torch.nn.functional.interpolate(x, size=size, mode="nearest")


def test_tiled_upscale_covers_every_pixel_and_frame():
    native = torch.full((24, 7, 10, 14), .25)
    result = tiled_pass(IdentityUpscaler(), native, [24, 7, 20, 28],
                        MemorySettings(upscale_frames=2, upscale_overlap=1), 8, "cpu")
    assert result.shape == (24, 7, 20, 28)
    torch.testing.assert_close(result, torch.full_like(result, .25), atol=5e-7, rtol=5e-7)


def test_node_registration_is_lazy():
    import openh3ic_test
    assert len(openh3ic_test.NODE_CLASS_MAPPINGS) == 4
    for name, cls in openh3ic_test.NODE_CLASS_MAPPINGS.items():
        assert cls.INPUT_TYPES()["required"]
        assert callable(getattr(cls, cls.FUNCTION))
