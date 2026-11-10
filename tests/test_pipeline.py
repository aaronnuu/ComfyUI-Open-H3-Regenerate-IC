import torch
from openh3ic_test.memory import MemorySettings
from openh3ic_test.upscale import tiled_pass

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
