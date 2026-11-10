"""Download original official LoRAs and small assets into a native ComfyUI installation."""
import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

REPO = "Efficient-Large-Model/Open-H3-Regenerate-2K-IC"
REVISION = "54a7104b204fb46486cf85c2f600942118186d57"
PLUGIN = Path(__file__).resolve().parents[1]
FILES = {
    "lora/open_h3_regenerate_2k_ic_lora.safetensors": "loras",
    "lora/minimax_h3_hyperflow_8step_v1.0.safetensors": "loras",
    "lora/minimax_h3_lms_v1.0_r64.safetensors": "loras",
    "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors": "latent_upscale_models",
}
CURVE = dict(repo='Comfy-Org/MiniMax-H3', revision='e5eb578a89295337b8ff433a035929ce0279e0b6',
             file='diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors', tensor='adaln_t_table',
             sha256='c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7')


def read_range(url, start, length, token):
    headers = {'Range': f'bytes={start}-{start + length - 1}', 'Accept-Encoding': 'identity'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        if response.status != 206:
            raise RuntimeError('server ignored the Range request; refusing a full checkpoint download')
        data = response.read()
    if len(data) != length:
        raise RuntimeError('truncated range response')
    return data


def fetch_curve(out):
    """Only the adaln_t_table tensor of the checkpoint, by byte range (safetensors header, then its bytes)."""
    from huggingface_hub import get_token, hf_hub_url
    url, token = hf_hub_url(CURVE['repo'], CURVE['file'], revision=CURVE['revision']), get_token()
    header_length, = struct.unpack('<Q', read_range(url, 0, 8, token))
    tensor = json.loads(read_range(url, 8, header_length, token))[CURVE['tensor']]
    assert tensor['dtype'] == 'F32' and tensor['shape'] == [1025, 8], tensor
    start, end = tensor['data_offsets']
    data = read_range(url, 8 + header_length + start, end - start, token)
    if hashlib.sha256(data).hexdigest() != CURVE['sha256']:
        raise ValueError('adaln_t_table SHA256 mismatch')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)


def main():
    from huggingface_hub import hf_hub_download
    import shutil
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comfy-root", type=Path, default=PLUGIN.parents[1])
    ap.add_argument("--assets-only", action="store_true", help="Only silence (~26 KB) and Sharp curve (32 KB)")
    args = ap.parse_args()
    asset_dir = PLUGIN / "assets"
    asset_dir.mkdir(exist_ok=True)
    silence = hf_hub_download(REPO, "assets/silence_audio_124f.safetensors", revision=REVISION)
    shutil.copyfile(silence, asset_dir / "silence_audio_124f.safetensors")
    curve = asset_dir / "lms_curve.bin"
    if not curve.is_file() or hashlib.sha256(curve.read_bytes()).hexdigest() != CURVE["sha256"]:
        fetch_curve(curve)
    if not args.assets_only:
        for name, category in FILES.items():
            dest = args.comfy_root / "models" / category
            if (dest / Path(name).name).is_file():
                print(f"Keeping existing {dest / Path(name).name}")
                continue
            # local_dir preserves downloads for resume; only move the completed file.
            fetched = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=dest / ".open_h3_ic_download"))
            dest.mkdir(parents=True, exist_ok=True)
            fetched.replace(dest / fetched.name)
    print("Official IC assets ready. Base MODEL / CLIP / VAE use standard ComfyUI loaders.")


if __name__ == "__main__":
    main()
