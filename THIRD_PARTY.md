# Sources and licenses

The `vendor/ref2va_ic/` inference components and `scripts/download_models.py`
come from [NVLabs/Sana, sol-engine, models/Open-H3-Regenerate-2K/IC](https://github.com/NVlabs/Sana/tree/1e3c9c27ce56ca03b203cff9f63bed225a1599cd/models/Open-H3-Regenerate-2K/IC),
commit `1e3c9c27ce56ca03b203cff9f63bed225a1599cd` (2026-11-09 retrieval).
The original README is included in `vendor/UPSTREAM_README.md`.

The upstream weights follow the **MiniMax H3 Community License**; see the
[weights LICENSE and NOTICE](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/tree/main).
This plugin does not include weights. Open-H3-Regenerate-2K-IC is not an official
MiniMax product. Powered by MiniMax H3.

The author latent upscaler, `vendor/ref2va_ic/upscaler_node.py`, is MIT;
its original notice is included in `vendor/ref2va_ic/LICENSE-upscaler-node.txt`.
`vendor/upscaler_model.py` extracts only its unchanged neural-layer definitions,
normalization constants and architecture detection, without ComfyUI UI imports.
Its standalone `upscale.py` shim is not used by this plugin, because replacing
`sys.modules['folder_paths']` would corrupt a running ComfyUI process.

The streaming transformer follows Diffusers' Apache-2.0
`MiniMaxH3Transformer3DModel` arithmetic. HyperFlow and LMS Sharp attribution
and their source links appear in the included upstream README and source files.
