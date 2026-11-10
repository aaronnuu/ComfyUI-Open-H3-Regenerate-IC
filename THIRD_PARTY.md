# Sources and licenses

This plugin is an independent ComfyUI integration, not an official NVIDIA or MiniMax product. Powered by MiniMax H3.

- The IC preprocessing, prompt, sampling recipe, Sharp curve arithmetic, and asset downloader follow
  [NVLabs/Sana IC](https://github.com/NVlabs/Sana/tree/1e3c9c27ce56ca03b203cff9f63bed225a1599cd/models/Open-H3-Regenerate-2K/IC),
  commit `1e3c9c27ce56ca03b203cff9f63bed225a1599cd`. The original documentation remains in `vendor/UPSTREAM_README.md`.
  v0.2 replaces the vendored Diffusers runtime with a native ComfyUI integration.
- `vendor/upscaler_model.py` extracts the model definitions, normalization constants and architecture detection
  from LBH-123-AI's 3D latent upscaler, included in the IC release. MIT notice: `vendor/LICENSE-upscaler-node.txt`.
  The plugin's chunked upscaler wrapper preserves the author's normalization and two-pass route above 2K.
- `vendor/hyperflow/` is adapted from
  [Addis-Pulse-Studio/ComfyUI-HyperFlow](https://github.com/Addis-Pulse-Studio/ComfyUI-HyperFlow),
  commit `a738d182f124180da7ed6167de3cdac448fc5639`, Copyright 2026 Addis Pulse Studio, Apache-2.0.
  Its upstream attribution is preserved in `vendor/HYPERFLOW_NOTICE.txt` and source headers.
  Local changes: optional endpoint validation in the key converter for the official IC adapter; an IC-specific
  non-pruned Ref2VA error message; skip step-zero time nudging when every audio row is pinned (the IC recipe).
  This code includes HyperFlow's schedule/header/two-time conditioning arithmetic
  from Video-Rebirth/hyperflow, after AnyFlow. See the notice for details.
- Native mapping and Sharp integration were cross-checked against the Apache-2.0 implementations
  [ThomasUK888/ComfyUI-OpenH3-Regenerate-IC](https://github.com/ThomasUK888/ComfyUI-OpenH3-Regenerate-IC)
  (`ff2442dcd0040e2f1b0f9414d8ab68bd55714b02`) and
  [rdh9616/ComfyUI-Open-H3-Regenerate](https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate).
  `tests/support.py` adapts the small-model fixtures from ComfyUI-HyperFlow under the same Apache-2.0 license.

No ComfyUI core code is copied into this repository; its native models and APIs are imported at runtime.
No weights or model-derived binary assets are included. The optional downloader obtains the three original LoRAs,
the author upscaler and the encoded silence from
[Efficient-Large-Model/Open-H3-Regenerate-2K-IC](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC),
revision `54a7104b204fb46486cf85c2f600942118186d57`. It reads only the 32 KB `adaln_t_table` tensor of
Comfy-Org/MiniMax-H3 revision `e5eb578a89295337b8ff433a035929ce0279e0b6`, checking its SHA256.

The weights and model-derived assets follow the **MiniMax H3 Community License**, not this code's Apache-2.0
license. Read the [model LICENSE and NOTICE](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/tree/main).
