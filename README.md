# ComfyUI Open-H3 Regenerate IC

**v0.2：使用 ComfyUI 原生 MODEL / CLIP / VAE，直接加载官方原版 LoRA，并保留显存分块。**

把草稿视频作为 Ref2VA 的 `<Video 1>`，使用 Open-H3-Regenerate-2K-IC 增强至 768p、1080p、2K 或 4K。
不再要求旧版的 `weights_dir`、Diffusers 模型目录或额外的独立推理管线。

- [官方 IC 模型与三个 LoRA](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC/tree/main/lora)
- [NVIDIA 参考代码](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K/IC)
- [本插件仓库](https://github.com/kkget/ComfyUI-Open-H3-Regenerate-IC)

## 安装

需要支持原生 MiniMax H3 的新版 ComfyUI；已在 **ComfyUI 0.44.0** 验证。旧版找不到 H3 或 `minimax` CLIP 类型时先升级 ComfyUI。

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kkget/ComfyUI-Open-H3-Regenerate-IC.git
cd ComfyUI-Open-H3-Regenerate-IC
python -m pip install -r requirements.txt
python scripts/download_models.py --assets-only
```

请使用启动 ComfyUI 的同一个 Python；便携版用其 `python_embeded/python.exe`。
依赖使用 ComfyUI 自带的 PyTorch、safetensors、einops；额外依赖只有下载脚本使用的 huggingface_hub。
`--assets-only` 仅下载官方静音潜变量和 Sharp 参考曲线，合计约 60 KB。它们保存在插件 `assets/` 中，不需要音频 VAE。

重启 ComfyUI，在 `video/Open H3 IC` 分类找到四个节点。

## 模型放置

LoRA 仍需要基础 MODEL、CLIP 和 VAE；这里使用标准加载器支持的单文件版本，可以复用已有模型。

| 文件 | 位置 / 加载方式 |
| --- | --- |
| `minimax_h3_ref2va_int8_convrot.safetensors` 或 `minimax_h3_ref2va_bf16.safetensors` | `models/diffusion_models/`，Load Diffusion Model |
| H3 专用 Qwen3-VL-32B，如 `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `models/text_encoders/`，Load CLIP，**type 选择 `minimax`** |
| `minimax_h3_video_vae_fp16.safetensors` 等标准 H3 视频 VAE | `models/vae/`，Load VAE |
| `open_h3_regenerate_2k_ic_lora.safetensors` | `models/loras/`，来自官方 IC 仓库 `lora/` |
| `minimax_h3_hyperflow_8step_v1.0.safetensors` | `models/loras/`，同上 |
| `minimax_h3_lms_v1.0_r64.safetensors` | `models/loras/`，同上 |
| `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | `models/latent_upscale_models/`，官方 IC 仓库 `latent_upsampler/` |

[基础模型、文本编码器与 VAE 的单文件下载](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main)。

**基础模型必须是非 pruned 的 Ref2VA。** HyperFlow 需要原始 `time_embedder` 和端点时间嵌入；`*_pruned_*` 的曲线压缩模型删掉了该层，节点会明确报错。不要选择 FL2VA、Turbo 合并模型或普通 T2V 模型代替。

使用上表的三个官方 LoRA 原文件，**无需预转换或重命名**。已转换的 `*_comfy*` IC/HyperFlow 文件不适用。
LoRA 加载器会处理 Q/K/V 行切片、SwiGLU 半行顺序、HyperFlow 双时间嵌入和 LMS Sharp 的 50 个曲线分支；无需另外安装 HyperFlow 插件。

自动下载三个 LoRA、作者 3D 放大模型和小资产（约 12 GB；不下载基础 MODEL / CLIP / VAE）：

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI
```

Windows 例子：`python scripts/download_models.py --comfy-root C:\Comfyui`。手动下载时按表放置，再运行一次 `--assets-only`。
脚本固定官方 IC 仓库版本 `54a7104b204fb46486cf85c2f600942118186d57`，Sharp 曲线有 SHA256 校验。已有同名模型文件会保留。

## 工作流

直接拖入以下 UI 工作流，选择输入视频与本机模型文件：

- [2K 示例](workflows/OpenH3IC_2K.json)
- [低显存 768p 示例](workflows/OpenH3IC_LowVRAM_768p.json)

同目录 `*_api.json` 用于 API。旧版独立 Diffusers 管线的工作流需要替换成新版示例。

连接方式：

```text
Load Diffusion Model → IC Official LoRA Loader → MODEL + SIGMAS → IC Sample
CLIP + VAE + 输入视频 → IC Encode Draft → CONDITIONING + LATENT → IC Sample
IC Sample → LATENT → IC Tiled Decode（另接 VAE）→ IMAGE + fps + AUDIO
                                                       ↓
                                              Create Video → Save Video
```

1. **Official LoRA Loader**：基础 MODEL 接入，选择三个官方文件；默认 `up3`、Sharp 1.0。输出 MODEL 与 SIGMAS。
2. **Encode Draft**：接 H3 CLIP、视频 VAE、输入帧及帧率。caption 建议填草稿原始提示词；节点自动加上官方增强指令和草稿视觉上下文。可接输入音频。
3. **Sample**：接以上输出；固定 Euler、CFG 1，使用官方 HyperFlow 时间表。
4. **Tiled Decode**：接视频 VAE 和采样结果，输出帧、24 fps 与输入音轨；无输入音轨时输出静音。

输入按 24 fps 取前 124 帧，至少需要 113 个实际帧（约 4.7 秒）。末尾不足 124 帧时重复最后一帧，输出会裁回实际长度。
较长输入只处理前约 5.2 秒。草稿采用官方横向画布并居中裁剪；1080p 在 1088 行计算后裁切，4K 在 2176 行计算后裁切。
`hf8` 使用纯噪声初始化；`up3` / `up4` 使用作者 3D 潜空间放大结果，再从官方时间表的最后 3 / 4 步去噪。

## 显存分块

| 选项 | 作用 | 默认 / 更省显存的起点 |
| --- | --- | --- |
| `token_chunk` | QKV、输出投影和 FFN 每批 token 数 | 1024 / 512 或 256 |
| `head_chunk` | 同时计算的注意力头数 | 4 / 1 |
| `attention` | `sdpa` 融合注意力；`online` 将 K/V 也分批传入 GPU | sdpa / online |
| `key_chunk` | online 模式每批 key 数 | 2048 / 1024 |
| `vae_tile` / `vae_overlap` | 原生 H3 VAE 实际使用的像素分块与重叠 | 256/64 / 192/48 |
| `upscale_frames` / `upscale_overlap` | 3D 放大每批潜帧与前后上下文 | 5/5 |
| `upscale_tile` | 3D 放大空间分块，单位为输入潜像素；0 关闭 | 64 / 32 |
| `oom_retries` | 采样 CUDA OOM 后缩小分块、切换 online，用同一噪声重新采样 | 2 |

**注意力始终覆盖全部上下文**，online 使用 FP32 稳定 softmax 累积，不是局部注意力或对各块结果简单平均。
Q/K/V 暂存系统内存；基础模型权重继续由 ComfyUI 原生加载、量化与卸载机制管理。所有模型 forward 修改通过 ModelPatcher 克隆安装和恢复，不修改 ComfyUI 核心。

分块会降低速度并增加内存用量，2K/4K 仍可能超出显存或系统内存；建议先跑低显存 768p 示例，并使用量化基础模型和 H3 文本编码器。
OOM 重试只涵盖扩散采样；编码/解码或放大阶段 OOM 时，请减小对应的 tile。完整残差流和当前层权重仍需占用显存，不能承诺任意显卡都能运行 2K。

为确保 VAE 分块有效，请用**标准 H3 视频 VAE**。Sol 非因果整段解码器不能分块，新节点会提示换用标准 VAE。
原生 VAE 使用 ComfyUI 的 posterior mean 编码；官方独立脚本从 posterior 采样。潜空间放大空间分块也会稍微改变 GroupNorm 统计，因此不承诺逐位复现独立脚本。
`lora_dtype=bf16` 只将 IC/Sharp 因子在 RAM 中转为 BF16，原文件保持不变；若要保留原始 FP32 因子，选 `fp32`，会使用更多系统内存。

## 验证

```bash
COMFYUI_PATH=/path/to/ComfyUI IC_TEST_CPU=1 python -m pytest tests -q
```

Windows PowerShell 先设置 `$env:COMFYUI_PATH='C:\path\to\ComfyUI'` 与 `$env:IC_TEST_CPU='1'`。
测试覆盖真实原生 H3 模块的 IC 映射、三 LoRA 组合前向、分块数值一致性、原生 Euler 采样静音固定、OOM 重启、VAE 参数恢复及 ComfyUI 工作流校验。CUDA 测试在有 GPU 时运行。
CI 固定经验证的 ComfyUI 核心提交；测试使用小模型，不下载完整基础权重。当前未完成完整官方权重的 2K 视频端到端生成测试。

代码许可证及来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。模型与从模型提取的小资产遵循各自的模型许可，不随插件代码打包。Powered by MiniMax H3。
