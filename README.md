# ComfyUI Open-H3-Regenerate-IC

将 [NVLabs Open-H3-Regenerate-2K/IC](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K/IC)
封装成 ComfyUI 节点，并加入显存分块。输入低清草稿视频和原始提示词，通过
**Ref2VA + HyperFlow + LMS Sharp + IC LoRA** 重生成 768p、1080p、2K 或 4K 视频。
原视频保留为参考上下文，原音轨通过节点传递并裁切到输出时长。

这是 **IC 专用实现**，需要 IC 仓库中的 Diffusers 权重目录。旧版非 IC refiner LoRA、
原生 ComfyUI 单文件模型、GGUF 或旧版工作流不能直接替代这些权重。

## 节点

分类：`video / Open H3 IC`，搜索 `Open H3 IC`。

| 节点 | 作用 |
| --- | --- |
| 模型与显存分块 `OpenH3ICLoader` | 配置权重、卸载方式、注意力／前馈／放大／VAE 分块；实际权重按阶段加载 |
| 视频编码 `OpenH3ICEncode` | 24 fps 重采样、裁切、分块 VAE 编码、Qwen3-VL 视频上下文编码 |
| 高清重生成 `OpenH3ICSampler` | `up3`、`up4`、`hf8`；完整参考上下文和 HyperFlow 双时间嵌入 |
| 分块解码 `OpenH3ICDecode` | 分块 VAE 或 fast VAE；输出标准 IMAGE、fps、裁切后的原 AUDIO |

连线：

```text
LoadVideo → GetVideoComponents → 视频编码 → 高清重生成 → 分块解码 → CreateVideo → SaveVideo
                                ↑          ↑           ↑
                                └──── 模型与显存分块 ────┘
```

`GetVideoComponents` 的 fps 接编码节点，audio 接解码节点。VHS 的 IMAGE/AUDIO/fps 也可以接入。
采样输出的 LATENT 包含 IC 的标准化 latent 和裁切元数据，**请使用配套 IC 解码节点**。

## 安装

将本目录放入 `ComfyUI/custom_nodes/ComfyUI-Open-H3-Regenerate-IC`。
使用启动 ComfyUI 的同一个 Python 安装依赖：

```powershell
cd C:\Comfyui
.\.venv\Scripts\python.exe -m pip install -r custom_nodes\ComfyUI-Open-H3-Regenerate-IC\requirements.txt
```

其他安装方式将上面的 Python 路径换成对应环境的解释器。需要 NVIDIA CUDA GPU、
PyTorch ≥ 2.10、包含 MiniMax-H3 modular pipelines 的 Diffusers ≥ 0.40、Transformers ≥ 5.6。
本插件不自动替换 PyTorch/CUDA，也不在导入节点时加载 Diffusers。

下载官方完整 IC 模型目录（**约 150 GB**，以实际仓库为准）：

```powershell
.\.venv\Scripts\python.exe custom_nodes\ComfyUI-Open-H3-Regenerate-IC\scripts\download_models.py --out models\open_h3_ic
```

下载脚本来自上游，会下载基础 Ref2VA transformer、32B 文本编码器、VAE、三个适配器、
潜空间放大模型、静音 latent 和 fast VAE，并提取 Sharp 曲线。支持 Hugging Face 登录状态。
重启 ComfyUI 后，Loader 的 `weights_dir` 填 `open_h3_ic` 或实际绝对路径。
节点运行时从本地加载，不自动下载大模型。

目录示意：

```text
models/open_h3_ic/
  config.json
  base_model/{transformer_ref,vae,text_encoder,tokenizer,processor,...}/
  lora/{open_h3_regenerate_2k_ic_lora,minimax_h3_hyperflow_8step_v1.0,minimax_h3_lms_v1.0_r64}.safetensors
  latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors
  assets/silence_audio_124f.safetensors
  sharp/lms_curve.bin
  vae_fast/fast_h3_vae_v2_24l.safetensors
```

## 显存分块

Transformer 的大序列激活和 Q/K/V 保存在 **CPU 内存**，按 token 块、注意力头分组搬到 GPU。
前馈、AdaLN、残差、输出投影同样分块，不生成完整序列的 GPU 前馈激活。
每个 query 仍访问**全部 reference、text、audio、target keys**，没有切断视频上下文。

| 参数 | 默认 | 调整方式 |
| --- | --- | --- |
| `weight_offload` | `cpu` | 逐层权重卸载；`disk` 进一步节省 RAM；`gpu` 仅适合能容纳全部权重的设备 |
| `token_chunk` | 1024 | 每次投影、前馈和 query 行数；显存紧张降至 512／256 |
| `head_chunk` | 4 | 同时计算的头数；显存紧张降至 2／1，56 个头仍全部计算 |
| `attention` | `sdpa` | 使用融合 SDPA；`online` 用稳定的在线 softmax 遍历全部 K/V，进一步降低峰值但更慢 |
| `key_chunk` | 2048 | `online` 模式每次读取的 key 行数 |
| `vae_tile` / `vae_overlap` | 256 / 64 | 像素空间分块，空间与时间输出在 CPU 拼接；保留原生时间重叠规则 |
| `upscale_frames` / `upscale_overlap` | 5 / 5 | 潜空间放大每块的中心帧数／两侧上下文；frames=0 关闭时间分块 |
| `upscale_tile` | 64 | 放大器输入的 latent 空间块边长，带重叠融合；0 关闭空间分块 |
| `oom_retries` | 2 | Transformer OOM 时缩小 token、head、key 块，改用 online，重试同一未更新的采样步骤 |

`disk` 首次创建权重缓存可能再占用 **100 GB 以上 SSD**；目录可在 `offload_dir` 指定。
CPU 激活依然占用内存，分块以更多 RAM、磁盘和 PCIe 传输换显存，速度会低于多卡原版。
2K／4K 的完整上下文注意力计算量仍然很大，**没有承诺任意显卡都能运行，也没有 24 GB 完整推理实测**。

注意力分块在浮点误差内保持完整计算。**潜空间放大器的时间／空间分块会改变 GroupNorm 的统计区域**，
所以可能轻微改变细节；重叠用于减轻边缘差异。若要使用原版整段放大路线，将
`upscale_frames=0`、`upscale_tile=0`。VAE 的 tile 参数改变也可能影响图像边缘融合。

`standard` 解码默认采用分块完整版 VAE；`fast_auto` 使用上游自动引擎选择，
显存估计不足或实际 OOM 时回退分块 VAE。Blackwell fast 引擎额外依赖 FlashAttention-4、Triton、quack，
没有安装时使用 PyTorch 参考引擎或分块 VAE；Windows 不要求安装这些加速库。

## 示例工作流

- [2K UI 工作流](workflows/OpenH3IC_2K.json)
- [低显存 768p UI 工作流](workflows/OpenH3IC_LowVRAM_768p.json)：disk、256 token、1 head、online attention
- 同目录的 `*_api.json` 用于 ComfyUI API。

把 UI JSON 拖入 ComfyUI，选择 `draft.mp4`，在 caption 中填写**草稿视频生成时的原提示词**，可留空。
节点自动加上官方“增强同一视频、保留场景与时序”的任务描述，不需要手动重复。
编码 seed 和采样 seed 默认均为 0，分别控制 VAE 后验采样和 IC 噪声。

官方输入约束：24 fps 的前 124 帧，至少 **113 个真实帧（约 4.7 秒）**。
短尾重复最后一帧；更长的视频只处理前约 5.17 秒。本插件不将长视频自动拆段。
高分辨率输出使用 1344×768 草稿画布，768p 输出支持 672×384／896×512／1024×576。
1080p 内部计算 1920×1088 后居中裁成 1080；4K 内部计算 3840×2176 后裁成 2160。

| 采样器 | 步数 | 起点 |
| --- | --- | --- |
| `up3` | 3 | 放大草稿，σ≈0.835（默认） |
| `up4` | 4 | 放大草稿，σ≈0.923 |
| `hf8` | 8 | 纯噪声，但保留同样的低清视频参考上下文 |

CFG 固定为 1；音频 token 使用官方静音 latent，成片音轨取自原视频。

## 验证状态

本机已通过 17 项测试：在线 softmax 与完整 SDPA 的数值对照，query/head 分块对照，
真实 Diffusers 小型 H3 Transformer（含 IC LoRA、HyperFlow 双时间嵌入及 Sharp 曲线）对照，
CUDA 下 Accelerate CPU／磁盘卸载、BF16 融合注意力、原生 VAE 时间重叠的 CPU 拼接对照，
帧重采样、布局时间步、放大分块覆盖及工作流接口验证。另已在 ComfyUI 0.22.3 中实际注册全部四个节点。
这些测试不下载训练权重。完整 2K／4K 成片质量、峰值显存和速度尚未实测。

```powershell
$env:USE_TF="0"  # 仅测试进程，避免无关 TensorFlow 安装的导入冲突
python -m pytest tests -q
```

源代码固定于 NVLabs/Sana `1e3c9c27ce56ca03b203cff9f63bed225a1599cd`。
详见 [第三方来源与许可](THIRD_PARTY.md)。本项目非 MiniMax 官方产品。Powered by MiniMax H3.
