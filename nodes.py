"""Legacy node schema for broad ComfyUI compatibility; no Diffusers startup import."""
from pathlib import Path

CATEGORY = "video/Open H3 IC"


def _progress():
    from comfy.utils import ProgressBar
    bar = ProgressBar(1000)
    return lambda current, total: bar.update_absolute(round(current / max(1, total) * 1000), 1000)


class OpenH3ICLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "weights_dir": ("STRING", {"default": "open_h3_ic", "tooltip": "Relative to ComfyUI/models, or an absolute IC weights directory."}),
            "weight_offload": (["cpu", "disk", "gpu"], {"default": "cpu", "tooltip": "cpu: sequential weight offload; disk: less RAM, more disk traffic; gpu: all weights resident."}),
            "token_chunk": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64, "tooltip": "Rows per projection, FFN and attention query chunk. Smaller saves VRAM."}),
            "head_chunk": ("INT", {"default": 4, "min": 1, "max": 56, "tooltip": "Attention heads loaded together. Smaller saves VRAM; all 56 heads still run."}),
            "attention": (["sdpa", "online"], {"default": "sdpa", "tooltip": "sdpa uses fused full-context attention. online streams all K/V in bounded chunks with FP32 softmax; slower."}),
            "key_chunk": ("INT", {"default": 2048, "min": 128, "max": 16384, "step": 128, "tooltip": "Keys per chunk in online attention."}),
            "vae_tile": ("INT", {"default": 256, "min": 128, "max": 1024, "step": 32, "tooltip": "VAE spatial tile in pixels. Released temporal overlap is preserved."}),
            "vae_overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
            "upscale_frames": ("INT", {"default": 5, "min": 0, "max": 64, "tooltip": "Latent frames per upscaler core chunk; 0 processes the full time axis."}),
            "upscale_overlap": ("INT", {"default": 5, "min": 0, "max": 32, "tooltip": "Extra latent frames on each side of an upscaler chunk."}),
            "upscale_tile": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8, "tooltip": "Source latent tile size; 0 disables spatial upscaler tiling. Tiling changes GroupNorm statistics slightly."}),
            "oom_retries": ("INT", {"default": 2, "min": 0, "max": 4, "tooltip": "Retry a failed transformer step with smaller chunks and online attention."}),
        }, "optional": {
            "offload_dir": ("STRING", {"default": "", "tooltip": "Disk weight cache. Empty uses ComfyUI/temp/open_h3_ic_offload; may require over 100 GB."}),
        }}

    RETURN_TYPES = ("OPEN_H3_IC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "configure"
    CATEGORY = CATEGORY
    DESCRIPTION = "Open-H3-Regenerate-2K-IC with streamed activations and configurable VRAM chunks. Loads each model only during its stage."

    def configure(self, weights_dir, weight_offload, token_chunk, head_chunk, attention, key_chunk,
                  vae_tile, vae_overlap, upscale_frames, upscale_overlap, upscale_tile, oom_retries,
                  offload_dir=""):
        import folder_paths
        from .memory import MemorySettings
        from .runtime import PipelineConfig
        if vae_overlap >= vae_tile:
            raise ValueError("vae_overlap must be smaller than vae_tile")
        if upscale_tile and upscale_tile < 8:
            raise ValueError("upscale_tile must be 0 or at least 8 latent pixels")
        weights = Path(weights_dir.strip()).expanduser()
        if not weights.is_absolute():
            weights = Path(folder_paths.models_dir) / weights
        cache = Path(offload_dir).expanduser() if offload_dir.strip() else Path(folder_paths.get_temp_directory()) / "open_h3_ic_offload"
        settings = MemorySettings(token_chunk, head_chunk, key_chunk, attention, weight_offload,
                                  vae_tile, vae_overlap, upscale_frames, upscale_overlap, oom_retries)
        return (PipelineConfig(weights.resolve(), settings, cache.resolve(), upscale_tile),)


class OpenH3ICEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipeline": ("OPEN_H3_IC_PIPELINE",),
            "images": ("IMAGE",),
            "input_fps": ("FLOAT", {"default": 24., "min": .1, "max": 240., "step": .01}),
            "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "The original draft's generation prompt. The IC enhancement instruction is added automatically."}),
            "resolution": (["768p", "1080p", "2k", "4k", "4k-dci"], {"default": "2k"}),
            "lq_for_768p": (["384", "512", "576"], {"default": "576", "tooltip": "Draft height when output is 768p; ignored at higher output resolutions."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("OPEN_H3_IC_DRAFT",)
    RETURN_NAMES = ("draft",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Resample the first 124 frames at 24 fps, tiled H3 VAE encode and Qwen3-VL video context encode. Requires at least 113 real frames at 24 fps."

    def encode(self, pipeline, images, input_fps, caption, resolution, lq_for_768p, seed):
        from .runtime import encode_draft, validate_weights
        validate_weights(pipeline, sharp=0, sampler="hf8")
        return (encode_draft(pipeline, images, input_fps, caption, resolution, lq_for_768p, seed, _progress()),)


class OpenH3ICSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipeline": ("OPEN_H3_IC_PIPELINE",),
            "draft": ("OPEN_H3_IC_DRAFT",),
            "sampler": (["up3", "up4", "hf8"], {"default": "up3"}),
            "sharp": ("FLOAT", {"default": 1., "min": 0., "max": 2., "step": .05}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("ic_latent",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = "Ref2VA + HyperFlow + LMS Sharp + IC LoRA. All reference tokens remain in context. Use Open H3 IC Decode for the normalized latent."

    def sample(self, pipeline, draft, sampler, sharp, seed):
        from .runtime import sample
        if str(pipeline.weights) != draft["weights"]:
            raise ValueError("Encode and Sample must use the same IC weights directory")
        return (sample(pipeline, draft, sampler, sharp, seed, _progress()),)


class OpenH3ICDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipeline": ("OPEN_H3_IC_PIPELINE",),
            "ic_latent": ("LATENT",),
            "decoder": (["standard", "fast_auto", "fast_ref"], {"default": "standard", "tooltip": "standard uses tiled released VAE. Fast decoder falls back to standard when it cannot fit."}),
        }, "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "fps", "audio")
    FUNCTION = "decode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Decode on spatial/temporal VAE tiles; stitch on CPU. Crops 1088/2176 to 1080/2160. Keeps original audio, trimmed to the output duration."

    def decode(self, pipeline, ic_latent, decoder, audio=None):
        from .runtime import decode
        images, fps = decode(pipeline, ic_latent, decoder, _progress())
        if audio is not None:
            audio = {**audio, "waveform": audio["waveform"][..., :round(len(images) / fps * audio["sample_rate"])]}
        return images, fps, audio


NODE_CLASS_MAPPINGS = {"OpenH3ICLoader": OpenH3ICLoader, "OpenH3ICEncode": OpenH3ICEncode,
                       "OpenH3ICSampler": OpenH3ICSampler, "OpenH3ICDecode": OpenH3ICDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenH3ICLoader": "Open H3 IC · 模型与显存分块",
                              "OpenH3ICEncode": "Open H3 IC · 视频编码",
                              "OpenH3ICSampler": "Open H3 IC · 高清重生成",
                              "OpenH3ICDecode": "Open H3 IC · 分块解码"}
