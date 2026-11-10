"""Four native ComfyUI nodes. Model loading/offloading belongs to ComfyUI."""
CATEGORY = "video/Open H3 IC"
IC = "open_h3_regenerate_2k_ic_lora.safetensors"
HF = "minimax_h3_hyperflow_8step_v1.0.safetensors"
SHARP = "minimax_h3_lms_v1.0_r64.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"


def files(category, preferred):
    import folder_paths
    if category == "latent_upscale_models" and category not in folder_paths.folder_names_and_paths:
        import os
        folder_paths.add_model_folder_path(category, os.path.join(folder_paths.models_dir, category))
    names = folder_paths.get_filename_list(category)
    selected = next((n for n in names if n.replace("\\", "/").split("/")[-1] == preferred), preferred)
    return (names if selected in names else [selected] + names, {"default": selected})


class OpenH3ICLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "ic_lora": files("loras", IC),
            "hyperflow_lora": files("loras", HF), "sharp_lora": files("loras", SHARP),
            "sampler": (["up3", "up4", "hf8"], {"default": "up3"}),
            "sharp_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": .05}),
            "lora_dtype": (["bf16", "fp32"], {"default": "bf16", "tooltip": "IC/Sharp factor storage in RAM; bf16 halves the FP32 IC file's RAM footprint."}),
            "token_chunk": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64}),
            "head_chunk": ("INT", {"default": 4, "min": 1, "max": 56}),
            "attention": (["sdpa", "online", "native"], {"default": "sdpa", "tooltip": "All use full context. online streams K/V from RAM; slower but uses less VRAM. native (aaronnuu/zoetrope) keeps ComfyUI's own attention backend with Q/K/V on the GPU: fastest, most VRAM; token_chunk then chunks only the FFN."}),
            "key_chunk": ("INT", {"default": 2048, "min": 128, "max": 16384, "step": 128}),
        }}
    RETURN_TYPES = ("MODEL", "SIGMAS")
    RETURN_NAMES = ("model", "sigmas")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Apply the three original official IC LoRAs to a native non-pruned Ref2VA MODEL. Includes HyperFlow two-time conditioning, Sharp curves and memory chunking."

    def load(self, model, ic_lora, hyperflow_lora, sharp_lora, sampler, sharp_strength, lora_dtype,
             token_chunk, head_chunk, attention, key_chunk):
        import folder_paths
        import torch
        from .native.adapters import apply_recipe
        from .memory import MemorySettings, apply_memory
        paths = [folder_paths.get_full_path_or_raise("loras", name) for name in (ic_lora, hyperflow_lora, sharp_lora)]
        model, sigmas = apply_recipe(model, *paths, sharp_strength=sharp_strength,
            dtype=torch.bfloat16 if lora_dtype == "bf16" else torch.float32, sampler=sampler)
        return apply_memory(model, MemorySettings(token_chunk=token_chunk, head_chunk=head_chunk,
                                                   attention=attention, key_chunk=key_chunk)), sigmas


class OpenH3ICEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "vae": ("VAE",), "images": ("IMAGE",),
            "input_fps": ("FLOAT", {"default": 24.0, "min": .1, "max": 240.0}),
            "caption": ("STRING", {"default": "", "multiline": True}),
            "resolution": (["768p", "1080p", "2k", "4k", "4k-dci"], {"default": "2k"}),
            "lq_for_768p": (["384", "512", "576"], {"default": "576"}),
            "upscaler_model": files("latent_upscale_models", UPSCALER),
            "vae_tile": ("INT", {"default": 256, "min": 128, "max": 1024, "step": 32}),
            "vae_overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 16}),
            "upscale_frames": ("INT", {"default": 5, "min": 1, "max": 37}),
            "upscale_overlap": ("INT", {"default": 5, "min": 0, "max": 16}),
            "upscale_tile": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8, "tooltip": "Source latent pixels; 0 disables spatial tiling. Tiling slightly changes GroupNorm statistics."}),
        }, "optional": {"audio": ("AUDIO",)}}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Tiled native H3 VAE, Qwen3-VL draft context and chunked author latent upscaler. First 124 frames at 24 fps; needs at least 113 real frames."

    def encode(self, clip, vae, images, input_fps, caption, resolution, lq_for_768p, upscaler_model,
               vae_tile, vae_overlap, upscale_frames, upscale_overlap, upscale_tile, audio=None):
        import folder_paths
        from .runtime import encode_draft
        path = folder_paths.get_full_path_or_raise("latent_upscale_models", upscaler_model)
        return encode_draft(clip, vae, images, input_fps, caption, resolution, lq_for_768p, path,
                            vae_tile, vae_overlap, upscale_frames, upscale_overlap, upscale_tile, audio)


class OpenH3ICSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "latent": ("LATENT",), "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "oom_retries": ("INT", {"default": 2, "min": 0, "max": 4}),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = "Native Euler, CFG 1, official HyperFlow grid, silence pinned. CUDA OOM restarts from the same noise with smaller chunks."

    def sample(self, model, positive, latent, sigmas, seed, oom_retries):
        from .runtime import sample
        return (sample(model, positive, latent, sigmas, seed, oom_retries),)


class OpenH3ICDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "latent": ("LATENT",),
            "vae_tile": ("INT", {"default": 256, "min": 128, "max": 1024, "step": 32}),
            "vae_overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 16})}}
    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "fps", "audio")
    FUNCTION = "decode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Tiled stock H3 VAE decode, crop padding, trim to real frames and preserve the input soundtrack."

    def decode(self, vae, latent, vae_tile, vae_overlap):
        from .runtime import decode
        return decode(vae, latent, vae_tile, vae_overlap)


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (OpenH3ICLoader, OpenH3ICEncode, OpenH3ICSampler, OpenH3ICDecode)}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenH3ICLoader": "Open H3 IC · Official LoRA Loader",
    "OpenH3ICEncode": "Open H3 IC · Encode Draft", "OpenH3ICSampler": "Open H3 IC · Sample",
    "OpenH3ICDecode": "Open H3 IC · Tiled Decode"}
