"""Validate the complete shipped graphs through ComfyUI's actual prompt validator."""
import asyncio
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("COMFYUI_PATH"), reason="Set COMFYUI_PATH")


def test_native_workflows_validate_with_core(monkeypatch):
    import execution
    import folder_paths
    import nodes
    import openh3ic_test

    async def check():
        core = Path(os.environ["COMFYUI_PATH"])
        assert await nodes.load_custom_node(str(core / "comfy_extras/nodes_video.py"))
        nodes.NODE_CLASS_MAPPINGS.update(openh3ic_test.NODE_CLASS_MAPPINGS)
        get_files = folder_paths.get_filename_list
        expected = {
            "diffusion_models": ["minimax_h3_ref2va_int8_convrot.safetensors"],
            "text_encoders": ["qwen3vl_32b_minimax_h3_int8_convrot.safetensors"],
            "vae": ["minimax_h3_video_vae_fp16.safetensors"],
            "loras": ["open_h3_regenerate_2k_ic_lora.safetensors", "minimax_h3_hyperflow_8step_v1.0.safetensors", "minimax_h3_lms_v1.0_r64.safetensors"],
            "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"],
        }
        # Supply a file catalogue in memory; no large weights or video are loaded.
        monkeypatch.setattr(folder_paths, "get_filename_list", lambda category: list(dict.fromkeys(get_files(category) + expected.get(category, []))))
        load_video = nodes.NODE_CLASS_MAPPINGS["LoadVideo"]
        monkeypatch.setattr(load_video, "validate_inputs", classmethod(lambda cls, file: True))
        original_inputs = load_video.INPUT_TYPES
        def video_inputs(cls):
            schema = original_inputs()
            schema["required"]["file"] = ("COMBO", {"options": ["draft.mp4"]})
            return schema
        monkeypatch.setattr(load_video, "INPUT_TYPES", classmethod(video_inputs))
        for path in (Path(__file__).parents[1] / "workflows").glob("*_api.json"):
            graph = json.loads(path.read_text(encoding="utf-8"))
            result = await execution.validate_prompt("ic-test", graph, None)
            if not result[0]:
                print(json.dumps(result[3], indent=2))
            assert result[0], (path.name, result)
    asyncio.run(check())
