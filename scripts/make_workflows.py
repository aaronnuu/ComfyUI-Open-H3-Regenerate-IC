"""Regenerate the UI graphs and API prompts shipped with the plugin."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def graph(low_memory=False):
    nodes, links = [], []
    def node(identity, kind, pos, inputs, outputs, widgets, size=(340, 220)):
        item = {"id": identity, "type": kind, "pos": pos, "size": list(size), "flags": {},
                "order": identity - 1, "mode": 0,
                "inputs": [{"name": name, "type": typ, "link": None} for name, typ in inputs],
                "outputs": [{"name": name, "type": typ, "links": [], "slot_index": i}
                            for i, (name, typ) in enumerate(outputs)],
                "properties": {"Node name for S&R": kind}, "widgets_values": widgets}
        nodes.append(item)
        return item
    def connect(src, src_slot, dst, dst_slot, widget=None):
        link = len(links) + 1
        typ = src["outputs"][src_slot]["type"]
        links.append([link, src["id"], src_slot, dst["id"], dst_slot, typ])
        src["outputs"][src_slot]["links"].append(link)
        dst["inputs"][dst_slot]["link"] = link
        if widget:
            dst["inputs"][dst_slot]["widget"] = {"name": widget}
    load = node(1, "LoadVideo", [20, 60], [], [("VIDEO", "VIDEO")], ["draft.mp4"], (300, 180))
    split = node(2, "GetVideoComponents", [360, 60], [("video", "VIDEO")],
                 [("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT")], [], (260, 110))
    settings = (["open_h3_ic", "disk", 256, 1, "online", 1024, 192, 48, 3, 5, 48, 2, ""] if low_memory else
                ["open_h3_ic", "cpu", 1024, 4, "sdpa", 2048, 256, 64, 5, 5, 64, 2, ""])
    loader = node(3, "OpenH3ICLoader", [20, 350], [], [("pipeline", "OPEN_H3_IC_PIPELINE")], settings, (380, 500))
    resolution = "768p" if low_memory else "2k"
    enc = node(4, "OpenH3ICEncode", [680, 70], [("pipeline", "OPEN_H3_IC_PIPELINE"), ("images", "IMAGE"), ("input_fps", "FLOAT")],
               [("draft", "OPEN_H3_IC_DRAFT")], [24., "", resolution, "576", 0, "fixed"], (380, 340))
    sampler = node(5, "OpenH3ICSampler", [1110, 70], [("pipeline", "OPEN_H3_IC_PIPELINE"), ("draft", "OPEN_H3_IC_DRAFT")],
                   [("ic_latent", "LATENT")], ["up3", 1., 0, "fixed"], (330, 240))
    dec = node(6, "OpenH3ICDecode", [1490, 70], [("pipeline", "OPEN_H3_IC_PIPELINE"), ("ic_latent", "LATENT"), ("audio", "AUDIO")],
               [("images", "IMAGE"), ("fps", "FLOAT"), ("audio", "AUDIO")], ["standard"], (330, 200))
    video = node(7, "CreateVideo", [1870, 70], [("images", "IMAGE"), ("fps", "FLOAT"), ("audio", "AUDIO")],
                 [("VIDEO", "VIDEO")], [24.], (280, 160))
    save = node(8, "SaveVideo", [2200, 70], [("video", "VIDEO")], [], ["OpenH3IC/refined", "mp4", "h264"], (330, 200))
    note = node(9, "MarkdownNote", [680, 470], [], [], [
        "# Open H3 Regenerate IC\n\n1. Install requirements and download IC Diffusers weights to models/open_h3_ic.\n"
        "2. Select a 4.7–5.2 second draft in LoadVideo (first 124 frames at 24 fps).\n"
        "3. Enter the original draft prompt in caption.\n"
        "4. Keep standard VAE for lower VRAM. Use IC Decode for this normalized latent.\n\n"
        "Low VRAM: lower token_chunk/head_chunk; online streams K/V. Disk offload needs substantial SSD space.\n"
        "Upscaler tiles can slightly change detail/normalization. Full trained-model inference has not been benchmarked."
    ], (750, 300))
    connect(load, 0, split, 0)
    connect(split, 0, enc, 1)
    connect(split, 1, dec, 2)
    connect(split, 2, enc, 2, "input_fps")
    connect(loader, 0, enc, 0)
    connect(loader, 0, sampler, 0)
    connect(loader, 0, dec, 0)
    connect(enc, 0, sampler, 1)
    connect(sampler, 0, dec, 1)
    connect(dec, 0, video, 0)
    connect(dec, 1, video, 1, "fps")
    connect(dec, 2, video, 2)
    connect(video, 0, save, 0)
    ui = {"last_node_id": 9, "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [],
          "config": {}, "extra": {"ds": {"scale": .65, "offset": [50, 80]}}, "version": .4}
    names = ["weights_dir", "weight_offload", "token_chunk", "head_chunk", "attention", "key_chunk", "vae_tile", "vae_overlap",
             "upscale_frames", "upscale_overlap", "upscale_tile", "oom_retries", "offload_dir"]
    api = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": "draft.mp4"}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "OpenH3ICLoader", "inputs": dict(zip(names, settings))},
        "4": {"class_type": "OpenH3ICEncode", "inputs": {"pipeline": ["3", 0], "images": ["2", 0], "input_fps": ["2", 2],
                "caption": "", "resolution": resolution, "lq_for_768p": "576", "seed": 0}},
        "5": {"class_type": "OpenH3ICSampler", "inputs": {"pipeline": ["3", 0], "draft": ["4", 0], "sampler": "up3", "sharp": 1., "seed": 0}},
        "6": {"class_type": "OpenH3ICDecode", "inputs": {"pipeline": ["3", 0], "ic_latent": ["5", 0], "decoder": "standard", "audio": ["2", 1]}},
        "7": {"class_type": "CreateVideo", "inputs": {"images": ["6", 0], "fps": ["6", 1], "audio": ["6", 2]}},
        "8": {"class_type": "SaveVideo", "inputs": {"video": ["7", 0], "filename_prefix": "OpenH3IC/refined", "format": "mp4", "codec": "h264"}},
    }
    return ui, api


if __name__ == "__main__":
    destination = ROOT / "workflows"
    destination.mkdir(exist_ok=True)
    for low_memory, name in [(False, "OpenH3IC_2K"), (True, "OpenH3IC_LowVRAM_768p")]:
        ui, api = graph(low_memory)
        for value, suffix in [(ui, ".json"), (api, "_api.json")]:
            (destination / (name + suffix)).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
