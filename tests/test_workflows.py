import json
from pathlib import Path

from openh3ic_test import NODE_CLASS_MAPPINGS


def test_workflow_ports_and_custom_required_inputs():
    root = Path(__file__).resolve().parents[1] / "workflows"
    files = list(root.glob("*.json"))
    assert len(files) == 4
    for path in files:
        graph = json.loads(path.read_text(encoding="utf-8"))
        if path.name.endswith("_api.json"):
            for item in graph.values():
                cls = NODE_CLASS_MAPPINGS.get(item["class_type"])
                if cls:
                    assert set(cls.INPUT_TYPES()["required"]) <= set(item["inputs"])
                for value in item["inputs"].values():
                    if isinstance(value, list):
                        assert value[0] in graph
        else:
            nodes = {item["id"]: item for item in graph["nodes"]}
            for identity, src, src_slot, dst, dst_slot, kind in graph["links"]:
                assert identity in nodes[src]["outputs"][src_slot]["links"]
                assert nodes[dst]["inputs"][dst_slot]["link"] == identity
                assert nodes[src]["outputs"][src_slot]["type"] == kind
                assert nodes[dst]["inputs"][dst_slot]["type"] == kind
