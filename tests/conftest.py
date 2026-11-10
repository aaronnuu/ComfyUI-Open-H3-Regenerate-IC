import importlib.util
import os
from pathlib import Path
import sys

# Test-process only: avoid importing unrelated, incompatible TensorFlow wheels.
os.environ["USE_TF"] = "0"
core = os.environ.get("COMFYUI_PATH")
if core:
    sys.path.insert(0, core)
    from comfy.cli_args import args
    if os.environ.get("IC_TEST_CPU") == "1":
        args.cpu = True
    import comfy.model_base  # fail loudly if the requested integration environment is broken
    import nodes  # reserve the core name before pytest adds this plugin's nodes.py directory
    assert hasattr(nodes, "load_custom_node"), nodes.__file__
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("openh3ic_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
# Pytest also tries to collect the hyphenated plugin root as `__init__`.
# Reuse the already imported package so its relative imports keep their context.
sys.modules.setdefault("__init__", package)
