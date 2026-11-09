import importlib.util
import os
from pathlib import Path
import sys

# Test-process only: avoid importing unrelated, incompatible TensorFlow wheels.
os.environ["USE_TF"] = "0"
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("openh3ic_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
# Pytest also tries to collect the hyphenated plugin root as `__init__`.
# Reuse the already imported package so its relative imports keep their context.
sys.modules.setdefault("__init__", package)
