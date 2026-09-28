"""List every pipeline smoke-test target: one per distinct `BasePipelineTesterConfig` in `tests/pipelines`.

A config is identified by the pipeline class plus the classes that define its `get_dummy_components` and
`get_dummy_inputs`, so the several `Test*` classes that share one config collapse to a single target.

    python utils/tpu_sweep/discover.py targets.json
"""

import importlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
os.environ.setdefault("DIFFUSERS_TEST_DEVICE", "cpu")


def _definer(cls, name):
    for base in cls.__mro__:
        if name in base.__dict__:
            return base
    return None


def discover():
    from tests.pipelines.testing_utils.common import BasePipelineTesterConfig

    targets, import_errors, seen = [], [], set()
    for path in sorted((REPO / "tests" / "pipelines").rglob("test_*.py")):
        module_name = ".".join(path.relative_to(REPO).with_suffix("").parts)
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            import_errors.append(
                {"module": module_name, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            )
            continue

        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name or not issubclass(cls, BasePipelineTesterConfig):
                continue
            components_def = _definer(cls, "get_dummy_components")
            inputs_def = _definer(cls, "get_dummy_inputs")
            if components_def is BasePipelineTesterConfig or inputs_def is BasePipelineTesterConfig:
                continue
            pipeline_class = cls.__dict__.get("pipeline_class") or getattr(cls, "pipeline_class", None)
            if not inspect.isclass(pipeline_class):
                continue
            key = (pipeline_class, components_def, inputs_def)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                {
                    "id": f"{module_name}::{name}",
                    "module": module_name,
                    "cls": name,
                    "pipeline": pipeline_class.__name__,
                    "family": path.parent.name,
                }
            )
    from utils.tpu_sweep.extra_configs import EXTRA_TARGETS

    module_name = "utils.tpu_sweep.extra_configs"
    for name, family in EXTRA_TARGETS.items():
        pipeline = getattr(importlib.import_module(module_name), name).pipeline_class.__name__
        targets.append(
            {"id": f"{module_name}::{name}", "module": module_name, "cls": name, "pipeline": pipeline, "family": family}
        )
    return targets, import_errors


if __name__ == "__main__":
    targets, import_errors = discover()
    with open(sys.argv[1], "w") as f:
        json.dump({"targets": targets, "import_errors": import_errors}, f, indent=1)
    print(f"{len(targets)} targets, {len(import_errors)} import errors", file=sys.stderr)
