"""Smoke-test one pipeline on one TPU chip, against a CPU reference, and write the result as JSON.

The pipeline is built from its fast-test config (`get_dummy_components` / `get_dummy_inputs`, tiny random weights),
run once on CPU, then a copy is moved to the TPU and run twice: the first call includes compilation, the second is the
steady-state time. The TPU output of the second call is compared to the CPU output.

Modes:
    eager    TorchTPU strict eager.
    compile  every diffusers model component is compiled with `backend="tpu", fullgraph=True, dynamic=False`:
             denoisers through `nn.Module.compile`, VAEs through their `decode` / `encode` methods (pipelines never
             call a VAE's `forward`). If that run fails, each component is retried compiled on its own — with
             `fullgraph=True`, then `fullgraph=False` — so the record names the component at fault.

    TPU_VISIBLE_CHIPS=0 python utils/tpu_sweep/smoke_worker.py <module>::<Class> eager out.json
"""

import importlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
# Dummy inputs and the reference are built on CPU; the TPU copy is moved explicitly.
os.environ["DIFFUSERS_TEST_DEVICE"] = "cpu"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_tpu  # noqa: E402, F401


ATOL = RTOL = 0.1

# Eight workers share the host; without a cap each one grabs every core for its CPU reference and they thrash.
torch.set_num_threads(int(os.environ.get("SWEEP_CPU_THREADS", max(1, (os.cpu_count() or 8) // 8))))
# Unset means the TorchTPU default (reduced-precision fp32 matmuls); the sweep re-runs numeric fails at "highest".
MATMUL_PRECISION = os.environ.get("SWEEP_MATMUL_PRECISION")
if MATMUL_PRECISION:
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
VAE_METHODS = ("decode", "encode")


def to_tensor(output):
    """Flatten a pipeline's first output (tensor, ndarray, PIL images, or nested lists of them) into a float tensor."""
    if isinstance(output, torch.Tensor):
        return output.detach().float().cpu()
    if isinstance(output, np.ndarray):
        return torch.from_numpy(output.astype(np.float32))
    if hasattr(output, "convert") and hasattr(output, "size"):  # PIL.Image
        return torch.from_numpy(np.asarray(output.convert("RGB"), dtype=np.float32) / 255.0)
    if isinstance(output, (list, tuple)):
        return torch.stack([to_tensor(o) for o in output])
    raise TypeError(f"Unsupported pipeline output type: {type(output).__name__}")


def run(pipe, config):
    inputs = config.get_dummy_inputs()
    torch.manual_seed(0)
    start = time.perf_counter()
    with torch.no_grad():
        output = to_tensor(pipe(**inputs)[0])
    return output, time.perf_counter() - start


def compilable_components(pipe, config):
    from diffusers import ModelMixin

    names = [name for name, component in pipe.components.items() if isinstance(component, ModelMixin)]
    # Configs whose model is not a `ModelMixin` (e.g. text diffusion LMs) name it explicitly.
    return names + [n for n in getattr(config, "extra_compile_components", ()) if n not in names]


def compile_components(pipe, names, fullgraph):
    compiled = []
    for name in names:
        component = getattr(pipe, name)
        methods = [m for m in VAE_METHODS if hasattr(component, m)] if name.startswith("vae") else []
        if methods:
            for method in methods:
                setattr(
                    component,
                    method,
                    torch.compile(getattr(component, method), backend="tpu", fullgraph=fullgraph, dynamic=False),
                )
            compiled.append(f"{name}.{'/'.join(methods)}")
        else:
            component.compile(backend="tpu", fullgraph=fullgraph, dynamic=False)
            compiled.append(name)
    return compiled


def short_error(e):
    return f"{type(e).__name__}: {e}"[:2000]


def compare(tpu_out, cpu_out):
    stats = {"output_shape": list(tpu_out.shape), "finite": bool(torch.isfinite(tpu_out).all())}
    if tpu_out.shape != cpu_out.shape:
        return False, {**stats, "reason": f"shape mismatch: TPU {tuple(tpu_out.shape)} vs CPU {tuple(cpu_out.shape)}"}
    diff = (tpu_out - cpu_out).abs()
    stats.update(
        max_abs_diff=diff.max().item() if diff.numel() else 0.0,
        mean_abs_diff=diff.mean().item() if diff.numel() else 0.0,
        ref_max_abs=cpu_out.abs().max().item() if cpu_out.numel() else 0.0,
    )
    if not stats["finite"]:
        return False, {**stats, "reason": "TPU output contains non-finite values"}
    close = torch.allclose(tpu_out, cpu_out, atol=ATOL, rtol=RTOL)
    return close, stats if close else {**stats, "reason": f"output differs from CPU beyond atol=rtol={ATOL}"}


def build_pipe(config):
    pipe = config.pipeline_class(**config.get_dummy_components())
    for component in pipe.components.values():
        if hasattr(component, "eval"):
            component.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def tpu_copy(config, cpu_pipe):
    """A fresh pipeline holding `cpu_pipe`'s exact weights, on the TPU.

    Built anew rather than deep-copied: some components hold non-leaf tensors that `copy.deepcopy` rejects, and a
    fresh build also leaves behind any state the CPU run put on the pipeline or its scheduler.
    """
    pipe = build_pipe(config)
    for name, component in pipe.components.items():
        if isinstance(component, torch.nn.Module):
            component.load_state_dict(getattr(cpu_pipe, name).state_dict())
    return pipe.to("tpu")


def smoke(target, mode):
    from diffusers.pipelines.pipeline_utils import DeprecatedPipelineMixin

    module_name, _, cls_name = target.partition("::")
    config = getattr(importlib.import_module(module_name), cls_name)()
    record = {
        "pipeline": config.pipeline_class.__name__,
        "atol": ATOL,
        "rtol": RTOL,
        "matmul_precision": MATMUL_PRECISION or "default",
    }

    if issubclass(config.pipeline_class, DeprecatedPipelineMixin):
        return {**record, "status": "skipped", "reason": "deprecated pipeline"}

    stage = "cpu_reference"
    try:
        cpu_pipe = build_pipe(config)
        cpu_out, record["cpu_time_s"] = run(cpu_pipe, config)
    except Exception as e:
        # A config that cannot even run on CPU says nothing about TPU support.
        return {
            **record,
            "status": "skipped",
            "stage": stage,
            "reason": f"CPU reference failed: {short_error(e)}"[:500],
            "trace": traceback.format_exc(),
        }

    try:
        stage = "to_tpu"
        tpu_pipe = tpu_copy(config, cpu_pipe)
        if mode == "compile":
            stage = "compile"
            record["compiled"] = compile_components(tpu_pipe, compilable_components(tpu_pipe, config), fullgraph=True)
        stage = "tpu_first_call"
        _, record["tpu_first_call_s"] = run(tpu_pipe, config)
        stage = "tpu_second_call"
        tpu_out, record["tpu_time_s"] = run(tpu_pipe, config)
    except Exception as e:
        record.update(status="error", stage=stage, error=short_error(e), trace=traceback.format_exc())
        if mode == "compile" and stage != "to_tpu":
            record["per_component"] = isolate_compile_failure(cpu_pipe, config)
        return record

    ok, stats = compare(tpu_out, cpu_out)
    return {**record, **stats, "status": "pass" if ok else "fail"}


def isolate_compile_failure(cpu_pipe, config):
    """Compile one component at a time to find which one breaks, and whether it only fails under `fullgraph=True`."""
    results = {}
    for name in compilable_components(cpu_pipe, config):
        result = {}
        for fullgraph in (True, False):
            torch.compiler.reset()
            key = "fullgraph" if fullgraph else "graph_breaks_allowed"
            try:
                tpu_pipe = tpu_copy(config, cpu_pipe)
                compile_components(tpu_pipe, [name], fullgraph=fullgraph)
                run(tpu_pipe, config)
                result[key] = "pass"
                break
            except Exception as e:
                result[key] = "error"
                result[f"{key}_error"] = short_error(e)
        results[name] = result
    return results


def main():
    target, mode, out_path = sys.argv[1:4]
    start = time.perf_counter()
    try:
        record = smoke(target, mode)
    except Exception as e:
        record = {"status": "error", "stage": "setup", "error": short_error(e), "trace": traceback.format_exc()}
    record.update(id=target, mode=mode, wall_s=time.perf_counter() - start)
    record = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in record.items()}
    with open(out_path, "w") as f:
        json.dump(record, f, indent=1)
    print(
        f"[smoke] {target} {mode}: {record['status']} {record.get('max_abs_diff', record.get('reason', record.get('error', '')))}"
    )


if __name__ == "__main__":
    main()
    # torch_tpu's runtime can hang on interpreter teardown; the result is already on disk.
    sys.stdout.flush()
    os._exit(0)
