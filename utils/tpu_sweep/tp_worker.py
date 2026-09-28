"""Tensor-parallel check for one model on all local TPU chips, in eager or compile mode; rank 0 writes JSON.

Follows `tests/models/transformers/_tp_worker_common.py`: identical weights on every rank, an unsharded single-chip
reference, then `enable_parallelism` and the sharded forward. The spec's head count is raised to the world size so
every rank gets whole heads. In compile mode the sharded model is compiled with
`backend="tpu", fullgraph=True, dynamic=False`, retried with `fullgraph=False` if that fails.

    torchrun --nproc_per_node=8 utils/tpu_sweep/tp_worker.py <module>:<spec_fn> eager out.json
"""

import copy
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_tpu  # noqa: E402, F401
from torch.distributed.device_mesh import DeviceMesh  # noqa: E402
from torch.distributed.tensor import DTensor  # noqa: E402
from torch_tpu._internal import sync as tpu_sync  # noqa: E402

from diffusers import TensorParallelConfig  # noqa: E402


ATOL = RTOL = 0.1


def sync():
    tpu_sync.synchronize(None, wait=True)


def forward(model, inputs):
    sync()
    start = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs, return_dict=False)[0]
    if isinstance(output, DTensor):
        output = output.full_tensor()
    output = output.float().cpu()
    return output, time.perf_counter() - start


def check(spec, mode):
    module_name, _, fn_name = spec.partition(":")
    model_class, init_dict, inputs = getattr(importlib.import_module(module_name), fn_name)()
    world_size = dist.get_world_size()
    init_dict = {**init_dict, "num_attention_heads": world_size}
    record = {"model": model_class.__name__, "tp_size": world_size, "init_dict": init_dict, "atol": ATOL, "rtol": RTOL}

    torch.manual_seed(0)
    model = model_class(**init_dict).eval()
    record["num_params"] = sum(p.numel() for p in model.parameters())
    tpu_inputs = {k: v.to("tpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    stage = "reference"
    with torch.no_grad():
        cpu_out = model(**inputs, return_dict=False)[0].float()
    ref_model = copy.deepcopy(model).to("tpu")
    ref_out, record["single_chip_time_s"] = forward(ref_model, tpu_inputs)
    del ref_model

    try:
        stage = "shard"
        model.enable_parallelism(config=TensorParallelConfig(mesh=DeviceMesh("tpu", list(range(world_size)))))
        model = model.to("tpu")
        record["num_sharded_params"] = sum(isinstance(p, DTensor) for p in model.parameters())
        if mode == "compile":
            stage = "compile"
            pristine = model
            for fullgraph in (True, False):
                try:
                    model = torch.compile(pristine, backend="tpu", fullgraph=fullgraph, dynamic=False)
                    stage = "tp_first_call"
                    _, record["tp_first_call_s"] = forward(model, tpu_inputs)
                    record["fullgraph"] = fullgraph
                    break
                except Exception as e:
                    if not fullgraph:
                        raise
                    record["fullgraph_error"] = f"{type(e).__name__}: {e}"[:2000]
                    torch.compiler.reset()
        else:
            stage = "tp_first_call"
            _, record["tp_first_call_s"] = forward(model, tpu_inputs)
        stage = "tp_second_call"
        tp_out, record["tp_time_s"] = forward(model, tpu_inputs)
    except Exception as e:
        return {
            **record,
            "status": "error",
            "stage": stage,
            "error": f"{type(e).__name__}: {e}"[:2000],
            "trace": traceback.format_exc(),
        }

    record["output_shape"] = list(tp_out.shape)
    record["finite"] = bool(torch.isfinite(tp_out).all())
    record["max_abs_diff_vs_single_chip"] = (tp_out - ref_out).abs().max().item()
    record["max_abs_diff_vs_cpu"] = (tp_out - cpu_out).abs().max().item()
    record["single_chip_max_abs_diff_vs_cpu"] = (ref_out - cpu_out).abs().max().item()
    ok = record["finite"] and torch.allclose(tp_out, ref_out, atol=ATOL, rtol=RTOL)
    if not ok:
        record["reason"] = (
            "non-finite output"
            if not record["finite"]
            else f"TP output differs from single chip beyond atol=rtol={ATOL}"
        )
    return {**record, "status": "pass" if ok else "fail"}


def main():
    spec, mode, out_path = sys.argv[1:4]
    dist.init_process_group(backend="tpu_dist")
    start = time.perf_counter()
    try:
        record = check(spec, mode)
    except Exception as e:
        record = {
            "status": "error",
            "stage": "setup",
            "error": f"{type(e).__name__}: {e}"[:2000],
            "trace": traceback.format_exc(),
        }
    record.update(id=spec, mode=mode, wall_s=time.perf_counter() - start)
    if dist.get_rank() == 0:
        with open(out_path, "w") as f:
            json.dump(record, f, indent=1)
        print(
            f"[tp] {spec} {mode}: {record['status']} {record.get('max_abs_diff_vs_single_chip', record.get('error', ''))}"
        )
    dist.barrier()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
