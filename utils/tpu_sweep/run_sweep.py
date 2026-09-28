"""Run the TPU sweep: pipeline smoke tests on one chip each, then tensor-parallel checks on all chips.

Both suites run in eager and compile mode.

Each job is a `smoke_worker.py` subprocess pinned to one chip with `TPU_VISIBLE_CHIPS`. Results land in
`<out>/smoke/<mode>/<id>.json` with the full log next to it in `<out>/logs/`; a job that crashes or times out gets a
record built from its log instead. Re-running skips jobs that already have a result, so an interrupted sweep resumes.

TP targets are the models with a `_tp_plan` and a `make_tpu_tp_spec` in their model test file; each runs as a
`torchrun` job over all chips, one at a time, with results in `<out>/tp/<mode>/`.

    python utils/tpu_sweep/run_sweep.py tpu_sweep_results/<run> [--suites smoke tp] [--modes eager compile]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PY = sys.executable


def safe(target_id):
    return target_id.replace("tests.pipelines.", "").replace("::", "__").replace(".", "_")


def environment():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=120).stdout.strip()
        except Exception as e:
            return f"unavailable: {e}"

    versions = run(
        [
            PY,
            "-c",
            "import json, importlib.metadata as m\n"
            "out = {}\n"
            "for p in ('torch', 'torch_tpu', 'libtpu', 'diffusers', 'transformers', 'accelerate'):\n"
            "    try: out[p] = m.version(p)\n"
            "    except Exception: out[p] = None\n"
            "print(json.dumps(out))",
        ]
    )
    chip = run([PY, "-c", "from torch_tpu._internal.utils import hardware as h; print(h.get_tpu_device_count())"])
    return {
        "date": time.strftime("%Y-%m-%d"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": platform.python_version(),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain", "--untracked-files=no"])),
        "versions": json.loads(versions) if versions.startswith("{") else versions,
        "tpu_chips": chip.splitlines()[-1] if chip else None,
        "tpu_accelerator_type": os.environ.get("TPU_ACCELERATOR_TYPE"),
    }


def run_job(target, mode, out, chips: Queue, timeout_s, precision=None):
    suite_dir = "smoke" if precision is None else f"smoke_{precision}"
    result_path = out / suite_dir / mode / f"{safe(target['id'])}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    log_path = out / "logs" / suite_dir / mode / f"{safe(target['id'])}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    chip = chips.get()
    start = time.perf_counter()
    try:
        env = {**os.environ, "TPU_VISIBLE_CHIPS": str(chip)}
        if precision is not None:
            env["SWEEP_MATMUL_PRECISION"] = precision
        cmd = [PY, str(HERE / "smoke_worker.py"), target["id"], mode, str(result_path)]
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO, start_new_session=True
            )
            try:
                returncode = proc.wait(timeout=timeout_s)
                reason = None if result_path.exists() else f"worker exited with code {returncode} and wrote no result"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
                proc.wait()
                reason = f"timed out after {timeout_s}s"
    finally:
        chips.put(chip)

    if reason is not None:
        tail = log_path.read_text(errors="replace").splitlines()[-60:]
        record = {
            "id": target["id"],
            "mode": mode,
            "status": "error",
            "stage": "worker",
            "error": reason,
            "trace": "\n".join(tail),
        }
        result_path.write_text(json.dumps(record, indent=1))
    record = json.loads(result_path.read_text())
    record.update(
        family=target["family"], log=str(log_path.relative_to(out)), chip=chip, job_wall_s=time.perf_counter() - start
    )
    result_path.write_text(json.dumps(record, indent=1))
    print(f"[{mode:7s}] {record['status']:7s} {target['id']}", flush=True)
    return record


TP_SPECS = {
    "flux": "tests.models.transformers.test_models_transformer_flux:make_tpu_tp_spec",
    "flux2": "tests.models.transformers.test_models_transformer_flux2:make_tpu_tp_spec",
    "qwenimage": "tests.models.transformers.test_models_transformer_qwenimage:make_tpu_tp_spec",
}


def run_tp_job(name, spec, mode, out, chips, timeout_s):
    result_path = out / "tp" / mode / f"{name}.json"
    if result_path.exists():
        return
    log_path = out / "logs" / "tp" / mode / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    env = {k: v for k, v in os.environ.items() if k != "TPU_VISIBLE_CHIPS"}
    wrapper = subprocess.run(
        [PY, "-m", "torch_tpu._internal.distributed.launchers.singlehost_wrapper"], capture_output=True, text=True
    )
    for line in wrapper.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            env[key.strip()] = value.strip().strip("'\"")
    cmd = [
        PY,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={chips}",
        str(HERE / "tp_worker.py"),
        spec,
        mode,
        str(result_path),
    ]
    start = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO, start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout_s)
            reason = None if result_path.exists() else f"torchrun exited with code {returncode} and wrote no result"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
            proc.wait()
            reason = f"timed out after {timeout_s}s"
    if reason is not None:
        tail = log_path.read_text(errors="replace").splitlines()[-80:]
        result_path.write_text(
            json.dumps(
                {
                    "id": spec,
                    "mode": mode,
                    "status": "error",
                    "stage": "worker",
                    "error": reason,
                    "trace": "\n".join(tail),
                },
                indent=1,
            )
        )
    record = json.loads(result_path.read_text())
    record.update(name=name, log=str(log_path.relative_to(out)), job_wall_s=time.perf_counter() - start)
    result_path.write_text(json.dumps(record, indent=1))
    print(f"[tp {mode:7s}] {record['status']:7s} {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--suites", nargs="+", default=["smoke", "tp"])
    parser.add_argument("--modes", nargs="+", default=["eager", "compile"])
    parser.add_argument("--filter", default=None, help="Only run targets whose id contains this substring.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Only run the `<mode> <id>` pairs listed here.")
    parser.add_argument(
        "--precision", default=None, help="fp32 matmul precision for smoke jobs; results go to `smoke_<precision>/`."
    )
    parser.add_argument("--chips", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    targets_path = out / "targets.json"
    if not targets_path.exists():
        subprocess.run([PY, str(HERE / "discover.py"), str(targets_path)], cwd=REPO, check=True)
    targets = json.loads(targets_path.read_text())["targets"]
    if args.filter:
        targets = [t for t in targets if args.filter in t["id"]]

    env_path = out / "environment.json"
    if not env_path.exists():
        env_path.write_text(json.dumps(environment(), indent=1))

    if "smoke" in args.suites:
        chips = Queue()
        for chip in range(args.chips):
            chips.put(chip)
        jobs = [(t, m) for m in args.modes for t in targets]
        if args.ids_file:
            wanted = {tuple(line.split()) for line in args.ids_file.read_text().splitlines() if line.strip()}
            jobs = [(t, m) for t, m in jobs if (m, t["id"]) in wanted]
        print(f"{len(jobs)} smoke jobs on {args.chips} chips", flush=True)
        with ThreadPoolExecutor(args.chips) as pool:
            list(pool.map(lambda job: run_job(*job, out, chips, args.timeout, args.precision), jobs))

    if "tp" in args.suites:
        for mode in args.modes:
            for name, spec in TP_SPECS.items():
                if not args.filter or args.filter in name:
                    run_tp_job(name, spec, mode, out, args.chips, args.timeout)


if __name__ == "__main__":
    main()
