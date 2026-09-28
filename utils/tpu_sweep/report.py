"""Aggregate a sweep directory into the files a dashboard reads.

Writes, next to the raw per-job JSON:
    results.jsonl        one flat record per job (suite, mode, status, numbers, error), traces dropped
    summary.json         environment, totals per suite/mode, and error clusters (a normalized error signature with
                         every job that hit it)
    model_results.json   per pipeline family, in the spirit of the transformers CI dashboard's `model_results.json`:
                         `success` / `failed` / `errors` / `skipped` totals, a per-mode breakdown, and `failures`
                         keyed by mode as `[{"line": <job id>, "trace": <error>}]`
    SUMMARY.md           a human-readable table
    REPORT.md            the scan report: pass rates, eager × compile outcomes, compile culprits, root causes,
                         precision-sensitive jobs, crashes, tensor parallelism, and a per-family table

    python utils/tpu_sweep/report.py tpu_sweep_results/<run>
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


STATUSES = ("pass", "fail", "error", "skipped")
TRACE_KEYS = ("trace",)


def signature(record):
    """A short, instance-independent label for why a job did not pass, so identical root causes group together."""
    text = record.get("error") or record.get("reason") or ""
    text = text.split("\n\nWhile executing")[0]
    lines = [line for line in text.splitlines() if line.strip()]
    # `BackendCompilerFailed: backend='tpu' raised:` carries the real error on the next line.
    if len(lines) > 1 and lines[0].rstrip().endswith("raised:"):
        lines = [f"{lines[0].split(':')[0]} > {lines[1]}"]
    head = lines[0] if lines else "unknown"
    head = re.sub(r"\[[\d, ]+\]", "[…]", head)
    head = re.sub(r"\(\d+(, \d+)*\)", "(…)", head)
    head = re.sub(r"\b\d+(\.\d+)?(e-?\d+)?\b", "N", head)
    head = re.sub(r"0x[0-9a-f]+", "0x…", head)
    head = re.sub(r"\b[\dx]*x(float|bfloat|int|complex|bool)\w*", "<tensor>", head)
    return head[:220]


def crash_reason(log_text):
    """The fatal line of a worker that died without writing a result: a TorchTPU `Check failed`, or a signal and the
    first TorchTPU frame it was raised in."""
    for line in log_text.splitlines():
        if "Check failed" in line:
            return "TorchTPU fatal: " + line.split("] ", 1)[-1].strip()
    signal = re.search(r"\*\*\* (SIG[A-Z]+)", log_text)
    if signal:
        frame = re.search(r"PC: @ +\S+ +\S+ +(.+)$", log_text, re.M) or re.search(
            r"@ +\S+ +\d+ +(torch_tpu::.+)$", log_text, re.M
        )
        return f"{signal.group(1)} in {frame.group(1).strip() if frame else 'unknown frame'}"
    return None


def load(out):
    records = []
    for suite in ("smoke", "tp"):
        for path in sorted((out / suite).glob("*/*.json")):
            record = json.loads(path.read_text())
            record["suite"] = suite
            record["result_file"] = str(path.relative_to(out))
            if suite == "tp":
                record.setdefault("family", record.get("name"))
            if record.get("stage") == "worker" and record.get("log") and (out / record["log"]).exists():
                reason = crash_reason((out / record["log"]).read_text(errors="replace"))
                if reason:
                    record["worker_exit"], record["error"] = record["error"], reason
            records.append(record)
    return records


def apply_precision_reruns(out, records):
    """Fold in `smoke_highest/`: numeric fails re-run with `torch.set_float32_matmul_precision("highest")`.

    TorchTPU runs fp32 matmuls at reduced precision by default, which a tiny random model can amplify past the
    tolerance. A fail that passes at `highest` is only precision-sensitive, so it counts as a pass and is flagged; the
    default-precision numbers stay on the record.
    """
    reruns = {}
    for path in (out / "smoke_highest").glob("*/*.json"):
        rerun = json.loads(path.read_text())
        reruns[(rerun["id"], rerun["mode"])] = rerun
    for record in records:
        rerun = reruns.get((record["id"], record["mode"]))
        if record["suite"] != "smoke" or record["status"] != "fail" or rerun is None:
            continue
        record["highest_precision_status"] = rerun["status"]
        record["highest_precision_max_abs_diff"] = rerun.get("max_abs_diff")
        if rerun["status"] == "pass":
            record["status"], record["precision_sensitive"] = "pass", True


def component_culprits(record):
    per_component = record.get("per_component") or {}
    return sorted(
        name
        for name, r in per_component.items()
        if r.get("graph_breaks_allowed") == "error" or r.get("fullgraph") == "error"
    )


def flatten(record):
    flat = {k: v for k, v in record.items() if k not in TRACE_KEYS}
    if record["status"] != "pass":
        flat["signature"] = signature(record)
    if record.get("per_component"):
        flat["compile_culprits"] = component_culprits(record)
    return flat


def _rate(counts):
    ran = counts["pass"] + counts["fail"] + counts["error"]
    return f"{100 * counts['pass'] / ran:.0f}% ({counts['pass']}/{ran})" if ran else "—"


def _short(record_id):
    return record_id.split("::")[-1].replace("PipelineTesterConfig", "").removesuffix("Config")


def _md(text):
    return str(text).replace("|", "/").replace("\n", " ")


def write_report(out, env, flat, summary):
    smoke = [r for r in flat if r["suite"] == "smoke"]
    tp = [r for r in flat if r["suite"] == "tp"]
    lines = [f"# TorchTPU sweep report — {env.get('date', '')}", ""]
    if env:
        versions = env.get("versions") or {}
        lines += [
            f"Branch `{env.get('git_branch')}` @ `{str(env.get('git_commit'))[:10]}` on {env.get('tpu_accelerator_type')}"
            f" ({env.get('tpu_chips')} chips).",
            ", ".join(f"{k} {v}" for k, v in versions.items()) if isinstance(versions, dict) else str(versions),
            "",
        ]

    targets = sorted({r["id"] for r in smoke})
    families = sorted({r.get("family") or "unknown" for r in smoke})
    lines += [
        "## Pass rates",
        "",
        f"{len(targets)} pipeline configs in {len(families)} families, each run in eager and compile mode on one chip;"
        f" {len({r.get('family') for r in tp})} tensor-parallel models on all chips. Pass rate = pass / (pass + fail +"
        " error).",
        "",
        "| suite/mode | pass rate | pass | fail | error | skipped |",
        "|---|---|---|---|---|---|",
    ]
    for key, counts in summary["totals"].items():
        lines.append(f"| {key} | {_rate(counts)} | " + " | ".join(str(counts[s]) for s in STATUSES) + " |")

    by_target = defaultdict(dict)
    for r in smoke:
        by_target[r["id"]][r["mode"]] = r["status"]
    outcomes = Counter((modes.get("eager"), modes.get("compile")) for modes in by_target.values())
    lines += ["", "## Eager × compile", "", "Configs per (eager, compile) outcome.", "", "| eager \\ compile | "]
    lines[-1] += " | ".join(STATUSES) + " |"
    lines.append("|---|" + "---|" * len(STATUSES))
    for eager in STATUSES:
        lines.append(f"| **{eager}** | " + " | ".join(str(outcomes.get((eager, c), 0)) for c in STATUSES) + " |")

    culprits = Counter(name for r in smoke for name in r.get("compile_culprits") or [])
    compile_failed = sum(1 for r in smoke if r["mode"] == "compile" and r["status"] in ("fail", "error"))
    lines += [
        "",
        "## What breaks compile",
        "",
        f"When a compile job fails, each component is retried compiled alone. Jobs where a component fails on its own"
        f" (out of {compile_failed} failing compile jobs):",
        "",
        "| component | jobs |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | {count} |" for name, count in culprits.most_common(12)]

    lines += ["", "## Top root causes", "", "| suite/mode | jobs | families | root cause |", "|---|---|---|---|"]
    family_of = {(r["id"], f"{r['suite']}/{r['mode']}"): r.get("family") for r in flat}
    for c in summary["error_clusters"][:20]:
        fams = sorted({family_of.get((i, c["suite_mode"])) or "?" for i in c["ids"]})
        shown = ", ".join(fams[:6]) + (f" +{len(fams) - 6}" if len(fams) > 6 else "")
        lines.append(f"| {c['suite_mode']} | {c['count']} | {shown} | `{_md(c['signature'])[:160]}` |")

    sensitive = sorted((r["mode"], _short(r["id"])) for r in smoke if r.get("precision_sensitive"))
    lines += [
        "",
        "## Precision-sensitive",
        "",
        "Failed at the default reduced fp32 matmul precision, passed with"
        ' `torch.set_float32_matmul_precision("highest")`; counted as passes.',
        "",
    ]
    lines += [f"- {mode}: `{name}`" for mode, name in sensitive] or ["- none"]

    crashes = sorted((r["mode"], _short(r["id"]), r.get("error")) for r in smoke if r.get("stage") == "worker")
    lines += ["", "## Crashes", "", "Workers that died (signal or TorchTPU fatal check) without writing a result.", ""]
    lines += [f"- {mode}: `{name}` — `{_md(error)[:160]}`" for mode, name, error in crashes] or ["- none"]

    lines += ["", "## Tensor parallelism", "", "| model | mode | status | root cause |", "|---|---|---|---|"]
    for r in sorted(tp, key=lambda r: (r.get("family") or "", r["mode"])):
        cause = f"`{_md(r['signature'])[:120]}`" if r.get("signature") else "—"
        lines.append(f"| {r.get('family')} | {r['mode']} | {r['status']} | {cause} |")

    lines += [
        "",
        "## Per family",
        "",
        "| family | configs | eager | compile | main root cause |",
        "|---|---|---|---|---|",
    ]
    for family in families:
        jobs = [r for r in smoke if (r.get("family") or "unknown") == family]
        rates = []
        for mode in ("eager", "compile"):
            counts = Counter(r["status"] for r in jobs if r["mode"] == mode)
            rates.append(_rate({s: counts.get(s, 0) for s in STATUSES}))
        causes = Counter(r["signature"] for r in jobs if r["status"] in ("fail", "error"))
        cause = f"`{_md(causes.most_common(1)[0][0])[:110]}`" if causes else "—"
        configs = len({r["id"] for r in jobs})
        lines.append(f"| {family} | {configs} | {rates[0]} | {rates[1]} | {cause} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    out = Path(sys.argv[1])
    env = json.loads((out / "environment.json").read_text()) if (out / "environment.json").exists() else {}
    records = load(out)
    apply_precision_reruns(out, records)
    flat = [flatten(r) for r in records]

    with open(out / "results.jsonl", "w") as f:
        for r in flat:
            f.write(json.dumps(r) + "\n")

    totals = defaultdict(Counter)
    clusters = defaultdict(list)
    for r in flat:
        key = f"{r['suite']}/{r['mode']}"
        totals[key][r["status"]] += 1
        if r["status"] in ("fail", "error"):
            clusters[(key, r["signature"])].append(r["id"])

    summary = {
        "environment": env,
        "totals": {k: {s: v.get(s, 0) for s in STATUSES} for k, v in sorted(totals.items())},
        "error_clusters": [
            {"suite_mode": k, "signature": sig, "count": len(ids), "ids": sorted(ids)}
            for (k, sig), ids in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    families = defaultdict(
        lambda: {"success": 0, "failed": 0, "errors": 0, "skipped": 0, "by_mode": {}, "failures": {}, "time_spent": {}}
    )
    for r in flat:
        fam = families[r.get("family") or "unknown"]
        mode = r["mode"] if r["suite"] == "smoke" else f"tp_{r['mode']}"
        counts = fam["by_mode"].setdefault(mode, {"success": 0, "failed": 0, "errors": 0, "skipped": 0})
        field = {"pass": "success", "fail": "failed", "error": "errors", "skipped": "skipped"}[r["status"]]
        fam[field] += 1
        counts[field] += 1
        if r["status"] in ("fail", "error"):
            fam["failures"].setdefault(mode, []).append(
                {
                    "line": r["id"],
                    "trace": r.get("error") or r.get("reason"),
                    "stage": r.get("stage"),
                    "signature": r["signature"],
                }
            )
        if r.get("wall_s") is not None:
            fam["time_spent"][mode] = round(fam["time_spent"].get(mode, 0) + r["wall_s"], 2)
    (out / "model_results.json").write_text(
        json.dumps({f"models_{k}": v for k, v in sorted(families.items())}, indent=1)
    )

    lines = [f"# TorchTPU sweep — {env.get('date', '')}", ""]
    if env:
        versions = env.get("versions", {})
        lines += [
            f"- Branch `{env.get('git_branch')}` @ `{str(env.get('git_commit'))[:10]}`, {env.get('tpu_accelerator_type')} ({env.get('tpu_chips')} chips)",
            "- " + ", ".join(f"{k} {v}" for k, v in versions.items())
            if isinstance(versions, dict)
            else f"- {versions}",
            "",
        ]
    lines += ["| suite/mode | pass | fail | error | skipped |", "|---|---|---|---|---|"]
    for k, v in summary["totals"].items():
        lines.append(f"| {k} | " + " | ".join(str(v[s]) for s in STATUSES) + " |")
    lines += ["", "## Top error clusters", "", "| suite/mode | count | signature |", "|---|---|---|"]
    for c in summary["error_clusters"][:40]:
        lines.append(f"| {c['suite_mode']} | {c['count']} | `{c['signature'].replace('|', '/')}` |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    write_report(out, env, flat, summary)
    print("\n".join(lines[: 12 + len(summary["totals"])]))


if __name__ == "__main__":
    main()
