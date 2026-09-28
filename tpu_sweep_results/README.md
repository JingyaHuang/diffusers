---
pretty_name: Diffusers TorchTPU CI
tags:
- diffusers
- tpu
---

# Diffusers TorchTPU CI results

Results of sweeping every `diffusers` pipeline fast-test config, plus the tensor-parallel models, on TorchTPU. One
folder per run, named by date (`YYYY-MM-DD/`).

## Suites

- **smoke** — each `BasePipelineTesterConfig` in `tests/pipelines` (tiny random weights from `get_dummy_components`)
  runs on one TPU chip and is compared to the same pipeline on CPU. `pass` means finite output within
  `atol = rtol = 0.1` of CPU. Pipelines whose fast tests don't use that contract (LEdits++, T2I-Adapter,
  DiffusionGemma, LLaDA2) are covered by the adapters in `utils/tpu_sweep/extra_configs.py`.
  - `eager`: TorchTPU strict eager.
  - `compile`: every diffusers model component compiled with `backend="tpu", fullgraph=True, dynamic=False`
    (VAEs through `decode` / `encode`). On failure, `per_component` retries each component alone, with `fullgraph=True`
    then `False`, to name the culprit.
  - Numeric fails are re-run with `torch.set_float32_matmul_precision("highest")` (`smoke_highest/`); a fail that
    passes there counts as `pass` with `precision_sensitive: true`.
- **tp** — models with a `_tp_plan`, sharded over all chips with `enable_parallelism` and compared to the unsharded
  model on one chip, in `eager` and `compile`.

## Files per run

| file | content |
|---|---|
| `environment.json` | date, git branch / commit, package versions, TPU type and chip count |
| `results.jsonl` | one flat record per job: `suite`, `mode`, `id`, `family`, `status`, timings, diffs, `signature`, `error` |
| `summary.json` | totals per suite/mode and `error_clusters` (normalized error signature → job ids) |
| `model_results.json` | per pipeline family (`models_<family>`): `success` / `failed` / `errors` / `skipped`, `by_mode`, `failures` keyed by mode as `[{line, trace, stage, signature}]`, `time_spent` |
| `SUMMARY.md` | human-readable totals and top error clusters |
| `smoke/<mode>/*.json`, `smoke_highest/<mode>/*.json`, `tp/<mode>/*.json` | raw per-job records, with full traces |
| `logs/` | full stdout/stderr of every job |

`status` is one of `pass`, `fail` (ran, but output mismatched or non-finite), `error` (raised or crashed), `skipped`.
