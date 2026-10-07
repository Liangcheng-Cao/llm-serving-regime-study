# Reproducing the study

Run commands from the repository root. Phase 2F performs no GPU experiments.
Archived measurement outputs are inputs, not destinations for new experiments.

## Recorded environment

| Component | Phase 2D/2E environment |
|---|---|
| GPU | NVIDIA GeForce RTX 4070, 12GB (12,282 MiB reported) |
| Host / guest | Windows with WSL2; Ubuntu 26.04, Linux 6.18.33.2-microsoft-standard-WSL2 |
| Driver | 581.08; WSL NVIDIA-SMI reports tool version 580.76.07 |
| PyTorch / CUDA runtime build | 2.13.0+cu130 / CUDA 13.0 |
| Installed nvcc toolkit | CUDA 13.3, V13.3.73; distinct from PyTorch's CUDA build |
| Nsight Systems | 2026.1.3.425-261338342291v0 |
| SGLang / Python / aiohttp | 0.5.19 / 3.12.14 / 3.14.3 |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Execution | BF16, FlashInfer, overlap enabled, radix cache disabled |
| Workload | Seed 42, temperature zero, input128, forced output128; secondary output256 |

The [run manifest](../outputs/gpu-mechanism-study/phase2d/runs/d-128-c-b1-trace/manifest.json)
records exact commands, environment probes, source hashes and effective controls.
The historical queueing measurements used a different environment; do not pool them
with these measurements. Full dependency locking and a clean-machine installer are
not supplied. Exact source hashes, not only package version labels, guard patches.

## 1. CPU-only validation from committed files

Python 3.10+ is required; the study was validated with Python 3.12.14.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python src/validate.py
```

These commands need neither CUDA nor the model. The historical validator checks
the analytic case, 60 frozen scenarios, conservation, causal contrast and service
cliff. `requirements.txt` covers offline plotting/model analysis, not a GPU runtime.

Committed JSON/CSV files contain the results and provenance. To regenerate a
Phase 2D summary into a new location without touching archived outputs:

```bash
python - <<'PY'
from pathlib import Path
import shutil
from profiling import analyze_phase2d as a
from profiling import report_phase2d as report
source = a.OUT
target = Path('outputs/local-summary-reproduction').resolve()
target.mkdir(parents=True, exist_ok=False)
# collect() reads these manifests and compact parsed JSON; no SQLite is needed.
shutil.copyfile(source/'trace_manifest.jsonl', target/'trace_manifest.jsonl')
for name in ('runs', 'summaries'):
    shutil.copytree(source/name, target/name,
                    ignore=shutil.ignore_patterns('*.sqlite', '*.log', '*trace.csv'))
a.OUT = target
report.OUT = target
report.main()
_, traces, _, _, _, families, _ = a.collect()
_, _, serving, _, _, _, _ = a.collect()
a.figures(traces, serving, families)
PY
```

The Phase 2E report can be rebuilt from its committed `analysis.json` plus Phase 2D
analysis into a new location by setting `report_phase2e.OUT` to a fresh directory
containing a copy of that JSON and calling `report_phase2e.main()`. Report generators
otherwise write into their default phase directories; avoid those defaults when
preserving a released snapshot.

## 2. Event-level reanalysis

`profiling.analyze_phase2e` requires the original SQLite exports under each Phase 2D
`summaries/<run-id>/trace.sqlite`. These large files are retained locally, not in Git.
Their SHA-256 hashes are in the [input manifest](../outputs/gpu-mechanism-study/phase2e/input_manifest.json).
If available, analyze into a fresh destination:

```bash
python - <<'PY'
from pathlib import Path
from profiling import analyze_phase2e as a
from profiling import report_phase2e as report
target = Path('outputs/local-event-reproduction').resolve()
target.mkdir(parents=True, exist_ok=False)
a.OUT = target
a.main()
report.OUT = target
report.main()
PY
```

The parser uses inspected SQLite tables, worker-scoped correlation IDs, and NVTX
submission markers. It clips intervals to submissions 10→80, computes unions across
streams, and normalizes by 70×useful B. Driver/runtime calls are distinguished;
graph replay correlations remain many-to-one. See the [collection protocol](../profiling/phase2d_protocol.md)
and [mechanism report](../outputs/gpu-mechanism-study/phase2e/REPORT.md).

## 3. Fresh GPU collection — optional reader workflow, not run during packaging

Prepare a compatible Linux/WSL SGLang environment and Nsight Systems on PATH. Cache
the exact model revision before running: the harness disables downloads. Do not
assume that installing SGLang 0.5.19 reproduces the recorded source files; guarded
patches fail closed on a hash mismatch. Inspect and revalidate instrumentation for
any changed installation rather than disabling guards. A fresh-machine GPU setup
has not been independently replayed.

| Arm | B | Decode cap | Forced decode eager | Verified expected path |
|---|---:|---:|---|---|
| C | 8 | 8 | false | Graph bucket8 |
| D | 8 | 8 | true | Eager |
| A | 9 | 8 | false | Natural eager |
| B | 9 | 16 | false | Graph bucket12 |

Inspect configuration without launching a server:

```bash
python -m profiling.run_decode_experiment --arm C --dry-run
python -m profiling.run_decode_experiment --arm D --dry-run
```

For an intentional complete replication, use an unused output directory. The
following preserves order, warmups, independent starts and the secondary gate:

```bash
python - <<'PY'
import asyncio
from pathlib import Path
from profiling import run_phase2d as campaign
from profiling import analyze_phase2d as analysis
from profiling import report_phase2d as report
target = Path('outputs/local-gpu-replication').resolve()
target.mkdir(parents=True, exist_ok=False)
campaign.OUT = target
asyncio.run(campaign.main('primary'))  # stops at the first invalid attempt
asyncio.run(campaign.main('secondary'))  # reached only after primary succeeds
analysis.OUT = target
report.OUT = target
report.main()
_, trace, serving, _, _, families, _ = analysis.collect()
analysis.figures(trace, serving, families)
PY
```

Do not run concurrent jobs on the measurement GPU. The campaign creates 24 primary
server starts and six secondary starts. Each has three excluded warmups. Primary
unprofiled starts measure three cohorts; traces measure one. It verifies useful B,
path, bucket, model/workload counts, worker and markers. Every attempt is indexed;
failures are retained, with no silent replacement.

Nsight profiles the owned server process tree with CUDA/NVTX software tracing,
node-level graph detail, no CPU sampling/context-switch trace, and CUDA profiler
start/stop hooks after warmup. SQLite and compact summaries are exported by the
runner. Actual replay is observed rather than inferred from a requested cap.

## Trace storage and publication scope

**Committed:** manifests, source hashes, compact parsed JSON/CSV, statistics,
quality warnings, figures, reports and tests. **Not committed:** `.nsys-rep`, SQLite,
full event exports, server logs, and reconstructible compressed per-gap association
tables. Raw captures can be regenerated by the fresh collection workflow, but
new timestamps and environmental variation will not reproduce original hashes.

Manifests retain local filesystem paths, host identifiers and package provenance;
these are not portable path instructions. No credentials or model weights are
needed to inspect the committed evidence. Public reproducibility is therefore
**summary-level out of the box, event-level with retained/regenerated traces,
and hardware-dependent for new measurements**.
