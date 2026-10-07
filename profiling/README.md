# Fixed-cohort runner and Nsight analysis

For the current publication workflow, see [reproduction](../docs/REPRODUCIBILITY.md).
This guide retains the Phase 2B/2C validation details; Phase 2D adds the controlled
campaign and submission window, and Phase 2E adds offline mechanism analysis.
Use fresh output locations when reproducing a released snapshot.

This pipeline extends the existing study without running or modifying its archived
scripts. Arm letters below follow the Phase 2B/2C request; the Phase 2A plan used
different table letters. Always identify an experiment by its explicit policy.

| Arm | Useful B | Decode cap | Forced decode eager | Expected path |
|---|---:|---:|---|---|
| A | 9 | 8 | false | Natural eager |
| B | 9 | 16 | false | Graph, historically bucket 12 |
| C | 8 | 8 | false | Graph, bucket 8 |
| D | 8 | 8 | true | Forced eager |

## Run inside the existing WSL environment

From the repository root with the SGLang environment's Python active:

```bash
python -m profiling.run_decode_experiment --arm A --dry-run
python -m profiling.run_decode_experiment --arm B --output-len 128 --warmup 3 --repeats 3
python -m profiling.run_decode_experiment --arm D --output-len 256 --warmup 3 --repeats 3
python -m profiling.run_nsys_profile --arm A --output-len 128
python -m profiling.run_nsys_profile --arm B --output-len 128
python -m unittest discover -s tests -v
python src/validate.py
```

Run GPU experiments sequentially on this single GPU. Nsight runs accept exactly
one measured cohort. Automatic run IDs include UTC time and random suffix; an
existing run directory is never reused. `--run-id`, `--output-dir`, `--port`,
`--batch-size`, `--max-decode-bs`, `--force-decode-eager`, `--input-len`,
`--output-len`, `--warmup`, `--repeats`, `--model`, `--revision`, `--dtype`,
`--seed` and `--graph-policy` expose configuration. Initial dtype/policy validation
fixes BF16/enabled policy. `--no-nvtx` is available for unprofiled perturbation
checks, but trace validation requires the measurement NVTX range.

`configs/validation.json` records the bounded validation design, not a general
campaign scheduler. No GPU package installations are performed by the runner.
Model downloads are disabled. The exact revision must already be cached. Optional
Nsight Compute counters and custom kernels are outside this phase.

## Architecture and verification

The Python client launches its own server process group on a checked-free loopback
port. It snapshots environment, command, Git state and source hashes, then creates
a guarded source-only SGLang copy in a unique Linux temporary directory. Three source
files receive observer calls: after the existing copy-completion synchronization,
immediately before actual graph backend replay, and at ModelRunner decode submission
for the Phase 2D boundary markers. Audited SHA-256 checks and
single-anchor replacements fail closed if the installation changes. The installed
package is unchanged; retained temporary copies support inspection.

The client checks `/health` and effective `/server_info` settings, sends three
excluded warmup cohorts, then fixed measured cohorts. Every request uses 128
copies of token ID 1000+i by default, deterministic temperature zero and forced
output length. Prompt/output counts are verified. SSE token counts, rather than
transport chunks, define the client proxy. If tokens 11 or 81 are not individually
observed, the proxy is unavailable; server completion timing remains independent.

The observer records actual B, returned graph flag, consecutive forward iteration
IDs and selected replay buckets in memory. It adds no CUDA synchronization and
flushes only at cohort completion. The stable window uses completion rows 9…79,
71 endpoints and 70 intervals. Terminal overlap rows are not counted in that
window. Wrong B/path, missing/mixed buckets, nonconsecutive steps or missing data
invalidate the run. Warmups are also checked but excluded from timing summaries.
`requested_mode` and `verified_mode` are separate fields.

Baseline includes this small observer, a completion NVTX range and two submission
markers per cohort in the current pipeline. It is
unprofiled, not instrumentation-free. This phase does not quantify all observer
overhead; Nsight comparison uses the same observer/annotation policy in both arms.

## Nsight integration and scope

Nsight launches the SERVER process tree, not only the HTTP client, using CUDA and
NVTX tracing, node-level graph detail, no CPU sampling and no context-switch trace.
The runner invokes SGLang `/start_profile` with only `activities=["CUDA_PROFILER"]`
after warmup, then `/stop_profile` after the one measured cohort. It does not start
a PyTorch profiler simultaneously. Only the run's owned process group is terminated.

The capture excludes startup, loading and warmup, but includes measured-cohort
prefill and tail. Two worker-side NVTX calls identify the steady completion window
as `GPU_MECH_MEASUREMENT_COMPLETION:<cohort>`; warmup has an analogous range outside
capture. This is deliberately not a claimed GPU decode-forward range. Separate
prefill/sampling/per-token GPU markers are not implemented. Native prefill logs and
the marked completion window make those boundaries inspectable without extensive
runtime instrumentation.

Supported exports verified with installed Nsight 2026.1.3 are `cuda_api_sum`,
`cuda_gpu_kern_sum`, `cuda_gpu_trace`, and `nvtx_pushpop_trace`, plus SQLite. CLI
exports opening successfully validate the report format without a GUI dependency.
SQLite validation requires kernels and runtime APIs in the exact PID recorded by
the observer, a same-PID measurement NVTX range and graph-launch presence/absence
consistent with the verified path. Graph node records must exist.

CSV summary metrics describe the whole captured cohort. The CUDA API summary can
include both runtime and driver records: counts are API records, and summed time
is not exclusive host wall time. Synchronization API names use `Synchronize`;
other waits such as stream wait events are not silently categorized as CPU blocking.

SQLite window metrics include events wholly contained within the completion range,
with explicit scope. Pipeline overlap means some corresponding GPU operations may
cross these boundaries. These are temporal diagnostics, not causally assigned
per-token kernel costs. Runtime API durations, GPU durations and synchronization
times overlap and must not be added into a latency decomposition.

Normalization utilities require explicit generated-token, decode-iteration and
request denominators. Window normalizations use B×70 useful tokens, 70 intervals
and B requests; padded slots never count as useful work. Full-cohort output rate
uses B×output_len and includes prefill. `useful_tokens_s` uses B/mean completion
interval and is a decode-rate proxy, not full serving throughput.

## Artifacts and failure handling

All measurements live below `outputs/gpu-mechanism-study/`:

- `runs/<id>/manifest.json`: requested policy, verified cohorts, statistics,
  exact commands, environment, source hashes and failures.
- `runs/<id>/server_info.json`, `server.log`, `observations.jsonl`: retained local
  raw evidence; ignored in Git to keep the repository compact.
- `traces/<id>.nsys-rep`: local report with path, size and hash in the manifest.
- `summaries/<id>/`: compact CSV/JSON, export log, ignored SQLite and full GPU trace.
- `experiment_manifest.jsonl`: append-only run index; invalid runs remain visible.
- `results.csv`, `validation_summary.json`: fresh validation summary only.

Parser/schema or mode-check failure invalidates the trace instead of assigning
zero metrics. No automatic retry or historical-data regeneration occurs.
`python -m profiling.summarize_validation` rebuilds the named sanity/trace validation
summary, excludes invalid runs and checks stored resolved configurations. It is
specific to this validation set, not a generic campaign analysis program.

`valid` on these first traces means structural capture and execution-path checks
passed, not exhaustive event collection. Later offline `quality_review.json` files
retain worker-scoped warnings, correlation coverage and `completeness_proven=false`.
Nsight reported CUDA/NVTX completeness warnings on both validation traces. All
observed steady-window kernels correlate to a worker API, but this cannot detect
an operation for which both records are absent. Treat extracted counts as observed
counts until completeness is independently established. Worker diagnostic errors
fail validation; warnings remain explicit and restrict interpretation.

Phase 2D adds independent restart statistics and bounded campaign automation;
Phase 2E adds event association and a Level-2 mechanism conclusion. Memory counters,
Level-3 causal attribution and an optimization study are not implemented. The
completion-window metrics described above are historical validation diagnostics;
current primary metrics use the submission window with interval clipping, as
defined in [the Phase 2D protocol](phase2d_protocol.md).
