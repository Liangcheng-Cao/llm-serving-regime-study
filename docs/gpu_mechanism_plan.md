# Phase 2A: GPU/CUDA mechanism study — audit and design

Audit date: 2026-10-07 (Asia/Shanghai). Baseline commit: `72bccf5`.
Status: design only; no new model inference, profiling, or GPU measurements performed.
All Phase 2B–2I work and mechanism conclusions remain pending.

## A. Repository and evidence audit

Continue this existing `llm-serving-regime-study` repository, not the older
`sglang-inference-benchmark`. Initial `git status --short --branch` showed clean
`main...origin/main`. Six existing commits separate initialization, sanitized
evidence, causal controls, queue model, figures, and documentation. No AGENTS.md
was found in the repository or its ancestor directories.

| Location | Existing purpose / reuse boundary |
|---|---|
| `README.md` | Execution boundary, same-B treatment, pulse/long-horizon results, classical queue explanation |
| `experiments/step1_regime_audit.md` | Installed source control-flow audit and native mode checks |
| `experiments/step2_service_cliff.md` | 2×2 causal design, observer controls, cohort results |
| `experiments/step3_pulse_response.md`, `step4_long_horizon.md` | Queue response and matched future-arrival experiments |
| `experiments/step5_classical_model.md` | Fresh service calibration and classical-model falsification |
| `docs/methodology.md`, `data_provenance.md`, `limitations.md` | Definitions, sanitization, environment and inferential limits |
| `data/inputs/` | Calibration intervals, reconstructed initial states, frozen simulation reference |
| `data/processed/` | Causal cohorts, observer pairs, service curve, pulse/queue summaries |
| `src/` | Service-curve analysis, processor-sharing simulation, validation, figures |
| `figures/` | Four existing study figures; retain unchanged |
| `requirements.txt` | Offline NumPy/Matplotlib dependencies, not a GPU environment lock |

There is no packaged GPU runner, profiling directory, conventional test directory,
or runtime configuration file in this public repository. `src/validate.py` is its
existing validation suite. Do not confuse the older HTTP benchmark's concurrency
with measured useful decode batch size in this study.

Private original archives were located alongside this repository under
`../outputs/execution-regime-step1` through `execution-regime-step5`. They contain
raw observations, server information, commands, protocols, patches and scripts.
They are audit inputs only: do not rerun those scripts in place because several
write back into their archive directory. The public data are sanitized projections,
not full raw logs. No private absolute paths or raw logs need to be imported here.

Read-only validation of Step 2 `repetitions.json` confirmed 80 cohorts: 40 observer
ON, each with 70 intervals and matching stable B/path flags, plus 40 observer OFF.
The installed hashes of all 18 files in the Step 1 source manifest still match.
A preservation baseline covers 205 files across repository data and the five
private archives. Existing validation passed: analytic case, 60 frozen scenarios,
flow conservation, same-B causal contrast and service cliff. No regeneration of
historical results was needed.

## B. What difference needs explaining?

The strongest historical evidence is already a SAME-B comparison at B=9.
The four Step 2 cells in `data/processed/step2_summary.csv` are:

| Useful B | Decode cap | Verified path / padded bucket | Mean completed-decode interval (ms) | Useful tokens/s |
|---:|---:|---|---:|---:|
| 8 | 8 | graph / 8 | 3.83768868 | 2084.5881 |
| 8 | 16 | graph / 8 | 3.59472581 | 2225.4827 |
| 9 | 8 | eager | 10.89182366 | 826.3079 |
| 9 | 16 | graph / 12 | 3.65508214 | 2462.3250 |

B=9 eager/graph interval ratio is 2.98; the absolute gap is about 7.237 ms per
completed batch-decode interval. B=8 is a negative control for changing cap:
its pooled difference is 6.76%, with block differences +15.16% and −1.71%.
The multiplicative difference-in-differences is 2.791. Ten cohorts per cell came
from only two server starts per cap; cohorts are not ten independent restarts.
Observer-off B=9 client proxies also differ substantially (+205.82% eager over
graph), but common hook checks remained. This is not zero-overhead proof.

Timing is recorded after an existing copy-completion synchronization; it includes
runtime/scheduler overlap and is not isolated GPU kernel time. Existing metrics
include mean/median/p95 cadence, cohort CV, useful token rate, client streaming
proxy, cohort E2E, actual B, graph flag, and later outstanding counts, arrivals,
completions, server residency and path fractions. There are no paired Nsight
host/GPU mechanism measurements in this repository.

Separately, Step 5 calibrated B={1,2,4,6,8,9,10,12,14,16}, three cohorts per B.
Its B=8/B=9 intervals are 3.26090743/10.13611997 ms. Capacity proxies are
19.16644409/6.93682595 requests/s for 128-token work. Do not combine these timings
with Step 2 as if they came from the same run. The proxy excludes explicit prefill
cost and is not directly measured end-to-end request throughput.

## C. Exact supported boundary

For this installed ordinary TP1/DP1 decoder configuration, decode graph captures
are [1,2,4,8], padding enabled. Source eligibility permits compatible useful B≤8;
native Step 1 observations directly verify graph at B={1,4,8} and eager at B={9,16}.
Step 5 adds cadence observations at the wider B set above. The boundary is therefore
8→9 for the recorded default/cap8 configuration, not a universal SGLang boundary.
The memory heuristic chooses cap8 below 20 GiB; it is not a hardware capacity limit.
Cap16 includes [1,2,4,8,12,16], and B=9 uses bucket 12. Prefill graph eligibility is
separate and does not rescue uncovered decode batches.

## D. Exact mode intervention

The original `run_step2.py` passes `--cuda-graph-max-bs-decode 8` or `16` to
`python -m sglang.launch_server`. It does not force eager at B=8. Native verification
uses `SGLANG_LOG_FORWARD_ITERS=1`, `SGLANG_LOG_DECODE_GRAPH_KEY=1`, and
`--decode-log-interval 1`; timed runs use a quiet interval and an in-memory observer
in a separate package copy. Installed package files remain unchanged.

Installed CLI help and current `srt/arg_groups/cuda_graph_hook.py` support
`--disable-decode-cuda-graph`: it disables the decode backend while leaving the
prefill setting alone. Thus forced eager is supported by the runtime, but is not
an exposed arm of the archived Step 2 harness and has not been measured here.
Avoid deprecated global `--disable-cuda-graph`, which changes both phases. Explicit
backend selectors and structured graph config can override convenience settings;
reject conflicting options and record resolved configuration. A requested mode
is not proof: require native B, returned execution flag, and replay-bucket evidence.

## E. Reusable infrastructure and frozen workload

Adapt, do not execute in place, the archived Step 2 cohort construction, lifecycle
management, shape warmup, path checks and completion observer. Step 5 supplies
balanced batch ordering and normalization. Reuse existing CSV conventions and
offline validation. The older streaming benchmark is a reference for transport
parsing; it is not a fixed-B GPU harness.

Freeze Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, BF16, FlashInfer, one GPU, TP/DP/PP=1,
overlap enabled, speculative decoding/LoRA off, radix cache off, memory fraction
0.669, chunked prefill 2048, seed 42. Historical dtype/backend were resolved from
defaults; the new manifest must store both launch arguments and effective settings.
Use exact `input_ids=[[1000+i]*128 for i in range(B)]`, temperature 0,
`ignore_eos=true`, and exact output lengths 128 and 256. Preserve request/token
assignment across paired arms. Check actual B at every accepted decode step.

Proposed default warmup: three full cohorts per B/output-length/server configuration,
excluded from all results, after model loading and capture finish. Separate warmup
from the first nine completed decode steps excluded within each measured cohort.
Retain the historical primary window: completion endpoints 10→80, 70 intervals.
Use explicit zero-based indices 9→79 and matching forward IDs to avoid off-by-one
errors. Prefill contributes the first output token; 128 outputs normally imply
127 useful decode steps. Terminal in-flight overlap rows cannot silently count as
useful output work. Longer-output runs retain the same early window; also predeclare
endpoints 138→208 for a 70-interval late window, reported separately because KV
context age changes. Full-cohort E2E is auxiliary, never labeled decode-only latency.

## F. Controlled experiment matrix

All rows below are planned, not newly measured. Preserve prefill policy across arms.

| Cell | Useful B | Decode cap | Decode disabled | Expected path/bucket | Purpose |
|---|---:|---:|---|---|---|
| A | 8 | 8 | no | graph/8 | Historical baseline; natural boundary lower side |
| B | 8 | 8 | yes | eager | Primary same-B, unpadded graph vs forced eager control |
| C | 9 | 8 | no | eager | Historical natural boundary upper side |
| D | 9 | 16 | no | graph/12 | Reproduce historical same-B coverage intervention |
| E | 8 | 16 | no | graph/8 | Historical negative control; compare E/D graph scaling |
| F | 9 | 8 | yes | eager | Compare B/F eager batch scaling and C/F eager-path equivalence |

Primary pairs: A/B and C/D. Batch scaling: B/F (same forced-eager policy) and E/D
(same graph policy, with bucket 8→12 explicitly part of that comparison). Natural
transition: A/C. Negative controls: A/E and C/F. Repeat at output lengths 128 and
256. An exact B=9 capture or B=12 pair is a later padding control only if kernel
differences make it informative; changing padding globally is not an initial arm.

For lightweight timing, predeclare four server blocks per configuration, with
balanced/reversed configuration order and three measured cohorts per B/length per
block (12 cohorts/cell/length). Seed the within-block order. Report cohort mean,
median, SD/CV and interval p50/p95 separately, plus block-level paired contrasts.
Never treat 70 within-cohort intervals as independent repeats. Keep failures and
rejected path/B checks visible; no repeated search for favorable timings.

After a bounded trace smoke test in a later phase, target three independent
server-start traces per primary cell A/B/C/D at 128 outputs; reuse these for A/C.
Add E/F or 256-output traces only if the unprofiled checks warrant the cost. Measure
matched unprofiled, annotation-only and Nsight-instrumented cohorts. Stop and
report missing/intrusive traces instead of silently expanding the run budget.

## G. Nsight integration for this environment

Read-only checks found WSL2 Ubuntu-26.04, RTX 4070 (12,282 MiB), Python 3.12.14,
SGLang metadata 0.5.19, PyTorch metadata 2.13.0 with CUDA build 13.0, toolkit 13.3,
Nsight Systems 2026.1.3 and Nsight Compute 2026.2.1 (installed outside PATH).
`nsys status --environment` passes process-tree CPU profiling but fails system-wide
CPU profiling at paranoid level 2. No profiler smoke test was run in Phase 2A.

The current `srt/entrypoints/http_server.py` accepts POST `/start_profile` and
`/stop_profile`. Both profiler implementations support activities
`["CUDA_PROFILER"]` and invoke cudaProfilerStart/Stop in the GPU worker. This is
preferable to accidentally starting a second PyTorch/CUPTI profiler. Initial
integration should launch the SERVER process tree under Nsight, finish warmups,
start CUDA-only capture, send one cohort, stop capture, and shut down only the
owned server after export. A client-only profile would miss the GPU worker.

The following is a **proposed later-phase template**, not a command executed or an
end-to-end validated runner. Run inside WSL with the existing environment active,
from the repository; use a fresh run directory and check the port is free.

```bash
# RUN is a new unique output directory, allocated by the future runner.
nsys profile --trace=cuda,nvtx --cuda-trace-scope=process-tree \
  --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --kill=none --output="$RUN/traces/B8-graph" \
  python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 30149 --mem-fraction-static 0.669 \
  --chunked-prefill-size 2048 --disable-radix-cache --random-seed 42 \
  --cuda-graph-max-bs-decode 8 --decode-log-interval 1000000
```

The runner must additionally pin model revision and effective dtype/backend using
locally verified arguments. For B8 forced eager append `--disable-decode-cuda-graph`;
for B9 graph change cap to 16. Warmup is driven by the future client, then:

```bash
curl -f -X POST http://127.0.0.1:30149/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"activities":["CUDA_PROFILER"],"with_stack":false,"record_shapes":false}'
# Future runner submits exactly one fixed cohort and awaits completion.
curl -f -X POST http://127.0.0.1:30149/stop_profile
nsys export --type=sqlite --output="$RUN/summaries/trace.sqlite" \
  "$RUN/traces/B8-graph.nsys-rep"
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv "$RUN/traces/B8-graph.nsys-rep"
```

The listed profile options/report names were checked against installed help.
Whole-capture stats include the cohort's prefill and tail and are diagnostic only;
the primary parser must select/correlate the declared steady decode window.
Capture starts after graph construction. Confirm node records, worker PIDs, stream
IDs, API correlations and absence of dropped events before accepting results.
Graph-level granularity can test lower overhead, but cannot supply individual
kernel counts; missing node records must never be interpreted as fewer kernels.

Minimal NVTX plan: request setup/cohort on the client; prefill, decode forward,
sampling, and existing synchronization in a small version/hash-guarded observer
layer in an isolated SGLang copy only where necessary. No new per-step synchronize,
disk writes, stacks or shape collection. Existing layerwise NVTX is optional and
off for baseline. Client ranges alone do not label worker GPU execution. With
overlap, completion windows and GPU launch windows differ: retain forward IDs,
map worker API-to-GPU correlations, include trace margins, and reject ambiguous
attribution instead of clipping by client timestamps. If such mapping cannot be
validated, report only full-cohort diagnostics with the limitation.

Relevant primary documentation: [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
[analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html),
and [CUDA on WSL support](https://docs.nvidia.com/cuda/wsl-user-guide/).
Installed help/source, rather than newer documentation syntax, governs this setup.

## Measurement definitions and falsifiable hypotheses

For K verified fixed-B decode intervals, useful decode tokens = B×K. Report totals,
per batch-step (total/K), per useful token (total/(B×K)), and per request's measured
window (total/B); the last is not full-request service cost. For variable B use
the sum of useful B over accepted steps. Padding slots never enter denominators.
Cadence in ms/step is also a per-request token interval for a stable batch;
cadence/B is aggregate amortized time per useful token, not an individual ITL.

Extract runtime AND driver launch variants, graph launches, synchronization APIs,
API durations/counts, GPU kernel names/counts/grid/block dimensions/durations,
memcpy direction/bytes/time, and stream IDs. Distinguish host launch calls from
GPU kernel executions: one graph launch can execute many kernels. Avoid counting
both driver and runtime nesting as independent wall-time costs.

GPU busy time is the union of kernel/memcopy intervals across relevant streams
within the stated window; kernel-duration sum is a separate work statistic.
Idle time is window duration minus that union. This is traced application activity,
not measured whole-device inactivity or SM occupancy. Compute within-stream and
cross-stream gaps separately. Submission-to-start delay can include queueing and
dependencies; an empty timeline gap alone does not prove a CPU bottleneck.

| Hypothesis | Supporting comparison | Evidence against / limitation |
|---|---|---|
| H1 host launch overhead | Lower host launch API time and fewer submissions, comparable GPU work, reduced exposed gaps | Large cadence gap persists without corresponding launch/gap reduction |
| H2 fewer CPU-induced bubbles | Increased traced timeline density with CPU/API-to-GPU correlation | Dense eager timeline or dependency-driven gaps; density is not occupancy |
| H3 regime causes cliff | Same-B A/B and C/D gaps, smaller same-mode B changes | Gap disappears in controlled fresh runs or is dominated by drift |
| H4 memory behavior differs | Changed copy bytes/kernels; later targeted DRAM/L2 evidence | Nsight Systems alone cannot establish DRAM/L2 bandwidth or memory boundedness |
| H5 kernel paths differ | Changed kernel multiset, shapes/topology or duration distribution | Similar node-level workloads weaken this explanation; padded B9 remains a caveat |

RTX 4070 uses GDDR6X, not HBM. DRAM throughput, L2 behavior and achieved occupancy
are pending optional Nsight Compute measurements subject to permissions/support.
Do not infer these from API durations, memcpy totals or GPU utilization percentages.

Do not add host API sums, synchronization waits, summed GPU durations and memory
times into a claimed causal latency identity: these overlap, and memory operations
may already be represented in compute/waits. First compare them as descriptive
metrics. Only estimate an additive exposed critical-path decomposition after
correlation establishes non-overlapping components; leave attribution and residual
unavailable if that cannot be done. Keep unprofiled cadence and trace timing separate.

## H. Blockers, risks and acceptance gates

1. Current GPU driver query returns **581.08**. The historical environment records
   NVIDIA-SMI 610.43.02, KMD 610.62 and UMD 13.3. This discrepancy is real in the
   recorded outputs, not evidence of equivalent environments. Record a fresh full
   stack and verify compatibility in a bounded later smoke test; do not update the
   driver or reinterpret historical measurements. Re-establish unprofiled controls.
2. Process-tree tracing is available in help, but successful worker CUDA collection
   under WSL is untested. Do not enable system-wide profiling or risky fork tracing
   by default. No privileged configuration changes are needed for the initial plan.
3. Forced-decode-eager flag exists and source mapping is verified, but resolved
   runtime path, performance and memory behavior need actual validation. Maintain
   prefill, overlap and all other settings; identify any unavoidable differences.
4. B9 graph padding to 12 may alter kernel work. That is why B8 unpadded control
   is primary too. Same execution mode does not guarantee the same kernel shapes.
5. Prior B8 drift and intrusive older PyTorch profiling warn against one trace or
   instrumented timing as baseline. Windows display load, GPU clocks, thermal state,
   startup, compilation and WSL filesystem I/O can vary. Record them; write raw
   traces to Linux storage when practical and export compact summaries afterward.
6. Reproduction scripts have hardcoded ports, paths and archive writes. Extract
   reusable logic into new files and preserve hashes; never overwrite originals.
7. No new hardware performance conclusion is established by this audit. A missing
   counter, dropped records or ambiguous window is unavailable, not zero.

## Service-rate interpretation and next phases

Use μ_tok(B,m)=1000B/T_ms(B,m), with useful B and explicit mode, capture bucket and
context window in the manifest. At fixed policy, the practical curve can follow
graph-covered or eager branches when eligibility changes. A scheduler changing B
can therefore cross a non-smooth runtime boundary; the historical system already
supports this serving-level statement. GPU attribution remains unmeasured.

Retain the original conclusion: a classical state-dependent queue already explains
the principal queue separation. Making mode explicit does not establish a new OR
gap, hidden state, hysteresis or a scheduler optimization. Do not overwrite the old
service calibration with new-session timing. No CUDA microbenchmark is justified
until the SGLang traces identify a mechanism worth isolating.

## I. Proposed files and validation

Only this plan is added in Phase 2A. Later proposed additions:

- `profiling/run_profile.py`: validated CLI (`--batch-size`, `--mode graph|eager`,
  `--input-len`, `--output-len`, `--warmup`, `--repeats`, `--output-dir`), explicit
  coverage policy, dry-run, owned-process lifecycle, fresh directories and manifest.
- `profiling/configs/gpu_mechanism.json`: frozen workload, matrix, seeds and budgets.
- `profiling/instrumentation/`: minimal optional hash-guarded isolated observer.
- `profiling/parsers/`: SQLite schema inspection, correlations, interval unions,
  normalized metrics and unavailable-field handling.
- `profiling/analyze.py`: cohort/block statistics and evidence-driven figures.
- `tests/`: config/command checks, captured-schema fixtures, parser failures,
  overlapping interval unions, normalization, aggregation and manifests, CPU-only.
- `outputs/gpu-mechanism-study/<run-id>/`: manifest, results, traces, summaries,
  figures and report; study-level `REPORT.md` indexes immutable run folders.
- `.gitignore`: large Nsight reports/SQLite/raw traces, scoped to the new study.
- `README.md`: reproduction link only once the pipeline is implemented and tested.

Manifest must capture timestamp/timezone, Git commit and dirty state, source hashes,
GPU/driver/CUDA/tool versions, model revision, requested/effective server config,
exact client/server/profiler commands, actual B/mode/bucket, warmup and measurement
windows, stream/worker IDs, seed, observer state, trace path/size/hash, trace quality,
and profile overhead. Never commit giant trace files. No fabricated placeholder
measurements or decorative figures: absent results remain explicitly pending.

Phase 2B must validate unprofiled runner and controls before Phase 2C collection;
2D obtains bounded controlled traces; 2E parses and normalizes; 2F assesses
hypotheses; 2G reconnects to service rate; 2H is conditional; 2I validates and reports
MECHANISM IDENTIFIED, PARTIALLY EXPLAINED or INCONCLUSIVE as warranted.
Phase 2A stops here. The full existing validation suite has passed; historical
data are preserved, and no implementation or profiling run is included in this change.
