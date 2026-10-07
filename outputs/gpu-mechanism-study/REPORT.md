# Phase 2B/2C: profiling pipeline validation

Date: 2026-10-07 (Asia/Shanghai). **READY WITH LIMITATIONS** for Phase 2D controlled
trace collection. Not ready for exhaustive event-count claims or a causal latency
decomposition until the limitations below are addressed. This is a pipeline
validation report, not a GPU mechanism interpretation or a full profiling campaign.

## Scope and implementation

Continued the existing repository from `8f29184`. Added:

- `profiling/run_decode_experiment.py`: fixed-cohort client, owned server process,
  fresh run directories, warmup, verification, serving metrics and manifests.
- `profiling/run_nsys_profile.py`: bounded server-process-tree capture, CLI export,
  structural validation, temporal normalization and retained trace metadata.
- `profiling/core.py`, `instrument.py`, `observer.py`: explicit policies, statistical
  contracts, isolated hash-guarded instrumentation and runtime evidence.
- `profiling/parsers/nsys.py`: supported CSV/SQLite parsing and trace quality checks.
- `profiling/configs/validation.json`, `profiling/README.md`: frozen validation
  design and executable usage; `summarize_validation.py`: offline validation summary.
- `tests/test_gpu_mechanism.py`: CPU-only tests.
- This new output tree: manifests, fresh results, compact trace exports and quality
  reviews. `.gitignore` excludes raw logs, SQLite, full trace CSV and Nsight reports.
  Root README links the extension. Historical implementation/data/figures are unchanged.

## Exact architecture and verified controls

The WSL client starts its own loopback SGLang server. Every server command pins:
Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, BF16, FlashInfer, seed 42,
memory fraction 0.669, chunked prefill 2048 and disabled radix cache. TP/DP/PP remain
1, overlap stays enabled and prefill graphs stay enabled. Commands include
`--cuda-graph-max-bs-decode 8` or `16`; only the forced-eager arm adds
`--disable-decode-cuda-graph`. Resolved configuration checks passed for all eight runs.

Each request i contains exactly 128 copies of token ID 1000+i, temperature zero,
`ignore_eos=true`, and 128 or 256 output tokens. Prompt/output counts are checked.
Three full warmup cohorts precede measurement. Every unprofiled run has three
measured cohorts; each trace has one. The primary window remains completion
endpoints 10→80 (70 intervals). No loading, startup, warmup, prefill or teardown
time enters the server cadence statistic. Cohort E2E is separately labeled and
does include prefill. The 256-token runs use the same early window, not a late-window
context-length experiment.

A unique Linux temporary copy of SGLang receives two small, source-hash-guarded
observer insertions. One records time, useful B, forward ID and the returned graph
flag immediately after the existing copy-completion synchronization. The other
records the actual selected backend replay bucket. No new per-step GPU synchronize
or file writes are added. Cohort-tail records are retained locally. An unexpected
mode/B, missing bucket or incomplete/nonconsecutive window invalidates the run.
Installed source hashes remained unchanged.

The requested policy and observed mode are separate manifest fields. B9/cap8 was
observed eager; B9/cap16 replayed bucket 12; B8/cap8 replayed bucket 8; B8 forced
decode eager was observed eager. Mode verification does not rely on a launch flag.

Baseline includes the observer and two worker-side NVTX calls per cohort; it is
unprofiled, not instrumentation-free. Client token 11→81 timing is an auxiliary
streaming proxy, never a count of transport chunks.

## Fresh unprofiled sanity results

Mean completed-decode cadence across three cohort means; SD is between cohorts
within one server start. This is not an independent-restart uncertainty estimate.

| Arm | B / cap / decode policy | Output tokens/request | Cadence mean ± SD (ms) | Useful decode tok/s | Cohort E2E mean (s) |
|---|---|---:|---:|---:|---:|
| A | 9 / 8 / natural eager | 128 | 10.8517 ± 1.1437 | 829.36 | 1.4107 |
| B | 9 / 16 / graph, bucket 12 | 128 | 3.1911 ± 0.0171 | 2820.33 | 0.4317 |
| C | 8 / 8 / graph, bucket 8 | 128 | 3.1271 ± 0.0094 | 2558.25 | 0.4211 |
| D | 8 / 8 / forced decode eager | 128 | 9.7676 ± 0.1728 | 819.04 | 1.2920 |
| A | 9 / 8 / natural eager | 256 | 10.2079 ± 0.1696 | 881.67 | 2.6365 |
| B | 9 / 16 / graph, bucket 12 | 256 | 3.1792 ± 0.0145 | 2830.94 | 0.8414 |

The historical qualitative ordering reproduces: B9 eager/graph cadence ratio is
about **3.40×** at 128 outputs and **3.21×** at 256. Historical 2.98× is neither a
target nor an exact replication criterion. B8 forced eager is also slower in this
small set. The A/128 cohort CV is 10.54%, so a precise new effect estimate is not
justified. B8 arms were validated at 128 outputs only. There were six unprofiled
server runs, 18 measured cohorts and 18 excluded warmups.

Current environment: RTX 4070, 12,282 MiB; driver 581.08, NVIDIA-SMI 580.76.07,
reported driver CUDA support 13.0; Python 3.12.14, PyTorch 2.13.0+cu130, SGLang
metadata 0.5.19; toolkit 13.3; Nsight Systems 2026.1.3. The historical driver stack
differs. These values are stored as fresh-environment evidence and were not merged
into historical service-rate tables.

See [results.csv](results.csv) for cohort SD/CV, medians and throughput, and
[validation_summary.json](validation_summary.json) for resolved configuration checks.

## Minimal Nsight pair

Nsight wraps `python -m sglang.launch_server` with:

```text
--trace=cuda,nvtx --cuda-trace-scope=process-tree
--sample=none --cpuctxsw=none --cuda-graph-trace=node
--capture-range=cudaProfilerApi --capture-range-end=stop
--kill=none --output=<unique local trace prefix>
```

After warmup, the client calls `/start_profile` with
`{"activities":["CUDA_PROFILER"],"with_stack":false,"record_shapes":false}`,
sends one measured cohort, then awaits `/stop_profile` before terminating only
its owned server process group. The report is exported before analysis. This
empirically captured the server GPU worker, not merely the HTTP client.

| Check | B9 natural eager | B9 graph, bucket 12 |
|---|---:|---:|
| Worker PID from observer and NVTX | 1355 | 638 |
| Local report size (bytes) | 2,801,941 | 1,509,952 |
| Observed graph-launch calls in completion window | 0 | 70 |
| Observed temporally contained kernels | 23,169 | 23,520 |
| Observed runtime API records in that window | 40,724 | 5,670 |
| Observed kernels without a worker API correlation | 0 | 0 |
| Nsight CLI report/SQLite opening and export | Pass | Pass |
| Steady measurement NVTX range in worker | Pass | Pass |
| Exhaustive event collection established | **No** | **No** |

Kernel/API counts above validate that usable events exist; they are not a mechanism
comparison. Each full capture includes prefill and tail, while the NVTX range
`GPU_MECH_MEASUREMENT_COMPLETION:measure-3` identifies the stable completion window.
The parser filters fully contained events by the verified PID. GPU/runtime overlap
means the window is not a causal assignment of GPU work to 70 decode steps.
Boundary-crossing events are excluded from these temporal diagnostics.

Raw reports are preserved in `traces/trace-a-128.nsys-rep` and
`traces/trace-b-128.nsys-rep`, locally ignored by Git. Absolute paths, sizes and
SHA-256 hashes appear in the corresponding manifests and trace summaries.
No GUI is required: successful CLI export verifies that both reports can be read.

## Profiling perturbation

| B9 arm | Unprofiled cadence (ms) | Profiled cadence (ms) | Apparent cadence change | Apparent E2E change |
|---|---:|---:|---:|---:|
| Natural eager | 10.8517 | 12.3569 | +13.87% | +14.95% |
| Graph, bucket 12 | 3.1911 | 3.2166 | +0.80% | +2.08% |

These are three baseline cohorts versus one trace cohort, from different server
starts. They quantify the observed perturbation, which includes drift; they are
not precise isolated profiler-overhead estimates. Retain unprofiled controls in
Phase 2D. Do not substitute trace latency for the unprofiled service curve.

## Extractable metrics and quality limits

Installed-supported exports are `cuda_api_sum`, `cuda_gpu_kern_sum`,
`cuda_gpu_trace` and `nvtx_pushpop_trace`, plus SQLite. The pipeline extracts
observed API counts and summed durations, named synchronization counts/time,
kernel counts/summed durations/distribution, dominant kernels, graph launches,
worker/stream IDs and marked window bounds. CUDA API CSV totals may mix driver
and runtime records and overlap; SQLite runtime metrics are separately named.
Summed kernel duration is work time, not union GPU busy time. Memory counters,
occupancy, GPU idle attribution and latency decomposition are not implemented.

Normalization utilities support per useful generated token, decode interval and
request, with explicit work counts. Window denominators are B×70, 70 and B;
padding never contributes useful tokens. Temporal normalized counts remain
diagnostic because they are not causally aligned to exact decode work.

Nsight diagnostics require explicit limitations:

1. Hardware trace is unavailable here; Nsight used software-instrumented tracing.
2. Both worker traces warn that not all CUDA and NVTX events might have been
   collected. All 70 expected graph launches are observed and all observed window
   kernels correlate, but this cannot detect missing kernel/API pairs. The
   produced/collected diagnostic counters have different totals; their difference
   is not automatically a dropped-kernel count.
3. Unified Memory tracing is unavailable on this setup. It is not an initial
   metric target and must not be reported as zero transfer activity.
4. Other process-tree members report no CUDA activity; those warnings are distinct
   from the verified worker's warnings and remain retained in the diagnostics.

The later offline quality review stores `completeness_proven=false` and explicitly
labels structural validation. Initial run manifests retain their original structural
`valid` result; [eager quality review](summaries/trace-a-128/quality_review.json) and
[graph quality review](summaries/trace-b-128/quality_review.json) restrict its meaning.
Errors in the worker invalidate future traces; warnings remain visible.

NVIDIA's [Nsight guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
documents flushing on cudaProfilerStop; installed help confirms it defaults true.
In a [separate newer-version case](https://forums.developer.nvidia.com/t/nsight-systems-2026-5-1-produced-112-collected-100-warning-despite-complete-coverage-of-32-kernels/384749),
NVIDIA attributed the completeness warning to a forced flush. That is a possible
explanation, not proof for these 2026.1.3 traces. No extra profiling runs were
performed to chase the warning. Resolve it or establish independent expected-event
coverage before using exhaustive counts in mechanism claims.

## Validation, provenance and stopping point

- 17 CPU-only tests passed: configuration, modes, manifests, IDs, invalid-run
  exclusion, source anchors, observer tail handling, normalization, statistics,
  CSV parsing, SQLite worker/window/correlation checks and diagnostic handling.
- The complete existing suite, `python src/validate.py`, passed: analytic case,
  60 frozen scenarios, conservation, causal contrast and service cliff.
- SHA-256 comparison confirmed all 205 baseline historical/archive/data files
  unchanged. No old scripts were rerun in place.
- All eight new runs passed execution-policy checks; no failed run was discarded.
  Only two profiling runs were performed, both B9/128. Full campaign not started.
- Early bootstrap runs honestly record a dirty development tree and their then
  current commit; observer and installed-source hashes are retained. Later runner
  versions additionally hash all pipeline sources and validate effective settings
  during execution. Offline resolved-config checks cover all eight saved runs.
  Do not relabel these as all produced by the final commit.
- Implementation commits: `c3543ef` (runner), `23b6d6b` (Nsight/tests),
  `e29533f` (configuration/quality checks and usage). A following documentation/data
  commit records this report and validation artifacts. Nothing was pushed.

**READY WITH LIMITATIONS** means the four-arm runner and bounded Nsight collection
can support controlled Phase 2D work with unprofiled controls. First address trace
completeness and completion-to-GPU attribution as acceptance gates; do not begin
a large campaign that assumes those questions are already settled. The next
scientific interpretation, full campaign, kernel optimization and microbenchmark
remain outside this completed Phase 2B/2C scope.
