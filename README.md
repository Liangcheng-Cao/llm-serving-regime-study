# SGLang Decode Execution Regimes

At the same useful batch size, changing the decode execution regime changes serving cadence by approximately **3.24×**. This study connects a controlled SGLang experiment to GPU trace structure: nearly the same observed computation, but many more individual launches and much larger inter-kernel gaps under eager decode. The conclusion is **Level-2 strong mechanism evidence**, with explicit profiling and attribution limits.

## Key Result

**B8 is the primary control:** graph bucket8 versus forced decode eager, input128/output128. Serving values are unprofiled; GPU/API values describe observed profiled windows.

| Metric | Graph | Eager | Comparison direction |
|---|---:|---:|---|
| Decode interval (unprofiled) | 3.117 ms | 10.114 ms | Eager/graph 3.244× |
| Decode throughput (unprofiled) | 2566.9 tok/s | 791.2 tok/s | Graph/eager 3.244× |
| Observed GPU-work union | 352.906 µs/token | 362.619 µs/token | Eager/graph 1.028× |
| Observed inter-kernel gaps | 40.955 µs/token | 1158.017 µs/token | Eager/graph 28.276× |
| Observed individual runtime launches | 1.750 /token | 29.125 /token | Eager/graph 16.643× |
| Observed GPU busy fraction | 89.74% | 23.86% | Graph +65.88 percentage points |
| Kernel-name sequence similarity | 98.442% shared comparison | — | Graph versus eager; exact LCS |

Decode-interval means are **3.117 ± 0.053 ms graph** and **10.114 ± 0.187 ms eager** (sample SD across three restart means). Throughput averages are computed separately, so their ratio need not equal the ratio of mean intervals. GPU quantities use 70×B useful token equivalents, not exact request-level attribution.

![Unprofiled serving comparison](outputs/gpu-mechanism-study/phase2e/figures/A_service_effect.png)
*Separate unprofiled starts establish the serving effect: three starts per arm, three measured cohorts per start. B8 is primary; B9 graph uses padded bucket12. Points show restart means.*

## Why This Matters

A batch-only service model, **μ(B)**, misses execution policy. Use **μ(B,m)**, where B is useful batch and m is execution regime. Crossing a graph/eager boundary can change service rate sharply even when useful work changes little.

## Experimental Design

- **B8 control:** graph bucket8 versus forced decode eager, both cap8.
- **B9 practical case:** natural eager at cap8 versus graph at cap16, verified bucket12; cadence is 9.952 versus 3.180 ms (3.130×).
- **Serving measurements:** three independent unprofiled starts per arm, three measured cohorts after three warmups.
- **Structural evidence:** three independent Nsight traces per arm; marked steady decode, worker verification, structured SQLite parsing and interval unions across streams.
- **Confirmation:** output256 adds two traces and one unprofiled start per B8 arm. No new data were collected during packaging.

System: RTX 4070 12GB, SGLang 0.5.19, Qwen2.5-0.5B-Instruct, BF16/FlashInfer. [Full environment and reproduction](docs/REPRODUCIBILITY.md).

## Main Evidence

All 25 B8 kernel names are shared; sequence similarity is 98.442%. Dominant GEMM time differs by about 1.1%. Eager has 16.643× more individual runtime launches and 28.276× larger recorded gaps. Every observed eager gap successor matches an individual launch through a worker-scoped correlation ID. This association does not assign every gap to CPU execution.

## Mechanism

The evidence strongly supports **host/runtime launch and orchestration structure as the dominant mechanism associated with much larger observed GPU execution gaps in eager decode**. Claim level: **2**, not exact causal decomposition. Graph synchronization waits largely overlap GPU work; summed CUDA API duration is not a measure of overhead that leaves the GPU idle.

## Scheduler / OR Implication

Calibrate service curves by regime, bucket and workload. Batching and queueing models should preserve execution boundaries rather than interpolate smoothly through them. This study introduces no scheduler algorithm. The earlier queueing investigation found that a classical state-dependent service model reproduced the main qualitative behavior; its **no-go for a new OR gap** remains intact. [Historical experiments](experiments/README.md).

## Limitations

One small model/device configuration; fixed synthetic requests; three primary traces per arm. Nsight perturbs eager cadence by about 20.24% at B8 versus 0.95% for graph. CUDA/NVTX completeness warnings remain. Gaps can include copies or dependency delays; their exact causes are unresolved. B9 changes useful batch and padding together. No Level-3 causal claim, hardware-counter conclusion, or universal speedup is made.

## Reproduce

CPU-only validation from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python src/validate.py
```

Committed summaries support inspection without GPU software. Event-level reanalysis requires retained SQLite files, which are not in Git. [Reproduction guide](docs/REPRODUCIBILITY.md) separates summary-only analysis, event analysis and fresh GPU collection into new output locations.

## Repository Structure

| Path | Purpose |
|---|---|
| `profiling/` | Guarded SGLang runner, Nsight extraction and mechanism analysis |
| `tests/` | Timing, interval accounting, correlation and sequence checks |
| `data/`, `src/` | Frozen historical inputs and classical queueing analysis |
| `experiments/`, `docs/` | Historical protocols, abstract, summaries and reproduction |
| `outputs/gpu-mechanism-study/` | Final report and preserved phase evidence |
| `figures/` | Historical service-curve and queue-response figures |

## Research Artifacts

- [Abstract](docs/ABSTRACT.md) · [Executive summary](docs/EXECUTIVE_SUMMARY.md) · [Final technical report](outputs/gpu-mechanism-study/FINAL_REPORT.md)
- [Phase 2D controlled campaign](outputs/gpu-mechanism-study/phase2d/REPORT.md) · [Phase 2E mechanism analysis](outputs/gpu-mechanism-study/phase2e/REPORT.md)
- [Aggregate metrics](outputs/gpu-mechanism-study/phase2d/aggregate_metrics.csv) · [Hypothesis evidence matrix](outputs/gpu-mechanism-study/phase2e/hypothesis_evidence.csv)
- [Curated figures and standalone captions](docs/FIGURES.md) · [Publication audit](docs/PUBLICATION_AUDIT.md)

Code and released study artifacts are provided under the [MIT License](LICENSE).
