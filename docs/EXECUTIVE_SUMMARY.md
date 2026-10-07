# Executive summary

## Finding

SGLang's decode execution regime can matter more than a small change in useful batch size. On the tested RTX 4070 with Qwen2.5-0.5B-Instruct, holding useful batch at eight and enabling CUDA Graph replay reduces unprofiled decode cadence from **10.114 ± 0.187 ms to 3.117 ± 0.053 ms**. Useful decode throughput rises from **791.2 to 2,566.9 tokens/s**. These are separate fixed-cohort measurements, not end-to-end production throughput predictions.

## Evidence

The primary comparison holds batch, model, synthetic workload, dtype and backend fixed. Three independent server starts per arm estimate serving behavior, while three separate Nsight traces characterize execution structure. A practical B9 comparison and a smaller output256 set check whether the pattern persists.

Observed GPU-work union differs by only about 2.75% eager/graph. All 25 B8 kernel names are shared, and kernel-name sequence similarity is 98.442%. Eager nevertheless has 16.643× more individual runtime launches and 28.276× larger inter-kernel gaps. Correlation IDs connect each observed eager gap successor to an individual launch. Graph's longer synchronization calls largely overlap GPU execution, explaining why summed API duration is not a useful proxy for harmful launch overhead.

## Interpretation and practical relevance

The evidence strongly supports repeated host/runtime launch and orchestration structure as the dominant mechanism associated with eager's larger GPU gaps. This is **Level 2: strong mechanism evidence**. It is not an exact causal decomposition: driver work, dependencies, host scheduling and profiler effects remain entangled.

For inference engineers, the practical lesson is to record execution regime alongside batch size. A service model μ(B,m) can preserve graph/eager boundaries that a smooth μ(B) model misses. Queueing simulations and batching decisions should use measured regime-specific rates and avoid unvalidated extrapolation. The earlier classical-model result remains: this phenomenon does not establish a novel OR problem or justify a new scheduler algorithm by itself.

## What the project demonstrates

The work combines controlled experiment design, execution-path verification, reproducible instrumentation, machine-readable GPU trace analysis, interval accounting across streams, event correlation, sequence comparison and queueing-model interpretation. It retains invalid-run policy, provenance, tests and explicit limits instead of inferring mechanisms from profiler screenshots.

## Scope

One GPU, one small model, fixed equal-length synthetic requests and limited independent restarts constrain generalization. Nsight perturbs eager substantially more than graph, and trace completeness is not certified. B9 padding cost is not isolated. No microbenchmark, custom kernel, Nsight Compute study or new scheduler is claimed.

See the [final report](../outputs/gpu-mechanism-study/FINAL_REPORT.md) and [reproduction guide](REPRODUCIBILITY.md).
