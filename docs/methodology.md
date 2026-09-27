# Methodology

## Tested system

RTX 4070, 12,282 MiB reported memory; WSL2; Python 3.12.14; PyTorch 2.13.0+cu130;
SGLang package metadata 0.5.19; Qwen/Qwen2.5-0.5B-Instruct at revision
`7ae557604adf67be50417f59c2c2f167def9a775`; BF16; FlashInfer.
TP/DP/PP=1; overlap enabled; speculative decoding and LoRA off; radix cache off;
memory fraction 0.669; chunked prefill 2048; seed 42. Installed decode graph defaults
capture [1,2,4,8], padding enabled. Changing graph maximum to 16 includes bucket 12.
The prefill graph uses separate aggregate-token buckets and does not remove the decode cliff.

## Fixed workload and timing

Except the 48-output-token Step 1 sanity audit, requests use 128 input and 128 output
tokens, `ignore_eos=true`, temperature zero and streaming responses. Step 2 assigns
repeated token ID 1000+i to request i; assignments are held fixed across its treatment
arms. Later queue experiments use 128 copies of token ID 1000. These are controlled
synthetic token workloads, not a natural-language quality evaluation.

Service measurements use completion endpoints 10–80, exactly 70 intervals. Time is
recorded immediately after an already-existing copy-completion synchronization;
the observer adds no GPU synchronization. Records accumulate in memory and flush at
the tail. The measured cadence includes overlapped scheduler/runtime work and is not
kernel-only latency. Useful rate uses actual B, excluding graph-padding slots.
Repeated intervals within a cohort are not independent experimental replications.

## Intervention and queue observation

Graph coverage is manipulated while holding actual useful B fixed. Both actual B
and the returned graph-execution flag are verified; configured ceilings alone are
insufficient evidence of execution. Observer-off trials disable record construction,
but retain common hook checks. Client streaming token timestamps supply the paired
proxy. This checks perturbation approximately, not proof of zero overhead.

Open-loop arrivals follow paired fixed schedules, independent of completions. Actual
dispatch jitter is retained in the original audit and summarized in experiment notes.
N counts client-sent minus client-completed requests; B counts active useful decode
requests. They are distinct quantities in the system, though the model assumes B=N.
The internal waiting-queue counter was unavailable and is not presented as measured.
Every accounting window verifies N_end − N_start = A − C, including pre-existing jobs.
Server residency starts at first prefill scheduling and ends at server completion.

Batch/path time fractions step-hold completed-decode observations, including intervening
prefill gaps; they are not GPU activity fractions. Pulse figure samples are for display,
not the source of exact duration or mean statistics. Step 4 convergence is defined in
its [experiment note](../experiments/step4_long_horizon.md).

## Model and reproducibility boundary

The deterministic event-driven processor-sharing queue advances to the next arrival,
completion or 60-s horizon. It drains afterward to measure all 600 future-job sojourns.
Measured aggregate service is linearly interpolated without smoothing. Initial work
comes from verified lifecycle reconstruction, with generic sensitivity initializations.
No label enters the service function. Conservation and an analytic constant-per-job
service case are tested, and all 60 outputs are compared with frozen references.

This repository reproduces service-curve analysis, queue simulation and four figures
offline. It does not ship or claim a tested GPU benchmark harness. Raw logs, model
weights, caches, machine-specific paths and installed package copies are excluded.
