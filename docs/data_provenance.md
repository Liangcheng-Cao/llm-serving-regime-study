# Data provenance and columns

The source audit covered all 192 files in five completed experiment archives (about
66 MB). Sixty-two files contained machine-specific path patterns. No obvious credential
pattern was detected. Originals remain private and unchanged. This package uses an
allowlist of numeric fields, rewritten notes and adapted standalone simulation logic;
it does not copy server logs, source manifests, private Git metadata or GPU harnesses.

| Public file | Frozen source / meaning |
|---|---|
| `data/inputs/calibration_intervals.csv` | Step 5 calibration; 30 cohorts × 70 intervals, milliseconds |
| `data/inputs/initial_states.json` | Step 5 reconstruction; ten remaining-token vectors and lifecycle-check count |
| `data/inputs/frozen_simulation_reference.json` | Step 5, 60 deterministic summaries; validation reference |
| `data/processed/service_curve.csv` | Rebuilt from intervals; T_ms, μ_tok/s and μ_req/s=μ_tok/s÷128 |
| `data/processed/step2_summary.csv` | Four causal cells; mean/median/p95 interval ms, cohort CV, useful tokens/s |
| `data/processed/step2_repetitions.csv` | All 40 instrumented cohorts; rep 1–5 block 1, 6–10 block 2 |
| `data/processed/step2_observer_pairs.csv` | Forty observer ON/OFF client-proxy pairs |
| `data/processed/step3_summary.csv` | Ten pulse trials; post window 8.05–20 s; empty recovery is right-censored >12 s |
| `data/processed/pulse_series.csv` | All five pairs sampled at 0.1 s, truncated at arrival stop t=20 s |
| `data/processed/step4_summary.csv` | Twenty 60-s trials, N/B means, path fractions, A/C and residency |
| `data/processed/step4_blocks.csv` | Every non-overlapping five-second accounting block |
| `data/processed/model_comparison.csv` | Ten primary cap8 real/model comparisons, frozen Step 5 |
| `data/processed/simulation_summary.json` | Recreated from simulator, all 60 scenarios |

`cap` is configured decode graph maximum, `B`/`batch` actual useful decode batch,
`N` client outstanding count in measurements and active count in the model. Times
are seconds unless named `_ms`; fractions lie in [0,1], percentages are named `_pct`.
`rep` identifies a repeated trial or paired initial state, not an independent machine.
Model `completions` includes initially present requests; model sojourn statistics cover
the 600 future arrivals only. `removed` denotes the extrapolated cliff-removal law.

Offline scripts overwrite only derived service-curve/simulation files and figures.
They do not rerun or alter experiments. Frozen reference comparisons use numerical
tolerance, so floating-point last digits can differ without changing the reported metric.
The four figures are descriptive; cohort dots or repetition ranges are not confidence intervals.
