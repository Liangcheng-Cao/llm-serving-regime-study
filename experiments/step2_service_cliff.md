# Same useful batch, different execution path

**Question.** Does graph coverage cause a service cliff at fixed actual B?

**Method.** A 2×2 design crosses graph ceiling {8,16} with actual useful batch {8,9}.
All cohorts use 128 input and 128 output tokens with EOS ignored. Each cell has ten
instrumented cohorts and ten observer-off proxy cohorts. Two server starts per ceiling:
repetitions 1–5 run cap8 then cap16; repetitions 6–10 reverse that order.
The sample was extended from five to the predeclared maximum of ten because the first
negative control drifted by about 15%; this was not an indefinitely repeated search.
Stable completion endpoints 10 through 80 yield 70 intervals per cohort.
At cap16, B=9 uses padded graph bucket 12, not bucket 16. B=8 uses bucket 8 in both arms.
Native path and actual-batch checks passed for all formal observations.

**Result.** The pooled mean completed-decode interval (not kernel-only time) is:

| Actual B | cap8 ms | cap16 ms | Relative difference |
|---:|---:|---:|---:|
| 8 | 3.8377 | 3.5947 | +6.76% |
| 9 | 10.8918 | 3.6551 | +197.99% |

The B=9 ratio is 2.98×. The multiplicative difference-in-differences is 2.791×
(log contrast 1.02649). Block-specific B=8 changes were +15.16% and −1.71%;
B=9 changes remained +201.77% and +194.20%. The conditional cohort-bootstrap
B=9 increase interval was 193.22–202.90%; it is not a server-restart population interval.

Observer ON/OFF mean client-proxy differences by (cap,B) were −4.53%, −1.50%,
−0.05%, and +0.29%. Their paired SDs were 8.81%, 8.73%, 1.93%, and 1.85%.
The observer-off B=9 treatment increase was 205.82%. Shared hook checks remain
in the off arm, so this does not establish zero instrumentation overhead.

**Verdict: PASS**, with negative-control drift reported rather than suppressed.
The path intervention has a large same-B effect; the design does not isolate GPU kernel time.

## Every instrumented cohort

Values below are per-cohort mean intervals. Rep is a cohort, not an independent server restart.
The CSV additionally contains cohort medians, p95, useful token rate and client proxy.

| cap | B | rep | mean ms |
|---:|---:|---:|---:|
| 8 | 8 | 1 | 4.486428 |
| 8 | 8 | 2 | 3.826243 |
| 8 | 8 | 3 | 4.463710 |
| 8 | 8 | 4 | 4.053742 |
| 8 | 8 | 5 | 3.957213 |
| 8 | 8 | 6 | 3.534188 |
| 8 | 8 | 7 | 3.581315 |
| 8 | 8 | 8 | 3.516906 |
| 8 | 8 | 9 | 3.456611 |
| 8 | 8 | 10 | 3.500530 |
| 8 | 9 | 1 | 11.170510 |
| 8 | 9 | 2 | 10.688423 |
| 8 | 9 | 3 | 11.364349 |
| 8 | 9 | 4 | 10.942969 |
| 8 | 9 | 5 | 11.094338 |
| 8 | 9 | 6 | 10.730432 |
| 8 | 9 | 7 | 11.029166 |
| 8 | 9 | 8 | 10.688985 |
| 8 | 9 | 9 | 10.542781 |
| 8 | 9 | 10 | 10.666285 |
| 16 | 8 | 1 | 3.590704 |
| 16 | 8 | 2 | 3.629505 |
| 16 | 8 | 3 | 3.612762 |
| 16 | 8 | 4 | 3.601179 |
| 16 | 8 | 5 | 3.616982 |
| 16 | 8 | 6 | 3.606969 |
| 16 | 8 | 7 | 3.543621 |
| 16 | 8 | 8 | 3.643909 |
| 16 | 8 | 9 | 3.554169 |
| 16 | 8 | 10 | 3.547457 |
| 16 | 9 | 1 | 3.608506 |
| 16 | 9 | 2 | 3.639181 |
| 16 | 9 | 3 | 3.675841 |
| 16 | 9 | 4 | 3.699928 |
| 16 | 9 | 5 | 3.688573 |
| 16 | 9 | 6 | 3.753737 |
| 16 | 9 | 7 | 3.629197 |
| 16 | 9 | 8 | 3.645760 |
| 16 | 9 | 9 | 3.615763 |
| 16 | 9 | 10 | 3.594337 |

[All cohort metrics](../data/processed/step2_repetitions.csv) · [Observer pairs](../data/processed/step2_observer_pairs.csv)
