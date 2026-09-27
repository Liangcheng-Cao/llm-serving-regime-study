# Classical state-dependent queue test

**Question.** Do these observations require a new OR / queueing abstraction?

**Method.** Measure a fresh cap8 service curve at B={1,2,4,6,8,9,10,12,14,16},
three cohorts per B and 70 fixed-window intervals per cohort. The three orders are
ascending, descending and a fixed permutation, with shape warmup excluded.
Maximum cohort-mean CV is 3.253%; no extra measurements are added.
Compute aggregate useful-token rate μ_tok(B)=1000 B/T_ms(B), then divide by 128
to obtain a capacity proxy in fixed-length requests/s. This proxy is not measured
end-to-end throughput including prefill. It is a separate measurement from Step 2.

The queue sets B=N and shares aggregate service equally: dw_i/dt=−μ_tok(N)/N.
Service is piecewise linear in aggregate rate between measured points.
New jobs require 128 tokens and arrive every 0.1 s for 60 s. There are no fitted
parameters, hidden modes, history terms, or explicit scheduler in the service law.
History labels only identify initial states; remaining work is ordinary queue state.

Reconstruct remaining work from Step 4: prefill produces one token, then exactly 127
completed decode iterations per request. All 7,028 lifecycle checks passed before
extracting the ten initial vectors. Also test uniform residual-work midpoints and
all-128 initial work. Across five input pairs and two service laws this gives 60
deterministic scenarios, not 60 independent statistical trials.

**Result.** Empirical-initial-state model LOW mean N=4.002–4.003 and HIGH=13.109–13.127,
against real 4.229–4.315 and 13.379–14.339. Future-job median model sojourn is 0.400 s
and 1.311 s; real server-residency medians are 0.417–0.424 and 1.323–1.436 s.
These are related but differently bounded metrics. The model underpredicts mean N by
5.3–7.2% (LOW) and 2.0–8.5% (HIGH); no corrective parameters are fitted.
Uniform and empirical primary cases remain within the measured B range.

For the cliff-removal counterfactual only, continue the B=8 per-job speed above 8:
μ_tok(B)=B/T(8). Keep arrivals and initial work fixed. Empirical HIGH returns to
B≤8 in 0.232–0.261 s. This extrapolation is not a measured cap16 service curve.
All-128 HIGH sensitivity peaks at B=31–33 under the original law and spends
2.59–3.02 s outside the measured range, where aggregate μ is held at μ(16).
Its early transient therefore depends on an unvalidated extension.

**Verdict: NO-GO FOR NEW OR GAP.** A minimal classical queue reproduces the principal
behavior. This does not prove that it is the unique explanation or an exact system model.
The systems mechanism remains substantial; no further screening experiment is needed.

[Model comparison](../data/processed/model_comparison.csv) · [Frozen references](../data/inputs/frozen_simulation_reference.json).
