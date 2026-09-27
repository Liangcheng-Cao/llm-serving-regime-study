# Matched future arrivals over 60 seconds

**Question.** Does LOW/HIGH separation persist beyond completion of the pulse itself?

**Method.** Five repetitions × two ceilings × LOW/HIGH gives 20 trials.
LOW starts from ordinary 10-request/s traffic. HIGH uses the same +6 pulse history.
The comparison origin for HIGH is the first 100-ms grid point at least 0.5 s after
the final pulse client completion; the pulse is absent from every comparison window.
Future traffic is exactly 600 scheduled arrivals at t=0,0.1,...,59.9 s in both histories.
Actual paired dispatch differences are at most 3.061 ms. Repetitions 1–3 use cap8 then
cap16 and LOW then HIGH; 4–5 reverse both orders. Trials drain between runs.

Convergence is the first passing 5-s window (ends 5,6,...,60): mean B difference ≤1,
mean N difference ≤2, HIGH fraction B>8 <5%, and HIGH eager fraction <5%.
This is an operational finite-window definition. A reported 5 s is the earliest
testable window end, not a measurement of an exact 5-second recovery time.

**Result.** At cap8, LOW mean N is 4.229–4.315 and HIGH is 13.379–14.339.
LOW is entirely graph-covered and HIGH entirely eager under the step-held path metric.
HIGH has 600 or 601 completions against 600 arrivals: net drift is 0 or −1/60 request/s.
Small positive and negative five-second block changes remain in the released block table;
there is no sustained downward trend. None of five cap8 pairs converges within 60 s.
At cap16, every pair passes all 56 windows, first passing at 5 s, and both histories
remain below the boundary. All safety guards and N-end minus N-start = A−C checks pass.

**Verdict: PASS for long-lived operating-state separation over the tested horizon.**
Initial queue counts and remaining-work distributions differ. The result does not
demonstrate a hidden history variable, equilibria, bistability, or infinite persistence.

[Run summaries](../data/processed/step4_summary.csv) · [Every five-second block](../data/processed/step4_blocks.csv).
