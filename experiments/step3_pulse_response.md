# Finite pulse and queue amplification

**Question.** Can a small finite pulse move the system into a sustained high-batch state?

**Method.** A single rate calibration selected 10 requests/s (baseline B≈4.68),
and one pilot selected a +6-request pulse. Five paired formal trials per ceiling
then use 20 seconds of open-loop traffic: warmup 0–4, baseline 4–8, pulse at 8 seconds,
and a frozen post window 8.05–20 seconds. Pulse requests are spaced 2 ms apart.
Each trial submits 206 requests, then drains. All formal pulses finish sending before 8.05.
The first three repetitions use cap8 then cap16; the last two reverse the order.
Paired scheduled traces are identical; maximum actual paired dispatch difference is 1.8 ms.
Safety limits (48 outstanding, 512 requests, 45 s) were not reached.

**Result.** Baseline median B=5 in every trial. Post-pulse peak B is 21–22 at cap8
and 11–12 at cap16. Cap8 stays above B=8 for essentially all 11.95 post-window seconds;
cap16 spends 0.463–0.502 s there. Recovery means first B<8 sustained for two seconds:
cap16 recovers in 0.513–0.552 s from pulse time; cap8 is right-censored beyond 12 s.
Cap8's pulse requests have all completed by 9.531 s at the latest, yet the high state
persists another 10.47–10.50 s until arrivals stop. Late B is approximately 14.4–14.7.

Post-window flow accounting is exact: N starts at 11, A=119 in both arms;
cap8 C=116 and N ends at 14; cap16 C=126 and N ends at 4.
All requests completed and actual execution-path mismatches were zero.

**Verdict: PASS.** A completed finite pulse leaves a sustained high operating state
over this window. This is not evidence of unbounded backlog growth or a hysteresis proof.

[All paired results](../data/processed/step3_summary.csv).
The figure uses 0.1-second samples; duration statistics above use the original timing records.
