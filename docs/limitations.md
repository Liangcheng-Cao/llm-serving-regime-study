# Limitations and scope

- One RTX 4070, one small Qwen model, SGLang 0.5.19, BF16 and one attention backend.
  Thresholds and effect sizes are not universal or predictions about another installation.
- The default graph ceiling is a software heuristic, not a GPU hardware capacity limit.
  Raising it changes capture/padding coverage and may affect more than isolated kernel cost.
- Two server starts per configuration and order reversal offer limited control of drift.
  The Step 2 B=8 negative control drifted; cohort resampling does not estimate all restart uncertainty.
- Fixed synthetic, equal-length requests simplify residual-work accounting. Natural prompt
  variation, output-length variation, multi-model service and realistic KV pressure are untested.
- Sixty seconds establishes a finite observation window, not infinite persistence,
  proven hysteresis, proven bistability, multiple equilibria or a stability theorem.
- LOW/HIGH start with different counts and work distributions. A hidden history term
  is not needed by this test; absence of every possible history effect is not proved.
- The model assumes B=N and omits explicit prefill, admission, scheduler, network,
  overlap, context age and padding behavior beyond their effect on the measured law.
  Real server residency and model sojourn are related, not identically defined.
- Cliff removal extrapolates constant per-job graph speed above B=8. It is a model
  intervention, not a calibrated prediction of the real max16 system.
- All-128 sensitivity visits B>16; the held-constant aggregate-rate extension is
  unvalidated there. Primary empirical and uniform cases stay in the measured range.
- Qualitative reproduction is a screening result, not a claim of model uniqueness
  or exact quantitative agreement. No new scheduler, OR formulation or algorithm is proposed.
- Public data are sanitized projections. Repeating the original GPU measurements
  requires rebuilding and validating instrumentation from the documented protocol.
