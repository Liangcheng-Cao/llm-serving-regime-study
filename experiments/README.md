# Experimental evidence

The five notes record completed screening decisions, not instructions to run new experiments.

| Evidence | Question | Verdict |
|---|---|---|
| [Regime audit](step1_regime_audit.md) | Does the boundary change actual decode execution? | PASS |
| [Same-batch contrast](step2_service_cliff.md) | Does changing graph coverage change service? | PASS, negative-control drift retained |
| [Pulse response](step3_pulse_response.md) | Can a finite pulse amplify queue state? | PASS |
| [Long horizon](step4_long_horizon.md) | Does separation last 60 seconds? | PASS within that horizon |
| [Classical model](step5_classical_model.md) | Is a new queueing abstraction required? | NO-GO FOR NEW OR GAP |

The public package contains numeric projections and rewritten notes. Raw server stdout,
machine paths, request identifiers, installed source copies, and the GPU harness are excluded.
Offline analysis and simulation are reproducible; GPU measurements are documented, not turnkey reproducible.
See [data provenance](../docs/data_provenance.md) and [methodology](../docs/methodology.md).
