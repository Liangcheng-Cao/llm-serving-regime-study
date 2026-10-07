# Publication audit — Phase 2F

## Scope and verdict

**Ready for public GitHub publication as a bounded empirical research project.**
Packaging adds no measurements, hardware counters, microbenchmark or scheduler.
Publication readiness does not certify a universal speedup, a turnkey clean-machine
GPU environment, exhaustive Nsight coverage or Level-3 causal attribution.

## Claim audit

Current narrative consistently uses Level-2 strong mechanism evidence and
host/runtime launch and orchestration structure. It distinguishes the unprofiled
serving effect from profiled temporal observations. No exact causal millisecond
decomposition or assertion that CUDA Graph eliminates CPU overhead is made.

Repository Markdown was searched for stronger phrases including “proved”, “proves”,
“exactly explains”, elimination of overhead and entire-slowdown causality. Historical
negative statements (for example “not proved”) remain contextual limitations, not
positive causal claims. Frozen phase reports, protocols and historical measurements
remain unchanged; phase-specific preregistered constraints are not rewritten as
current conclusions. The original no-go for a new OR gap remains visible.

Stale runner documentation was updated to distinguish the three current patch sites,
submission markers, campaign/restart analysis and historical completion-window
validation. The original limitations page now identifies its historical scope.

## Numerical audit

The README hero table is rendered from Phase 2D `analysis.json`; its sequence value
comes from Phase 2E cross-trace LCS comparisons. The audit records full precision,
formatted values and source fields in
[headline_audit.json](../outputs/gpu-mechanism-study/phase2f/headline_audit.json).
Ratios explicitly identify eager/graph or graph/eager direction. Busy-fraction
differences are percentage points. Sequence similarity is one pairwise measure,
not independent graph/eager percentages.

Headline values in the abstract, executive summary and final report were checked
against these generated summaries and the preserved phase evidence. Unprofiled
cadence and throughput ratios use their own aggregates. Historical 10.892/3.655 ms
values remain explicitly separate from the fresh 10.114/3.117 ms B8 result.

## Figure review

Six existing Phase 2E figures were curated: serving interval, work versus gaps,
launch calls, busy fraction, gap distributions and conceptual timeline. New
standalone captions specify the independent unit, normalization, profiling scope
and interpretation limits. The schematic is labeled conceptual, not literal trace
scale. Historical queue figures and the Phase 2D composition figure remain supporting
material. No trace screenshot was fabricated and no plotted data were changed.

## Reproduction and storage

[Reproduction instructions](REPRODUCIBILITY.md) distinguish CPU validation,
summary-only regeneration, event-level parsing and new GPU collection. New output
locations prevent archived results from being overwritten. The summary workflow
was checked in scratch storage; GPU collection is documented but deliberately not
rerun in Phase 2F. Compatible hardware/software, cached model weights and guarded
source hashes are prerequisites; a full environment lock is not supplied.

Raw `.nsys-rep` and SQLite files remain excluded. Compact evidence and input hashes
are committed. Manifests intentionally retain local paths and host identifiers for
provenance; readers must substitute their own paths. A targeted tracked-file scan
checks common credential patterns, without treating absence of matches as a
comprehensive security certification. The MIT license is retained; model weights
and external runtime software are not redistributed by this repository.

## Validation record

[validation_checks.json](../outputs/gpu-mechanism-study/phase2f/validation_checks.json)
records all 47 tests, historical validation, protected-file checks, local link checks,
numerical checks and reproduction checks. All work is committed locally; no push or
public posting was performed.
