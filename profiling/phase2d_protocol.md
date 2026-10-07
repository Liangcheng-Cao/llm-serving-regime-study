# Phase 2D preregistered collection and descriptive analysis

Base: 9c55be2. All outputs below outputs/gpu-mechanism-study/phase2d; older
Phase 2B/2C artifacts and historical archives are immutable inputs.

Primary: B8 graph bucket8 vs B8 forced eager. Secondary: B9 natural eager cap8
vs B9 graph cap16/bucket12. Reuse these arms for eager scaling (with different
disable policies explicitly acknowledged) and graph batch/padding scaling.
Arm IDs remain C,D,A,B from Phase 2B/2C, respectively; reports use descriptive names.

Budget: three independent trace server starts per arm at output128, and three
independent unprofiled starts per arm with three measured cohorts each. Every start
has three excluded warmups. This is 24 initial server starts, approximately 15–20
minutes on the measured startup path. Use three rather than five traces to bound
cost; do not increase repetitions to remove high CV. No automatic failed-run
replacement. Every failure remains indexed; stop for correction if one occurs.
Three block orders: C,D,A,B; B,A,D,C; D,C,B,A. Alternate baseline/profile order
by block and arm position. No concurrent GPU experiments.

After all 12 primary traces validate structurally, collect a smaller B8-only
output256 confirmation: two traces per arm and one unprofiled start with three
cohorts per arm (six additional starts). This confirms the primary comparison;
it is not a full output-length interaction study. All other workload controls,
model revision, dtype, seed, prefill and overlap policies stay fixed.

Add two worker NVTX instant markers bracketing the starts of decode submissions
10 and 80, after one fixed prefilling cohort. A source-hash-guarded call in the
isolated ModelRunner copy counts only decode submissions. No per-token NVTX range,
additional synchronization or per-token I/O. Existing completion timing stays
unchanged. Require both markers in the verified worker at the requested B.

The primary trace region is the host timestamp interval between these markers:
70 decode-submission intervals. Clip observed GPU kernels, copies and memsets to
that interval for union-time metrics, including boundary-crossing work only for
its in-window duration. Count intersecting kernels and record clipped-boundary
counts. Count API starts within the region; durations for API activity are clipped
to the region. No exact per-request GPU attribution is claimed. Report sensitivity
to the existing 70-completion window. Graph arms must show 70 runtime graph launches
and eager arms zero in the submission window. Reject mixed B/path cohorts.

Nsight 2026.1.3 SQLite schema was inspected from the prior exported database:
RUNTIME includes both cuda* and cu* names; DRIVER is absent. Split by API name
prefix, retaining driver metrics separately. Kernels include globalPid, streamId,
correlationId, demangledName, grid/block dimensions. MEMCPY and MEMSET expose
timestamps, bytes and worker IDs. NVTX text or StringIds resolves markers.
Preserve schema snapshots in compact summaries; fail on missing required columns.

All metrics are observed, not exhaustive, while completeness warnings remain.
Union kernel intervals across streams for kernel busy time; union kernels+copies+
memsets for observed work. Region minus observed work is inactive time. Positive
interior gaps between merged kernel intervals are inter-kernel gaps; exclude
leading/trailing edges and also report their duration. Copy work inside a kernel
gap means that gap is not device inactivity. Gap fraction uses first-to-last
observed kernel span; inactive fraction uses the full marked region.

Normalize by B*70 useful decode tokens, 70 steps and B requests. Report these as
window-normalized observations. Kernel duration sum, union GPU busy time, host API
sum and synchronization wait are overlapping quantities, never additive causes.
Unprofiled cohort cadence/E2E/throughput are the serving effect estimates.

Across independent traces report mean, median/p50, sample SD, CV and p95 for all
numeric metrics; flag CV>10%, never silently discard it. For serving metrics report
both all cohorts and restart means (the independent unit). Preserve all kernel
names; conservative family rules leave ambiguous names unknown. No Nsight Compute,
hardware-counter claims, tuning, microbenchmark or final mechanism verdict.
