"""Render numerical Phase 2D descriptive report from committed-format summaries."""
import json
import statistics as st
from .analyze_phase2d import analyze,write_csv
from .phase2d_metrics import LABELS
from .run_phase2d import OUT


def fmt(x,d=3):
    return "—" if x is None else f"{x:.{d}f}"


def main():
    result,traces,serving,families,kernels=analyze()
    T={(r["output_len"],r["arm"],r["metric"]):r for r in result["trace_aggregates"]}
    S={(r["output_len"],r["arm"],r["metric"]):r for r in result["unprofiled_aggregates"]}
    def stat(arm,metric,source=T,nout=128):return source.get((nout,arm,metric))
    def mean(arm,metric,source=T,nout=128):return stat(arm,metric,source,nout)["mean"]
    lines=["# Phase 2D — controlled GPU trace campaign", "", "**Descriptive evidence only; B8 graph versus B8 forced eager is primary.**",
      "", "**Unprofiled measurements estimate the serving effect. Profiled measurements describe observed execution structure. Differential profiler perturbation prevents an unbiased absolute latency decomposition.**", "",
      "## Scope and trace acceptance", "",
      "All runs use the fixed model revision, BF16/FlashInfer, input length 128, seed 42, greedy generation, forced output length, overlap enabled, radix cache disabled and unchanged prefill policy from Phase 2B/2C. Driver 581.08 differs from the original study. Three excluded warmup cohorts precede each independent server start.", "",
      "Primary output128: three independent traces per arm, and three separate unprofiled starts per arm with three measured cohorts each. Three rather than five traces bounds the 24-start primary campaign cost; high CV does not trigger selective repetitions. The smaller output256 confirmation covers only the primary B8 pair: two traces per arm and one separate unprofiled start with three cohorts per arm. It begins only after primary acceptance.", "",
      "| Arm | Policy | Output128 valid traces | Unprofiled starts/cohorts | Output256 valid traces |",
      "|---|---|---:|---:|---:|"]
    for a in "CDBA":
        lines.append(f"| {a} | {LABELS[a]} | {sum(r['arm']==a and r['output_len']==128 for r in traces)} | {sum(r['arm']==a and r['output_len']==128 for r in serving)}/9 | {sum(r['arm']==a and r['output_len']==256 for r in traces)} |")
    lines += ["",f"Invalid campaign runs: **{len(result['invalid_runs'])}**. All attempts remain in trace_manifest.jsonl; no silent replacement. Each accepted trace verifies B/path, actual replay bucket, both submission markers in the GPU worker, the completion window, CLI export, kernel/API correlations and expected graph-launch count. All reported counts remain observed counts, not certified exhaustive totals.","",
      "## Measurement regions and normalization", "",
      "A guarded call in an isolated ModelRunner copy resets a decode counter at prefill and emits two NVTX instant markers at the starts of decode submissions 10 and 80. There is no per-token NVTX range, added GPU synchronization or per-token I/O. The installed package is unchanged. The existing completion observer separately measures endpoints 10→80 for serving cadence.","",
      "The primary structured trace region is the time between the two submission markers. GPU activity is clipped to this 70-interval steady-state decode window across all worker streams. Kernel counts include intersecting records; duration distributions use wholly contained kernels. Boundary crossings are recorded. API counts select starts within the window and clip ending durations. This is **GPU activity observed during the steady-state decode measurement window**, not exact per-request GPU attribution. Completion-window results are retained as a sensitivity check.","",
      "Normalization uses 70 intervals, B×70 useful decode-token equivalents and B requests; padding slots are excluded. Aggregate throughput and an individual token's latency are distinct. Per-token GPU metrics below are window-normalized observations. Inter-kernel gaps are positive interior gaps between merged kernel intervals, excluding leading/trailing edges. Their fraction uses the first-to-last kernel span. GPU busy/inactive fractions instead use the complete marked region and the union of kernels, copies and memsets. Copy time can lie within a kernel gap, so kernel gaps are not automatically GPU inactivity or CPU overhead.","",
      "Nsight 2026.1.3's actual SQLite schema was inspected before implementation and is saved in each parsed summary. Its RUNTIME table contains both runtime cuda* and driver cu* calls; these are separated by name. Runtime kernel launch counts exclude driver-layer duplication. API-duration sums and synchronization waits overlap GPU work; none of these quantities are added to explain the real latency gap.","",
      "## Unprofiled serving metrics — source of effect size", "",
      "Statistics below use independent restart means (three cohorts averaged within each restart). Median/p50 and p95 here describe the three restart means, not a population tail estimate. Detailed cohort and within-cohort p50/p95 values are also exported. SD is sample SD; CV is SD/mean. No confidence interval or steady-state population claim follows from three restarts.","",
      "| Arm | Metric | Mean | Median/p50 | SD | CV % | p95 |",
      "|---|---|---:|---:|---:|---:|---:|"]
    for a in "CDBA":
        for metric,label in [("decode_ms","Decode cadence ms"),("e2e_s","Cohort E2E s"),("decode_tokens_s","Useful decode tok/s"),("serving_tokens_s","Full-cohort output tok/s"),("requests_s","Cohort req/s")]:
            r=stat(a,metric,S)
            lines.append(f"| {LABELS[a]} | {label} | {fmt(r['mean'])} | {fmt(r['median'])} | {fmt(r['sd'])} | {fmt(100*r['cv'],2) if r['cv'] is not None else '—'} | {fmt(r['p95'])} |")
    lines += ["",f"B8 unprofiled eager/graph cadence ratio: **{mean('D','decode_ms',S)/mean('C','decode_ms',S):.3f}×**. B9 unprofiled eager/graph: **{mean('A','decode_ms',S)/mean('B','decode_ms',S):.3f}×**. Compare these fresh values with the historical qualitative ordering; do not pool environments.","",
      "## Primary B8 and practical B9 pairwise structure", "",
      "Each profile entry is the mean across three independent traces. Full mean/median/SD/CV/p95 for every numeric metric is in aggregate_metrics.csv; ratios/deltas for all four requested pairs are in pairwise_effects.csv. Busy time includes observed kernel/copy/memset union unless explicitly labeled kernel-only."]
    chosen=[("runtime_api_calls_per_token","Observed runtime API calls/token"),("runtime_api_us_per_token","Runtime API duration sum µs/token"),
      ("kernel_launch_calls_per_token","Runtime kernel launch calls/token"),("driver_api_calls_per_token","Driver API calls/token (separate)"),
      ("kernel_count_per_token","Observed kernels/token"),("kernel_sum_us_per_token","Kernel-duration sum µs/token"),
      ("kernel_busy_us_per_token","Kernel-union busy µs/token"),("gpu_busy_us_per_token","GPU-work union µs/token"),
      ("inter_kernel_gap_us_per_token","Interior kernel gaps µs/token"),("gpu_inactive_us_per_token","Observed GPU inactive µs/token"),
      ("gpu_busy_fraction","GPU busy fraction"),("sync_calls_per_token","Runtime sync calls/token"),
      ("sync_us_per_token","Runtime sync duration sum µs/token"),("graph_launch_count","Observed graph launches / window"),
      ("copy_union_us_per_token","Copy union µs/token"),("active_kernel_streams","Active kernel streams")]
    for name,g,e in [("B8 primary","C","D"),("B9 practical (padded graph)","B","A")]:
        lines += ["",f"### {name}","","| Metric | Graph | Eager | Eager/graph | Eager − graph |","|---|---:|---:|---:|---:|"]
        for metric,label in [("decode_ms","**Unprofiled cadence ms**")]+chosen:
            src=S if metric=="decode_ms" else T
            gv,ev=mean(g,metric,src),mean(e,metric,src)
            lines.append(f"| {label} | {fmt(gv)} | {fmt(ev)} | {fmt(ev/gv) if gv else '—'} | {fmt(ev-gv)} |")
    lines += ["", "## API, kernel duration and gap distributions", "",
      "These are means of each trace's event-distribution statistics. Their cross-trace dispersion is separately exported. Positive gap distributions exclude overlapping/adjacent kernel intervals.","",
      "| Arm | Distribution | Mean µs | Median µs | p95 µs | Max gap µs |","|---|---|---:|---:|---:|---:|"]
    for a in "CDBA":
        for prefix,label in [("runtime_api_duration","Runtime API"),("kernel_duration","Kernel"),("gap","Positive inter-kernel gap")]:
            values=[mean(a,f"{prefix}_{k}_us") for k in ("mean","median","p95")]
            lines.append(f"| {LABELS[a]} | {label} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | {fmt(mean(a,'gap_max_us')) if prefix=='gap' else '—'} |")
    lines += ["", "## Kernel composition, streams and padding", "",
      "Full demangled names, clipped cumulative time, count and grid/block dimensions are preserved in kernel_composition.csv and individual JSON summaries. Classification is name-based and conservative. An attention kernel named BatchPrefillWithPagedKVCache can execute in decode; its name does not establish that prefill leaked into the window. Ambiguous reduction, positional and metadata kernels remain unknown.","",
      "| Arm | Family | Mean cumulative µs/token |","|---|---|---:|"]
    for a in "CDBA":
        for f in ["GEMM/matmul","attention","normalization","sampling","elementwise","KV/cache","unknown"]:
            rows=[r["time_us_per_token"] for r in families if r["arm"]==a and r["output_len"]==128 and r["family"]==f]
            lines.append(f"| {LABELS[a]} | {f} | {fmt(st.mean(rows))} |")
    signatures={a:{r["name"] for r in kernels if r["arm"]==a and r["output_len"]==128} for a in "ABCD"}
    for pair,g,e in [("B8","C","D"),("B9","B","A")]:
        lines += ["",f"{pair} observed name sets: {len(signatures[g]&signatures[e])} shared, {len(signatures[g]-signatures[e])} graph-only, {len(signatures[e]-signatures[g])} eager-only. Equal names do not prove equal shapes or work; see preserved shape histograms."]
    lines += ["","Top five exact kernel names per arm by average observed cumulative time/token (name wrapping is intentional):",""]
    for a in "CDBA":
        values=defaultdict_float()
        counts=defaultdict_float()
        for k in kernels:
            if k["arm"]==a and k["output_len"]==128:
                values[k["name"]]+=k["time_us_per_token"]/3;counts[k["name"]]+=k["count"]/3
        lines += [f"### {LABELS[a]}","","| Kernel | µs/token | Mean observed count |","|---|---:|---:|"]
        for name,time in sorted(values.items(),key=lambda x:x[1],reverse=True)[:5]:
            lines.append(f"| `{name.replace('|','&#124;')}` | {fmt(time)} | {fmt(counts[name],1)} |")
    lines += ["",f"B9 graph/bucket12 versus B8 graph/bucket8: kernel-duration sum per token ratio **{mean('B','kernel_sum_us_per_token')/mean('C','kernel_sum_us_per_token'):.3f}**, GPU-work union per token ratio **{mean('B','gpu_busy_us_per_token')/mean('C','gpu_busy_us_per_token'):.3f}**. Per-step kernel-work ratio (removing useful-token denominator change) is **{mean('B','kernel_sum_us_per_token')*9/(mean('C','kernel_sum_us_per_token')*8):.3f}**. This comparison changes useful B and padded bucket together; it cannot isolate padding cost. Copy activity is observable but DRAM/L2 traffic is not measured.","",
      "| Arm | Kernel streams | Kernel overlap excess µs/token | Kernel-gap fraction | Submission − completion busy fraction (pp) |","|---|---:|---:|---:|---:|"]
    for a in "CDBA":lines.append(f"| {LABELS[a]} | {fmt(mean(a,'active_kernel_streams'),1)} | {fmt(mean(a,'kernel_overlap_excess_us_per_token'))} | {fmt(mean(a,'inter_kernel_gap_fraction'))} | {fmt(mean(a,'window_busy_fraction_delta_pp'))} |")
    lines += ["", "Stream IDs are process-local identifiers, not semantic roles. Kernel interval overlap and active-stream counts describe serialization; kernel overlap excess is duration-sum minus union and not itself a percentage of simultaneous occupancy. Graph arms require exactly 70 observed runtime graph launches per window; individual runtime kernel launches can remain outside graphs (for example surrounding runtime work).", "",
      "## Cross-trace stability and profiler perturbation", "",
      "CV >10% is a warning, not an exclusion rule. CV uses SD/absolute mean; near-zero signed diagnostic deltas can have enormous CV, so their absolute percentage-point SD matters more. Only three primary traces per arm are available, so extreme-event p95/max estimates can vary. The raw trace metrics, every CV and all flagged metrics remain exported.","",
      "| Metric CV % | B8 graph | B8 eager | B9 graph | B9 eager |","|---|---:|---:|---:|---:|"]
    for metric,label in chosen:
        vals=[stat(a,metric)["cv"] for a in "CDBA"]
        lines.append(f"| {label} | "+" | ".join(fmt(v*100,2) if v is not None else '—' for v in vals)+" |")
    flags=[r for r in result["trace_aggregates"] if r["output_len"]==128 and r["warning_cv_gt_10pct"]]
    lines += ["",f"Primary metric/arm entries with CV>10%: **{len(flags)}** (including derived and tail metrics). No such entry is silently removed.","",
      "| Arm | Cadence perturbation mean % | SD (percentage points) | E2E perturbation mean % |","|---|---:|---:|---:|"]
    oh={(r["output_len"],r["arm"],r["metric"]):r for r in result["overhead_aggregates"]}
    for a in "CDBA":
        r=oh[128,a,"cadence_change_pct"];e=oh[128,a,"e2e_change_pct"]
        lines.append(f"| {LABELS[a]} | {fmt(r['mean'],2)} | {fmt(r['sd'],2)} | {fmt(e['mean'],2)} |")
    lines += ["", "Each perturbation comparison matches arm/output length/block but uses separate server starts. Nsight software instrumentation and environment drift are both included. No profiled absolute serving-time difference is assigned to launch, gap, sync or compute causes.","",
      "## Output256 confirmation", "",
      "| Arm | Unprofiled cadence mean ms | Cohort E2E mean s | Trace GPU busy µs/token | Trace gap µs/token | Valid traces |","|---|---:|---:|---:|---:|---:|"]
    for a in "CD":
        if stat(a,"decode_ms",S,256):lines.append(f"| {LABELS[a]} | {fmt(mean(a,'decode_ms',S,256))} | {fmt(mean(a,'e2e_s',S,256))} | {fmt(mean(a,'gpu_busy_us_per_token',T,256))} | {fmt(mean(a,'inter_kernel_gap_us_per_token',T,256))} | 2 |")
    lines += ["", "Only one unprofiled restart per output256 arm is available; restart SD is unavailable. Two traces provide limited confirmation, not a formal output-length interaction estimate. All comparisons use the same early decode window; no longer-context conclusion follows.","",
      "## Data quality and descriptive answers", "",
      "All captures retain Nsight's CUDA/NVTX completeness warnings. Hardware tracing is unsupported, so software tracing was used. Unified Memory tracking is unavailable; no memory-bandwidth conclusion is made. Missing auxiliary-process CUDA events are distinguished from worker warnings. All observed selected kernels have worker API correlations; that does not rule out jointly missing API/kernel records. Structural acceptance is not exhaustive-record certification.","",
      "Reliable observations are the verified paths/buckets, observed graph-launch pattern, observed event names/shapes, marked steady region and cross-trace reproducibility of recorded metrics. Event sums, busy fractions and gaps are conditional on captured records. Missing work can reduce observed busy time and inflate observed gaps; stable repeated results do not mathematically prove completeness.","",
      "The report separates: (1) the repeated unprofiled serving effect; (2) repeated profiled launch/API structure; (3) recorded GPU work/gaps and kernel composition. Pairwise tables show whether GPU work changes are smaller than serving cadence changes without equating the two timing environments. Larger gaps are not labeled CPU launch overhead. Synchronization duration is blocking API time and can overlap GPU execution; more graph-side waiting would not imply a larger serving cost.","",
      "The B9 graph comparison changes both useful B and bucket size. Name/shape histograms and per-step versus per-token work quantify the observed difference but cannot isolate padding. Kernel-duration sums do not measure DRAM traffic or occupancy. No hypothesis is promoted to a final mechanism claim here.","",
      "## Artifacts, figures and validation", "",
      "- trace_manifest.jsonl indexes every attempt; runs/ contains immutable manifests; ignored traces/ retains raw reports and summaries/ retains SQLite plus compact parsed evidence.",
      "- trace_metrics.csv, aggregate_metrics.csv and pairwise_effects.csv preserve actual values, all cross-trace statistics, ratios and deltas.",
      "- unprofiled_* files preserve restart/cohort statistics; profiler_perturbation.csv keeps the two measurement roles separate.",
      "- kernel_composition.csv and kernel_families.csv preserve names, counts, shapes and conservative classification.","",
      "![Unprofiled decode cadence](figures/01_unprofiled_decode.png)","",
      "![Observed GPU union busy time](figures/02_gpu_busy.png)","",
      "![Observed inter-kernel gaps](figures/03_kernel_gaps.png)","",
      "![Observed kernel launch calls](figures/04_launch_calls.png)","",
      "![Observed GPU busy fraction](figures/05_gpu_busy_fraction.png)","",
      "![Kernel families](figures/06_kernel_families.png)","",
      "Test and preservation results are recorded separately in validation_checks.json. Historical archives and Phase 2B/2C outputs were not overwritten. No Nsight Compute, custom CUDA kernels, parameter tuning or push was performed.","",
      "**READY FOR PHASE 2E — MECHANISM ANALYSIS**", "",
      "The repeated observed structure and independently measured serving effect support proceeding to cautious mechanism analysis. Readiness does not certify exhaustive counts or precise GPU attribution. Resolve worker completeness warnings or establish independent event coverage before claims that depend on exhaustive accounting. Do not treat the profiler's gap increase as an unbiased decomposition of the unprofiled ~3× effect, and do not promote these descriptive observations to a final causal mechanism claim."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    print("Report written",len(traces),"traces")


def defaultdict_float():
    from collections import defaultdict
    return defaultdict(float)


if __name__=="__main__":main()
