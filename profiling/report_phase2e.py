"""Build the mechanism report from retained event analysis and Phase 2D metrics."""
from collections import defaultdict
import csv
import json
import statistics as st
from .analyze_phase2e import SOURCE, OUT
from .analyze_phase2d import write_csv
from .phase2d_metrics import LABELS


def f(x,n=3):return '—' if x is None else f'{x:,.{n}f}'


def main():
    r=json.loads((OUT/'analysis.json').read_text());d=json.loads((SOURCE/'analysis.json').read_text())
    ts=r['traces'];primary=[x for x in ts if x['output_len']==128]
    def rows(a,n=128):return [t for t in ts if t['arm']==a and t['output_len']==n]
    def avg(a,section,key,n=128):return st.mean(t[section][key] for t in rows(a,n))
    def old(a,key,source='trace_aggregates',n=128):
        return next(x for x in d[source] if x['arm']==a and x['output_len']==n and x['metric']==key)
    def mean(a,key,source='trace_aggregates',n=128):return old(a,key,source,n)['mean']
    lines=['# Phase 2E — mechanism analysis and targeted falsification','',
      '**Claim level 2: STRONG MECHANISM EVIDENCE. B8 is the primary controlled comparison.**','',
      'Repeated controlled observations strongly support host-driven/runtime orchestration and launch structure as the primary mechanism behind the graph/eager service difference. They do not establish that every GPU gap is CPU launch execution time, nor quantify an exact causal decomposition of real serving latency.','',
      '## 1. Research question and scope','',
      'Which competing mechanism best explains the approximately 3× service-rate change? This phase performs read-only analysis of all 16 accepted Phase 2D SQLite traces: three per primary arm and two per output256 B8 arm. No new GPU capture, microbenchmark, tuning or Nsight Compute run was performed. Source traces, archived results and Phase 2D artifacts are preserved.','',
      'The fixed model is Qwen2.5-0.5B-Instruct at the recorded revision on RTX 4070, BF16/FlashInfer, input128, with overlap enabled. All configuration and environment details remain in the Phase 2D manifests. Generalization to other models, devices, policies or contexts is untested.','',
      '## 2. Controlled evidence','',
      '**Unprofiled measurements estimate the serving effect; profiled records describe execution structure.** Primary statistics use three independent server starts per arm, three measured cohorts per unprofiled start. No event-level sample size is substituted for independent traces. Mean ± sample SD is used instead of a fragile confidence interval from n=3.','',
      '| Arm | Unprofiled decode ms, mean ± SD | E2E s, mean ± SD | Useful decode tok/s | Primary traces |',
      '|---|---:|---:|---:|---:|']
    for a in 'CDBA':
        q=old(a,'decode_ms','unprofiled_aggregates');e=old(a,'e2e_s','unprofiled_aggregates')
        lines.append(f"| {LABELS[a]} | {f(q['mean'])} ± {f(q['sd'])} | {f(e['mean'])} ± {f(e['sd'])} | {f(mean(a,'decode_tokens_s','unprofiled_aggregates'),1)} | 3 |")
    lines+=['',f"Eager/graph cadence ratio: **{mean('D','decode_ms','unprofiled_aggregates')/mean('C','decode_ms','unprofiled_aggregates'):.3f}× B8**, **{mean('A','decode_ms','unprofiled_aggregates')/mean('B','decode_ms','unprofiled_aggregates'):.3f}× B9**. B8 changes execution policy at the same useful batch without padding; B9 graph additionally uses bucket12.",'']
    metrics=[('gpu_busy_us_per_token','GPU-work union µs/token'),('kernel_sum_us_per_token','Kernel-duration sum µs/token'),
      ('inter_kernel_gap_us_per_token','Interior kernel gaps µs/token'),('gpu_busy_fraction','GPU busy fraction'),
      ('kernel_launch_calls_per_token','Runtime individual launches/token'),('graph_launch_count','Graph launches/window'),
      ('runtime_api_calls_per_token','Runtime API calls/token'),('runtime_api_us_per_token','Runtime API duration µs/token'),
      ('sync_calls_per_token','Sync calls/token'),('sync_us_per_token','Sync duration µs/token'),
      ('family_GEMM_matmul_us_per_token','GEMM µs/token')]
    structural=[]
    for pair,g,e in [('B8 primary','C','D'),('B9 practical','B','A')]:
        lines += [f'### {pair}: observed structure','', '| Metric | Graph mean ± SD | Eager mean ± SD | Eager/graph | Eager − graph |','|---|---:|---:|---:|---:|']
        for key,label in metrics:
            a=old(g,key);b=old(e,key);ratio=b['mean']/a['mean'] if a['mean'] else None
            lines.append(f"| {label} | {f(a['mean'])} ± {f(a['sd'])} | {f(b['mean'])} ± {f(b['sd'])} | {f(ratio)} | {f(b['mean']-a['mean'])} |")
            structural.append(dict(pair=pair,metric=key,graph_mean=a['mean'],graph_sd=a['sd'],eager_mean=b['mean'],eager_sd=b['sd'],ratio=ratio,delta=b['mean']-a['mean']))
        lines+=['']
    write_csv(OUT/'structural_effects.csv',structural)
    hypothesis=[]
    for pair,g,e in [('B8 primary','C','D'),('B9 practical','B','A')]:
        work=mean(e,'gpu_busy_us_per_token')/mean(g,'gpu_busy_us_per_token');gap=mean(e,'inter_kernel_gap_us_per_token')/mean(g,'inter_kernel_gap_us_per_token')
        launch=mean(e,'kernel_launch_calls_per_token')/mean(g,'kernel_launch_calls_per_token')
        items=[
          ('H1 More GPU compute','Eager GPU work comparable to the ~3× serving ratio',f'GPU-work ratio {work:.3f}; GEMM ratio {mean(e,"family_GEMM_matmul_us_per_token")/mean(g,"family_GEMM_matmul_us_per_token"):.3f}', 'Contradicts dominant explanation','High within captured work','Duration is not FLOP count; incomplete traces and small compute differences remain'),
          ('H2 Different computational path','Different dominant names/families/sequence', '25 shared names; '+('zero mode-only names' if g=='C' else 'two additional graph fill kernels')+'; exact sequence comparisons below','Contradicts dominant explanation','High for observed names; moderate for full workload','Equal names do not prove equal shapes, inputs or dependencies; B9 has padding'),
          ('H3 Host/runtime orchestration bubbles','Similar GPU work; more individual launches, much larger gaps and lower density',f'Launch ratio {launch:.3f}; gap ratio {gap:.3f}; eager gap successors all correlated to individual launches','Supports strongest mechanism','Strong, level 2','Correlation is not an intervention on host dispatch; dependencies, driver and tracing effects remain'),
          ('H4 Synchronization-dominated eager','Eager blocking wait time dominates',f'Eager sync count 3× graph but duration {mean(e,"sync_us_per_token"):.3f} versus {mean(g,"sync_us_per_token"):.3f} µs/token','Contradicts explicit-sync-time explanation','High for recorded explicit APIs','Implicit/device synchronization and dependencies are not all measured'),
          ('H5 Much faster GPU kernels under graph','Large shared-kernel duration reduction sufficient for ~3× effect',f'Kernel-duration-sum ratio {mean(e,"kernel_sum_us_per_token")/mean(g,"kernel_sum_us_per_token"):.3f}; GEMM dominates both','Weak secondary effects; contradicts dominant explanation','Moderate to high','Some small kernels differ materially; duration includes environment and tracing effects'),
          ('H6 Pure profiler artifact','Serving effect absent unprofiled, or structural difference fails replication','Unprofiled ~3× effect; stable repeated structure; B8 output256 retains gap/work contrast','Contradicts pure serving artifact; structural inflation unresolved','High for real serving effect; moderate for unprofiled structure','Stable instrumentation can induce stable bias; repetition cannot exclude systematic structural distortion')]
        for h,p,o,v,c,l in items:hypothesis.append(dict(pair=pair,hypothesis=h,required_observation=p,observed_evidence=o,assessment=v,confidence=c,limitations=l))
    write_csv(OUT/'hypothesis_evidence.csv',hypothesis)
    lines+=['## 3. Competing hypotheses','']
    for pair in ['B8 primary','B9 practical']:
        lines += [f'### {pair}','', '| Hypothesis | Required observation | Observed evidence | Assessment | Confidence | Limitations |','|---|---|---|---|---|---|']
        for h in hypothesis:
            if h['pair']==pair:lines.append('| '+' | '.join(h[k] for k in ['hypothesis','required_observation','observed_evidence','assessment','confidence','limitations'])+' |')
        lines+=['']
    lines+=['## 4. GPU computation and sequence comparison','',
      'Kernels are ordered by GPU start time, with deterministic end/stream/name tie breaks. Exact longest-common-subsequence similarity is 2LCS/(Ngraph+Neager), computed with bitset dynamic programming; repeated names are retained. All nine cross-trace pairings are descriptive sensitivity comparisons, not nine independent experiments. Cross-stream ordering is not a dependency graph. Dominant-stream comparisons reduce that ordering sensitivity.','',
      '| Pair | Sequence | Mean similarity | Min–max |','|---|---|---:|---:|']
    for pair in ['B8','B9','graph_padding','B8_output256']:
        for kind in ['all_names','all_families','dominant_stream_names']:
            v=[x['similarity'] for x in r['sequence_comparisons'] if x['pair']==pair and x['sequence']==kind]
            lines.append(f'| {pair} | {kind} | {100*st.mean(v):.3f}% | {100*min(v):.3f}–{100*max(v):.3f}% |')
    lines+=['','Repeated layer patterns can make LCS high even when some events differ. This is evidence of observed ordering similarity, not proof of equal tensors or work. Boundary clipping and cross-stream reordering account for some mismatch; they are not assumed to explain all mismatch.','',
      'Per-kernel duration distributions use wholly contained records. Counts include intersecting records; cumulative time is clipped and normalized. The following B8 top kernels are ranked by graph cumulative time; full demangled names and all 25 comparisons are in kernel_comparisons.csv. Entries are averages of per-trace means/medians, avoiding pooled-event pseudoreplication.','',
      '| B8 shared kernel (abbreviated) | Count graph/eager | Mean µs graph/eager | Median µs graph/eager | Cumulative µs/token graph/eager |','|---|---:|---:|---:|---:|']
    kp=[x for x in r['kernel_comparisons'] if x['pair']=='B8']
    for k in sorted(kp,key=lambda k:k['cumulative_us_per_token_reference_mean'],reverse=True)[:8]:
        name=k['name'];short=('CUTLASS GEMM 128x2' if '128x2' in name else 'CUTLASS GEMM 128x1' if '128x1' in name else 'FlashInfer attention' if 'BatchPrefillWithPaged' in name else 'Fused add RMSNorm' if 'fused_add_rmsnorm' in name else 'Merge attention states' if 'MergeStates' in name else 'Activation and multiply' if 'act_and_mul' in name else name[:80])
        cells=[' / '.join(f(k[f'{metric}_{side}_mean']) for side in ['reference','comparison']) for metric in ['observed_count','mean_us','median_us','cumulative_us_per_token']]
        lines.append('| '+short+' | '+' | '.join(cells)+' |')
    lines+=['','B8 GEMM cumulative time changes by only 1.08% eager/graph and accounts for roughly 85% of graph kernel time. Smaller families do differ: normalization is 8.579→10.129 µs/token (+18.1%), elementwise 10.475→12.267 (+17.1%), KV/cache 2.885→3.588 (+24.4%). These effects are not dismissed, but their small absolute contributions cannot account descriptively for a 3.244× serving ratio. Without hardware counters, neither equal FLOPs nor intrinsic kernel efficiency is established.','',
      '## 5. Launch structure and event association','',
      'Matches use the same worker and correlationId over the entire captured worker API set, allowing calls before the marked window to launch work inside it. The unique runtime individual launch is preferred; otherwise a unique driver individual launch is used. Ambiguous, absent and impossible-order matches are excluded rather than replaced with nearest-time guesses. Graph correlations remain many-kernels-to-one replay. No runtime/driver records are summed as independent launches for the same kernel.','',
      '| Arm | Runtime-matched kernels/window | Driver-matched | Graph-replay-matched | Unmatched/ambiguous |','|---|---:|---:|---:|---:|']
    for a in 'CDBA':
        vals=[st.mean(t['correlation_counts'].get(k,0) for t in rows(a)) for k in ['runtime_individual','driver_individual','graph_replay']]
        bad=st.mean(sum(t['correlation_counts'].get(k,0) for k in ['unmatched','ambiguous','invalid_order']) for t in rows(a))
        lines.append(f'| {LABELS[a]} | '+' | '.join(f(x,1) for x in vals)+f' | {f(bad,1)} |')
    lines+=['','For eager, every observed gap successor is uniquely matched, covering 100% of recorded gap count and time. Individual matched APIs each map to one selected eager kernel; graph replay is explicitly not one-to-one. This validates a record-level association, not event completeness.','',
      '| Eager arm | Launch starts inside preceding gap | Gap time before matched API entry | During matched API | After API return |','|---|---:|---:|---:|---:|']
    for a in 'DA':
        vals=[avg(a,'association',k)*100 for k in ['launch_start_in_gap_fraction','before_launch_fraction_of_matched_gap','during_launch_fraction_of_matched_gap','after_launch_fraction_of_matched_gap']]
        lines.append(f'| {LABELS[a]} | '+' | '.join(f(x,2)+'%' for x in vals)+' |')
    lines+=['','Each positive union gap is partitioned relative to the matched next kernel’s API entry/return. The portions sum to that gap, but are not causal components. A launch already submitted before a gap may still await dependencies, driver scheduling or GPU readiness. Work on other streams, copies and tracing can also affect interpretation.','',
      '| Eager arm | Matched layer | Return→kernel mean µs | Median µs | p95 µs |','|---|---|---:|---:|---:|']
    for a in 'DA':
        for kind in ['runtime_individual','driver_individual']:
            vals=[st.mean(t['return_to_kernel_delay_us'][kind][k] for t in rows(a)) for k in ['mean','median','p95']]
            lines.append(f'| {LABELS[a]} | {kind} | '+' | '.join(f(x) for x in vals)+' |')
    lines+=['','Negative return→kernel delays are retained: asynchronous work can start before the CPU call returns. These distributions are not a host-overhead estimate. In graph mode the remaining individual launches can queue milliseconds ahead (B8 runtime median about 2,393 µs), despite faster serving and smaller gaps. Queue delay alone is therefore not a measure of harmful launch latency.','',
      'The temporal results strengthen H3, but level 3 is not selected: correlations link submission to execution without isolating the cause of the intervening delay. There is no host dispatch counterfactual, CPU scheduling trace, complete dependency graph or independently certified event coverage.','',
      '## 6. Gap distribution and timeline accounting','',
      'Positive gaps are between the union of all selected kernel intervals across streams, excluding outer edges. They can contain copies or memsets. Tiny graph gaps (median 32 ns) should not be interpreted below practical timestamp/profiler resolution.','',
      '| B8 metric | Graph mean across traces | Eager mean across traces |','|---|---:|---:|']
    for key,label in [('count','Gap count'),('median','Median gap µs'),('p90','p90 µs'),('p95','p95 µs'),('p99','p99 µs'),('maximum','Maximum µs')]:
        lines.append(f"| {label} | {f(avg('C','gaps',key))} | {f(avg('D','gaps',key))} |")
    for threshold in (5,10,20,50):
        key=f'time_fraction_gt_{threshold}us';lines.append(f"| Fraction of total gap time >{threshold} µs | {100*avg('C','gaps',key):.2f}% | {100*avg('D','gaps',key):.2f}% |")
    lines+=['','The gap counts differ only modestly; eager’s gaps are broadly larger, rather than merely a few extra extreme stalls. About 89.38% of eager gap time is in gaps >20 µs and 46.08% in gaps >50 µs. Graph also has a long tail, but its absolute total gap time is much smaller. Per-trace distributions and their SD/CV are retained in the CSVs; p99/max estimates vary more than central workload metrics.','',
      '| B8 accounting, µs/token | Graph | Eager |','|---|---:|---:|']
    for key,label in [('marked_span_us_per_token','Marked host-timestamp region'),('kernel_span_us_per_token','First-to-last observed kernel span'),('kernel_union_us_per_token','K: kernel union'),('gpu_work_union_us_per_token','W: all GPU-work union'),('interior_gaps_us_per_token','G: interior kernel gaps'),('edge_gaps_us_per_token','E: outer edges'),('nonkernel_outside_kernels_us_per_token','N: nonkernel work outside kernels'),('naive_residual_us_per_token','Marked span − W − G'),('corrected_residual_us_per_token','Marked span − (W + G + E − N)')]:
        lines.append(f"| {label} | {f(avg('C','accounting',key),6)} | {f(avg('D','accounting',key),6)} |")
    lines+=['','K + G equals the first-to-last kernel span exactly; W + G is close to the marked span but double-counts nonkernel work outside K. Correcting edges and that overlap gives zero residual in every trace. This is an interval-accounting identity and a parser consistency check, not independent proof of completeness or a decomposition of unprofiled latency. Missing events could change K and G while preserving this identity.','',
      '## 7. Synchronization interpretation','',
      '| Arm | API | Calls/window | Mean call µs | Median call µs | Duration overlapping observed GPU work | Calls in outer 1% regions |','|---|---|---:|---:|---:|---:|---:|']
    sync_summary=[]
    for a in 'CDBA':
        names=sorted({x['name'] for t in rows(a) for x in t['sync_events']})
        for name in names:
            ev=[x for t in rows(a) for x in t['sync_events'] if x['name']==name]
            z=dict(arm=a,api=name,calls_per_window=len(ev)/3,mean_us=st.mean(x['duration_us'] for x in ev),median_us=st.median(x['duration_us'] for x in ev),
                   gpu_overlap_fraction=sum(x['overlap_gpu_work_us'] for x in ev)/sum(x['duration_us'] for x in ev),boundary_calls_per_window=sum(x['boundary_1pct'] for x in ev)/3)
            sync_summary.append(z)
            lines.append(f"| {LABELS[a]} | {name} | {f(z['calls_per_window'],1)} | {f(z['mean_us'])} | {f(z['median_us'])} | {100*z['gpu_overlap_fraction']:.2f}% | {f(z['boundary_calls_per_window'],2)} |")
    write_csv(OUT/'sync_summary.csv',sync_summary)
    lines+=['','Graph waits are distributed throughout the region: 70 event synchronizations across 70 decode intervals, consistent with one per iteration on average, not one per kernel or just a boundary artifact. Eager has 70 event synchronizations plus 140 stream synchronizations. The graph event waits dominate summed API duration and mostly overlap observed GPU work. The overlap calculation uses all worker GPU streams; it does not identify which event or stream is being waited on.','',
      'CUDA documents event synchronization as host blocking until the event completes, and stream synchronization as blocking until stream work completes. Those semantics and the recorded overlap support a host-wait interpretation. API arguments/event handles and blocking flags are not exported here, so the exact dependency and whether the host sleeps or polls are unresolved. [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).','',
      'Thus summed CUDA API duration includes waiting while the GPU works; it is not equivalent to host activity that leaves the GPU idle. Counts, blocking waits, asynchronous submission time and GPU gaps must remain separate.','',
      '## 8. Profiler perturbation','',
      '| Arm | Cadence perturbation % | Runtime API records/window | Runtime kernel launches/window | Selected API + GPU records/window |','|---|---:|---:|---:|---:|']
    for a in 'CDBA':
        h=[h for h in r['profiler_event_association'] if h['arm']==a and h['output_len']==128]
        lines.append(f'| {LABELS[a]} | '+' | '.join(f(st.mean(x[k] for x in h),2) for k in ['cadence_change_pct','runtime_api_count','kernel_launch_count','selected_record_count'])+' |')
    lines+=['', 'Across the 12 primary captures, descriptive Pearson correlations with measured cadence perturbation are '+', '.join(f"{k}: {v:.3f}" for k,v in r['profiler_pearson_primary'].items())+'. These are confounded by execution mode, batch/padding and separate-run drift; they do not estimate per-event instrumentation cost or establish causation. Selected records include runtime and driver layers plus kernels/copies/memsets; this is not a count of all trace-buffer events.','',
      'Eager’s higher event volume is consistent with larger software-tracing cost. Kernel record counts alone are similar between modes. The unprofiled serving effect rejects a purely profiler-created speedup; stable repeated/output256 structure does not reject systematic profiler inflation of gaps. Profiler perturbation is supporting evidence only.','',
      '## 9. Output256 falsification check','',
      '| B8 output256 | Graph | Eager | Eager/graph |','|---|---:|---:|---:|']
    for key,label,source in [('decode_ms','Unprofiled cadence ms','unprofiled_aggregates'),('gpu_busy_us_per_token','Observed GPU-work union µs/token','trace_aggregates'),('inter_kernel_gap_us_per_token','Observed gaps µs/token','trace_aggregates')]:
        a=mean('C',key,source,256);b=mean('D',key,source,256);lines.append(f'| {label} | {f(a)} | {f(b)} | {f(b/a)} |')
    lines+=['','The pattern persists with two traces per arm, even though eager GPU-work union is slightly lower. This is inconsistent with much greater eager GPU computation as the primary explanation. Only one unprofiled restart per arm exists; the same early decode window is analyzed. This is a limited falsification check, not an independent context-length scaling study.','',
      '## 10. B9 padding case','',
      'B8 bucket8→B9 bucket12 graph changes useful batch and padding together. Kernel time per step increases 1.47%; copy-union time per step increases 18.0%. Kernel-grid dimensions change for KV indices/RoPE (8→12), normalization/KV store (2→3), activation (19→29) and some copies. Dominant GEMM grids remain unchanged; two fill kernels are added. Launch grids do not specify complete tensor shapes.','',
      f"Graph gap time per token changes {mean('C','inter_kernel_gap_us_per_token'):.3f}→{mean('B','inter_kernel_gap_us_per_token'):.3f} µs (per-step {mean('C','inter_kernel_gap_us_per_token')*8:.3f}→{mean('B','inter_kernel_gap_us_per_token')*9:.3f} µs). Busy fraction remains {100*mean('C','gpu_busy_fraction'):.2f}%→{100*mean('B','gpu_busy_fraction'):.2f}%, with 70 graph launches in both. The observed name-sequence similarity is 98.09% across all cross-trace comparisons. The core dense-replay versus fragmented-eager pattern survives; an isolated padding cost is not identifiable.",'',
      '## 11. Mechanism conclusion and claim level','',
      '**LEVEL 2 — STRONG MECHANISM EVIDENCE.** The primary B8 control combines a repeated 3.244× unprofiled cadence effect with nearly the same observed GPU computation, 16.643× more eager runtime kernel-launch calls, 28.276× larger recorded gaps, high kernel-sequence similarity, and event-correlated host submission activity. Together these strongly support repeated host/runtime launch and orchestration structure as the primary mechanism.','',
      'Event correlation increases specificity beyond name/count association. It does not demonstrate that the observed delay would disappear under an isolated host-dispatch intervention, nor that all GPU dependencies and records are accounted for. We therefore do not claim level 3 or assign an exact number of real milliseconds to CPU overhead.','',
      '## 12. Remaining uncertainty','',
      '- Nsight CUDA/NVTX completeness warnings remain; all selected kernels match APIs, but jointly missing records are possible.','- Software tracing perturbs eager much more than graph. No unbiased unprofiled gap or occupancy estimate is available.','- CPU scheduling, Python/framework dispatch, driver work and device dependency delays are not separately identified.','- The submission window contains boundary-crossing/asynchronous work; normalization is by 70×useful B token equivalents, not exact request attribution.','- Kernel time is not FLOPs, memory bandwidth or intrinsic efficiency. Equal names and high sequence similarity do not prove equal tensors.','- Three primary traces per arm and two secondary traces bound precision. Cross-trace SD and raw observations are reported without treating thousands of events as independent replications.','',
      '## 13. OR and service-rate implications','',
      'Use **μ(B, m)**, with useful batch B and execution regime m, rather than a smooth batch-only μ(B). Define units explicitly: cohort decode cadence is milliseconds per step; aggregate useful decode service is B divided by step time. The reported throughput averages reciprocals within restart/cohort summaries, so its ratio need not exactly equal the ratio of mean intervals.','',
      'At B8, graph serves about 2,566.9 useful decode tokens/s versus 791.2 for forced eager. The controlled mode change at fixed B shows that the execution regime can produce a roughly 3.24× cadence change without a B8→B9 transition. That intervention supports a policy-level serving effect; detailed attribution to individual host operations remains level 2.','',
      'Empirical service curves should record regime, bucket, model/hardware and workload context. Batching models should include regime eligibility and padding rather than interpolate smoothly across a graph/eager boundary. Scheduler optimization should evaluate the measured regime-specific service rate and latency tradeoff before preferring a larger batch. Queueing simulations should select the appropriate μ(B,m) at each modeled state and retain uncertainty and transition costs where measured. Prediction should flag extrapolation across unsupported regimes. This phase adds no scheduler algorithm, changes no historical simulation, and does not infer unmeasured switching costs.','',
      '## 14. Targeted experiment decisions','',
      '**MICROBENCHMARK NOT NECESSARY** for the intended level-2 claim. Existing controlled SGLang comparisons, direct event correlation, sequence analysis and the output256 check discriminate the main alternatives. A generic small-operation replay benchmark would demonstrate a possibility but would not resolve this workload’s profiler coverage or dispatch/dependency attribution. If a level-3 claim is later required, design a targeted intervention and an independent coverage check first.','',
      '**NSIGHT COMPUTE NOT NEEDED at this stage.** GPU-work and dominant GEMM differences are much too small descriptively to motivate kernel-efficiency analysis as the leading explanation. Modest small-kernel differences do not justify broad counter collection now.','',
      '## 15. Figures, reproducibility and final recommendation','',
      'Reproduce from the repository root with the existing environment: `python -m profiling.analyze_phase2e`, then `python -m profiling.report_phase2e`. The first command reads retained Phase 2D SQLite files; it starts no server. Tests use `python -m unittest discover -s tests` and `python src/validate.py`. Input hashes and worker/region selections are in input_manifest.json; validation_checks.json records preservation and test results.','',
      'The event-level gap association tables are retained locally as ignored compressed CSVs and are reproducible from the hashed inputs. Compact per-trace metrics, sync events, full kernel names, hypothesis/effect tables, sequence comparisons, figures and report are committed. Figures A/C/D reuse unchanged Phase 2D primary measurements; B/E/F are new.','']
    for name,caption in [('A_service_effect','A — unprofiled serving effect'),('B_work_and_gaps','B — recorded GPU work and gaps'),('C_launch_calls','C — observed runtime launches'),('D_busy_fraction','D — observed GPU busy fraction'),('E_gap_ecdf','E — per-trace B8 gap distributions'),('F_conceptual_timeline','F — conceptual model, not literal trace data')]:lines+= [f'![{caption}](figures/{name}.png)','']
    lines+=['**MECHANISM ANALYSIS COMPLETE — LEVEL 2 SUPPORTED.** Use execution-regime-aware service modeling. Retain completeness and asymmetric-profiler limits; do not promote the result to exact causal accounting. No additional GPU experiment is recommended for this claim.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('Phase 2E report written')


if __name__=='__main__':main()
