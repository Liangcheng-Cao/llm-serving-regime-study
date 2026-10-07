"""Descriptive cross-trace analysis. Serving effect comes only from unprofiled runs."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics as st

from .core import distribution, save
from .phase2d_metrics import LABELS
from .run_phase2d import OUT

ORDER="CDAB"


def accepted(entries):
    seen=set();result=[]
    for row in entries:
        if row["run_id"] in seen:raise ValueError("Duplicate campaign run ID")
        seen.add(row["run_id"])
        if row.get("accepted") is True:result.append(row)
    return result


def aggregate(rows,keys,metrics):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    output=[]
    for group,items in groups.items():
        for metric in metrics:
            values=[r[metric] for r in items if r.get(metric) is not None]
            if not values:continue
            stats=distribution(values)
            # CV of signed diagnostic deltas uses |mean|; near-zero means can
            # produce huge CV and are interpreted with their absolute SD too.
            stats["cv"]=stats["sd"]/abs(stats["mean"]) if stats["sd"] is not None and stats["mean"] else None
            output.append(dict(zip(keys,group),metric=metric,**stats,p50=stats["median"],
                               warning_cv_gt_10pct=stats["cv"] is not None and stats["cv"]>.1))
    return output


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=keys,lineterminator="\n")
        w.writeheader();w.writerows(rows)


def collect():
    entries=[json.loads(l) for l in (OUT/"trace_manifest.jsonl").read_text().splitlines()]
    trace=[];serving=[];cohorts=[];kernel=[];families=[];quality=[]
    for entry in accepted(entries):
        rid=entry["run_id"];manifest=json.loads((OUT/entry["manifest"]).read_text())
        base=dict(run_id=rid,arm=entry["arm"],label=LABELS[entry["arm"]],block=entry["block"],output_len=entry["output_len"])
        measured=[x for x in manifest["cohorts"] if not x["warmup"]]
        if entry["profiled"]:
            parsed=json.loads((OUT/entry["metrics"]).read_text())
            trace.append(dict(**base,**parsed["metrics"],profiled_completion_ms=measured[0]["interval_ms"]["mean"],
                              profiled_e2e_s=measured[0]["e2e_s"]))
            quality.append(dict(**base,worker_pid=parsed["worker_pid"],**parsed["quality"],
                                report_bytes=manifest["trace"]["bytes"]))
            tokens=parsed["useful_tokens"]
            for k in parsed["kernel_composition"]:
                kernel.append(dict(**base,name=k["name"],family=k["family"],count=k["count"],
                                   time_us_per_token=k["clipped_total_ns"]/tokens/1000,
                                   shapes=json.dumps(k["shapes"],sort_keys=True)))
            for f,values in parsed["families"].items():
                families.append(dict(**base,family=f,count=values["count"],time_us_per_token=values["total_ns"]/tokens/1000))
        else:
            for m in measured:
                cohorts.append(dict(**base,cohort=m["cohort"],decode_ms=m["interval_ms"]["mean"],
                                    within_cohort_p50_ms=m["interval_ms"]["median"],within_cohort_p95_ms=m["interval_ms"]["p95"],
                                    e2e_s=m["e2e_s"],decode_tokens_s=m["useful_tokens_s"],
                                    serving_tokens_s=m["full_cohort_output_tokens_s"],requests_s=m["request_throughput_s"]))
            selected=cohorts[-len(measured):]
            serving.append(dict(**base,**{k:st.mean(x[k] for x in selected) for k in
                            ("decode_ms","e2e_s","decode_tokens_s","serving_tokens_s","requests_s")}))
    return entries,trace,serving,cohorts,kernel,families,quality


def analyze():
    entries,trace,serving,cohorts,kernels,families,quality=collect()
    keys=["output_len","arm"]
    metrics=[k for k in trace[0] if k not in ("run_id","arm","label","block","output_len")]
    tagg=aggregate(trace,keys,metrics)
    smetrics=["decode_ms","e2e_s","decode_tokens_s","serving_tokens_s","requests_s"]
    sagg=aggregate(serving,keys,smetrics)
    cagg=aggregate(cohorts,keys,smetrics+["within_cohort_p50_ms","within_cohort_p95_ms"])
    lookup={(r["output_len"],r["arm"],r["metric"]):r for r in tagg}
    sl={(r["output_len"],r["arm"],r["metric"]):r for r in sagg}
    effects=[]
    pairs=[("primary_B8","C","D"),("practical_B9","B","A"),("eager_scaling_policy_caveat","D","A"),("graph_scaling_padding","C","B")]
    for length in sorted({r["output_len"] for r in trace}):
        for pair,reference,comparison in pairs:
            for source,data,items in (("unprofiled",sl,smetrics),("profiled_observed",lookup,metrics)):
                for metric in items:
                    a=data.get((length,reference,metric));b=data.get((length,comparison,metric))
                    if not a or not b:continue
                    effects.append(dict(pair=pair,output_len=length,source=source,metric=metric,
                                        reference=LABELS[reference],comparison=LABELS[comparison],reference_mean=a["mean"],
                                        comparison_mean=b["mean"],delta=b["mean"]-a["mean"],
                                        ratio=b["mean"]/a["mean"] if a["mean"] else None))
    overhead=[]
    for t in trace:
        b=next((r for r in serving if r["output_len"]==t["output_len"] and r["arm"]==t["arm"] and r["block"]==t["block"]),None)
        if b:
            overhead.append(dict(run_id=t["run_id"],arm=t["arm"],output_len=t["output_len"],block=t["block"],
                                 cadence_change_pct=100*(t["profiled_completion_ms"]/b["decode_ms"]-1),
                                 e2e_change_pct=100*(t["profiled_e2e_s"]/b["e2e_s"]-1)))
    for filename,rows in [("trace_metrics.csv",trace),("aggregate_metrics.csv",tagg),("unprofiled_restart_metrics.csv",serving),
                          ("unprofiled_cohorts.csv",cohorts),("unprofiled_aggregate.csv",sagg),("unprofiled_cohort_aggregate.csv",cagg),
                          ("pairwise_effects.csv",effects),("kernel_composition.csv",kernels),("kernel_families.csv",families),
                          ("profiler_perturbation.csv",overhead)]:write_csv(OUT/filename,rows)
    result=dict(trace_aggregates=tagg,unprofiled_aggregates=sagg,cohort_aggregates=cagg,
                invalid_runs=[e for e in entries if not e["accepted"]],quality=quality,
                overhead_aggregates=aggregate(overhead,keys,["cadence_change_pct","e2e_change_pct"]),
                pairwise_effects=effects)
    save(OUT/"analysis.json",result)
    return result,trace,serving,families,kernels


def figures(trace,serving,families):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder=OUT/"figures";folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":170})
    xs=[0,1,3,4];colors=["#177e89","#d96b4c","#177e89","#d96b4c"]
    labels=["B8 graph\nbucket8","B8 forced\neager","B9 graph\nbucket12","B9 natural\neager"]
    order="CDBA"
    for filename,title,metric,data,ylabel in [
        ("01_unprofiled_decode","Unprofiled serving effect", "decode_ms",serving,"Completed-decode interval (ms)"),
        ("02_gpu_busy","Observed GPU work: interval union", "gpu_busy_us_per_token",trace,"Busy time / useful decode token (µs)"),
        ("03_kernel_gaps","Observed inter-kernel gaps", "inter_kernel_gap_us_per_token",trace,"Gap time / useful decode token (µs)"),
        ("04_launch_calls","Observed runtime kernel-launch calls", "kernel_launch_calls_per_token",trace,"Runtime kernel launches / useful token"),
        ("05_gpu_busy_fraction","Observed GPU busy fraction", "gpu_busy_fraction",trace,"Kernel + copy + memset union / window")]:
        fig,ax=plt.subplots(figsize=(8,4.5),layout="constrained")
        for x,arm,color in zip(xs,order,colors):
            v=[r[metric] for r in data if r["arm"]==arm and r["output_len"]==128]
            ax.bar(x,st.mean(v),color=color,width=.65,alpha=.75)
            ax.scatter(np.linspace(x-.12,x+.12,len(v)),v,c="#152b3c",s=23,zorder=4)
        ax.set(xticks=xs,xticklabels=labels,ylabel=ylabel,title=title)
        ax.text(.5,-.24,"B8 primary pair · B9 padding comparison     |     independent restart points",ha="center",transform=ax.transAxes,fontsize=9)
        if metric!="decode_ms":ax.text(.02,.98,"Profiled observations; completeness unproven",va="top",transform=ax.transAxes,fontsize=9)
        fig.savefig(folder/(filename+".png"));fig.savefig(folder/(filename+".svg"));plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.2),layout="constrained");bottom=[0.]*4
    for family in ["GEMM/matmul","attention","normalization","sampling","elementwise","KV/cache","unknown"]:
        means=[st.mean([r["time_us_per_token"] for r in families if r["output_len"]==128 and r["arm"]==a and r["family"]==family]) for a in order]
        ax.bar(xs,means,bottom=bottom,label=family,width=.65);bottom=[x+y for x,y in zip(bottom,means)]
    ax.set(xticks=xs,xticklabels=labels,ylabel="Observed kernel-duration sum / token (µs)",title="Kernel-family composition (profiled)")
    ax.legend(bbox_to_anchor=(1.02,1),loc="upper left",fontsize=9)
    ax.text(.5,-.2,"Duration sum, not union busy time; names classified conservatively",ha="center",transform=ax.transAxes,fontsize=9)
    fig.savefig(folder/"06_kernel_families.png");fig.savefig(folder/"06_kernel_families.svg");plt.close(fig)


if __name__=="__main__":
    result,trace,serving,fams,kernels=analyze()
    figures(trace,serving,fams)
    print(json.dumps(dict(traces=len(trace),unprofiled_starts=len(serving),
                         invalid=len(result["invalid_runs"]),cv_flags=sum(r["warning_cv_gt_10pct"] for r in result["trace_aggregates"])),indent=2))
