"""Observed temporal metrics for the inspected Nsight 2026.1.3 SQLite schema."""
from collections import defaultdict, Counter
import json
import sqlite3
from .core import distribution

LABELS={"C":"B8 graph","D":"B8 forced eager","A":"B9 natural eager","B":"B9 graph bucket12"}


def clip(intervals, region):
    lo,hi=region
    if hi<=lo:
        raise ValueError("Invalid region")
    out=[]
    for start,end in intervals:
        if end<start:
            raise ValueError("Negative interval")
        a,b=max(start,lo),min(end,hi)
        if b>a:
            out.append((a,b))
    return out


def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if b<a:
            raise ValueError("Negative interval")
        if a==b:
            continue
        if merged and a<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:
            merged.append((a,b))
    return merged


def length(intervals):
    return sum(b-a for a,b in intervals)


def timeline(intervals, region):
    clipped=clip(intervals,region); merged=union(clipped)
    span=region[1]-region[0]
    gaps=[b[0]-a[1] for a,b in zip(merged,merged[1:])]
    kernel_span=merged[-1][1]-merged[0][0] if merged else 0
    busy=length(merged); summed=length(clipped)
    return dict(busy_ns=busy,inactive_ns=span-busy,busy_fraction=busy/span,
                inactive_fraction=1-busy/span,overlap_excess_ns=summed-busy,
                interior_gap_ns=sum(gaps),interior_gap_fraction=sum(gaps)/kernel_span if kernel_span else None,
                gap_distribution_ns=distribution(gaps) if gaps else None,
                max_gap_ns=max(gaps,default=0),edge_gap_ns=span-busy-sum(gaps),merged_segments=len(merged))


def family(name):
    n=name.lower()
    if "rmsnorm" in n or "layernorm" in n:
        return "normalization"
    if "store_kvcache" in n or "reshape_and_cache" in n:
        return "KV/cache"
    if "gemm" in n or "matmul" in n or "splitkreduce" in n:
        return "GEMM/matmul"
    if "attention" in n or "batchprefillwithpaged" in n or "batchdecodewith" in n or "mergestates" in n:
        return "attention"
    if "argmax" in n or "sampling" in n or "topp" in n or "topk" in n:
        return "sampling"
    if "elementwise" in n or "act_and_mul" in n:
        return "elementwise"
    return "unknown"


def api_group(name):
    if name.startswith("cudaGraphLaunch"):
        return "graph_launch"
    if name.startswith(("cudaLaunchKernel","cudaLaunchCooperativeKernel")):
        return "kernel_launch"
    if name.startswith("cuda") and "Synchronize" in name:
        return "sync"
    return "other"


def extract(database, manifest):
    c=manifest["config"]; measured=next(x for x in manifest["cohorts"] if not x["warmup"])
    worker=measured["worker_pid"]; batch=c["batch_size"]; tokens=batch*70
    db=sqlite3.connect(f"file:{database}?mode=ro",uri=True); db.row_factory=sqlite3.Row
    try:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        names=("StringIds","NVTX_EVENTS","CUPTI_ACTIVITY_KIND_RUNTIME","CUPTI_ACTIVITY_KIND_KERNEL",
               "CUPTI_ACTIVITY_KIND_MEMCPY","CUPTI_ACTIVITY_KIND_MEMSET")
        schema={t:[r[1] for r in db.execute(f'PRAGMA table_info("{t}")')] for t in names}
        for t,cols in {"StringIds":["id","value"],"NVTX_EVENTS":["start","end","globalTid","text","textId"],
                       "CUPTI_ACTIVITY_KIND_RUNTIME":["start","end","globalTid","nameId","correlationId"],
                       "CUPTI_ACTIVITY_KIND_KERNEL":["start","end","globalPid","streamId","correlationId","demangledName","gridX","blockX"]}.items():
            if not set(cols)<=set(schema[t]):
                raise ValueError(f"Missing required schema columns: {t}")
        strings={r["id"]:r["value"] for r in db.execute("SELECT * FROM StringIds")}
        def owned(table,pidkey):
            if table not in tables:return []
            return [dict(r) for r in db.execute(f'SELECT * FROM "{table}"') if ((r[pidkey]>>24)&0xffffff)==worker]
        events=owned("NVTX_EVENTS","globalTid")
        for e in events:e["name"]=e.get("text") or strings.get(e.get("textId"),"")
        marker={}
        for edge in ("BEGIN","END"):
            matches=[e for e in events if e["name"]==f"GPU_MECH_MEASUREMENT_SUBMIT_{edge}:{measured['cohort']}:B{batch}"]
            if len(matches)!=1:raise ValueError(f"Missing/duplicate submission marker: {edge}")
            marker[edge]=matches[0]["start"]
        region=(marker["BEGIN"],marker["END"])
        if region[1]<=region[0]:raise ValueError("Invalid submission window")
        completion=[e for e in events if e["name"].startswith("GPU_MECH_MEASUREMENT_COMPLETION:") and e.get("end")]
        if len(completion)!=1:raise ValueError("Missing completion window")
        api=owned("CUPTI_ACTIVITY_KIND_RUNTIME","globalTid")
        for r in api:r["name"]=strings[r["nameId"]]
        kernels=owned("CUPTI_ACTIVITY_KIND_KERNEL","globalPid")
        copies=owned("CUPTI_ACTIVITY_KIND_MEMCPY","globalPid")
        memsets=owned("CUPTI_ACTIVITY_KIND_MEMSET","globalPid")
        def intersects(r):return r["start"]<region[1] and r["end"]>region[0]
        selected=[r for r in kernels if intersects(r)]
        if not selected:raise ValueError("No GPU kernels in submission window")
        spans=lambda rows:[(r["start"],r["end"]) for r in rows]
        kt=timeline(spans(selected),region)
        allwork=timeline(spans(kernels+copies+memsets),region)
        metrics=dict(region_ms=(region[1]-region[0])/1e6,kernel_count=len(selected),
                     kernel_count_per_token=len(selected)/tokens,kernel_sum_us_per_token=length(clip(spans(selected),region))/tokens/1000,
                     kernel_busy_us_per_token=kt["busy_ns"]/tokens/1000,kernel_busy_fraction=kt["busy_fraction"],
                     gpu_busy_us_per_token=allwork["busy_ns"]/tokens/1000,gpu_busy_fraction=allwork["busy_fraction"],
                     gpu_inactive_us_per_token=allwork["inactive_ns"]/tokens/1000,gpu_inactive_fraction=allwork["inactive_fraction"],
                     inter_kernel_gap_us_per_token=kt["interior_gap_ns"]/tokens/1000,
                     inter_kernel_gap_fraction=kt["interior_gap_fraction"],edge_gap_us=kt["edge_gap_ns"]/1000,
                     kernel_overlap_excess_us_per_token=kt["overlap_excess_ns"]/tokens/1000,
                     kernel_boundary_crossings=sum(not(region[0]<=r["start"] and r["end"]<=region[1]) for r in selected),
                     active_kernel_streams=len({r["streamId"] for r in selected}),
                     copy_union_us_per_token=length(union(clip(spans(copies),region)))/tokens/1000,
                     memset_union_us_per_token=length(union(clip(spans(memsets),region)))/tokens/1000,
                     observed_copy_bytes=sum(r["bytes"] for r in copies if intersects(r)))
        full=[r["end"]-r["start"] for r in selected if region[0]<=r["start"] and r["end"]<=region[1]]
        for key,value in distribution(full).items():
            if key in ("mean","median","p95","sd"):metrics[f"kernel_duration_{key}_us"]=None if value is None else value/1000
        gd=kt["gap_distribution_ns"]
        for key in ("mean","median","p95","sd"):
            metrics[f"gap_{key}_us"]=gd[key]/1000 if gd and gd[key] is not None else None
        metrics["gap_max_us"]=kt["max_gap_ns"]/1000
        # Runtime and driver calls coexist in this version's RUNTIME table.
        started=[r for r in api if region[0]<=r["start"]<region[1]]
        runtime=[r for r in started if r["name"].startswith("cuda")]
        driver=[r for r in started if r["name"].startswith("cu") and not r["name"].startswith("cuda")]
        apisummaries={}
        groups={"runtime_api":runtime,"driver_api":driver,**{g:[r for r in runtime if api_group(r["name"])==g] for g in ("kernel_launch","graph_launch","sync")}}
        for group,rows in groups.items():
            d=[min(r["end"],region[1])-r["start"] for r in rows]
            stats=distribution(d) if d else None
            metrics[group+"_count"]=len(rows); metrics[group+"_calls_per_token"]=len(rows)/tokens
            metrics[group+"_us_per_token"]=sum(d)/tokens/1000
            for key in ("mean","median","p95"):
                metrics[group+"_duration_"+key+"_us"]=stats[key]/1000 if stats else None
            apisummaries[group]=dict(count=len(rows),sum_ns=sum(d),duration_ns=stats,names=dict(Counter(r["name"] for r in rows)))
        expected=70 if measured["verified_mode"]!="eager" else 0
        if metrics["graph_launch_count"]!=expected:
            raise ValueError(f"Expected {expected} graph launches, got {metrics['graph_launch_count']}")
        metrics["graph_launches_per_iteration"]=metrics["graph_launch_count"]/70
        composition={}; streams={}
        for r in selected:
            name=strings[r["demangledName"]]; duration=min(r["end"],region[1])-max(r["start"],region[0])
            item=composition.setdefault(name,dict(name=name,family=family(name),count=0,clipped_total_ns=0,shapes=Counter()))
            item["count"]+=1;item["clipped_total_ns"]+=duration
            shape=tuple(r[k] for k in ("gridX","gridY","gridZ","blockX","blockY","blockZ"))
            item["shapes"][str(shape)]+=1
        for stream in sorted({r["streamId"] for r in selected}):
            rows=[r for r in selected if r["streamId"]==stream]
            streams[str(stream)]=dict(count=len(rows),**timeline(spans(rows),region))
        families=defaultdict(lambda:dict(count=0,total_ns=0))
        for x in composition.values():
            families[x["family"]]["count"]+=x["count"];families[x["family"]]["total_ns"]+=x["clipped_total_ns"]
        for f in ("GEMM/matmul","attention","normalization","sampling","elementwise","KV/cache","unknown"):
            metrics["family_"+f.replace("/","_")+"_us_per_token"]=families[f]["total_ns"]/tokens/1000
        cregion=(completion[0]["start"],completion[0]["end"])
        comparison=timeline(spans(kernels+copies+memsets),cregion)
        metrics["completion_window_busy_fraction"]=comparison["busy_fraction"]
        metrics["window_busy_fraction_delta_pp"]=100*(allwork["busy_fraction"]-comparison["busy_fraction"])
        correlation_ids={r["correlationId"] for r in api}
        unmatched=sum(r["correlationId"] not in correlation_ids for r in selected)
        if unmatched:raise ValueError(f"{unmatched} kernels lack observed API correlations")
        return dict(metrics=metrics,region_ns=list(region),completion_region_ns=list(cregion),
                    region_definition="GPU activity observed during decode submission starts 10→80; 70 intervals; clipped across all worker streams",
                    useful_tokens=tokens,worker_pid=worker,replay_bucket=measured["replay_bucket"],
                    apis=apisummaries,kernel_composition=sorted(composition.values(),key=lambda x:x["clipped_total_ns"],reverse=True),
                    families=dict(families),streams=streams,schema=schema,
                    quality=manifest["trace"]["completion_window"]["quality"],unmatched_kernel_correlations=unmatched)
    finally:
        db.close()
