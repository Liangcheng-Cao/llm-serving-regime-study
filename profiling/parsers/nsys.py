import csv
import io
import sqlite3
from collections import Counter

from ..core import distribution, normalize


def parse_summary(text, kind):
    """Nsight CSV contains optional progress lines before its real header."""
    lines = text.splitlines()
    count_key = "Num Calls" if kind == "api" else "Instances"
    for index, line in enumerate(lines):
        header = next(csv.reader([line]))
        if "Name" in header and "Total Time (ns)" in header and count_key in header:
            break
    else:
        raise ValueError("Unsupported/missing Nsight summary header")
    rows=[]
    for row in csv.DictReader(io.StringIO("\n".join(lines[index:]))):
        if not row.get("Name"):
            continue
        rows.append(dict(name=row["Name"],count=int(row[count_key]),total_ns=int(row["Total Time (ns)"])))
    if not rows or any(x["count"] < 1 or x["total_ns"] < 0 for x in rows):
        raise ValueError("Empty or invalid Nsight metrics")
    result=dict(count=sum(x["count"] for x in rows),total_ns=sum(x["total_ns"] for x in rows),
                dominant=sorted(rows,key=lambda x:x["total_ns"],reverse=True)[:10])
    if kind=="api":
        sync=[x for x in rows if "Synchronize" in x["name"]]
        graph=[x for x in rows if "GraphLaunch" in x["name"]]
        result.update(sync_count=sum(x["count"] for x in sync),sync_total_ns=sum(x["total_ns"] for x in sync),
                      graph_launch_count=sum(x["count"] for x in graph))
    return result


def inspect_sqlite(path, worker_pid, expected_graph):
    """Trace validation and temporally contained completion-window diagnostics.

    No claim of causal GPU iteration alignment: overlap crosses window boundaries.
    Full CSV statistics and temporal window statistics are deliberately separate.
    """
    db=sqlite3.connect(f"file:{path}?mode=ro",uri=True)
    try:
        db.row_factory=sqlite3.Row
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        needed={"StringIds","NVTX_EVENTS","CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_RUNTIME"}
        if not needed <= tables:
            raise ValueError(f"Missing trace tables: {needed-tables}")
        strings={r["id"]:r["value"] for r in db.execute("SELECT * FROM StringIds")}
        events=[]
        for r in db.execute("SELECT * FROM NVTX_EVENTS"):
            r=dict(r)
            name=r.get("text") or strings.get(r.get("textId"),"")
            if name.startswith("GPU_MECH_MEASUREMENT_COMPLETION:") and r.get("end") is not None:
                events.append((name,r))
        if len(events)!=1:
            raise ValueError(f"Expected one measurement NVTX range; got {len(events)}")
        name, event=events[0]
        lo,hi=event["start"],event["end"]
        nvtx_pid=(event["globalTid"] >> 24) & 0xffffff
        if nvtx_pid != worker_pid:
            raise ValueError("NVTX range belongs to wrong process")
        kernels=[dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL")]
        apis=[dict(r) for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME")]
        kernels=[r for r in kernels if ((r["globalPid"]>>24)&0xffffff)==worker_pid]
        apis=[r for r in apis if ((r["globalTid"]>>24)&0xffffff)==worker_pid]
        if not kernels or not apis:
            raise ValueError("No CUDA kernels/APIs in verified server worker")
        window_kernels=[r for r in kernels if lo<=r["start"] and r["end"]<=hi]
        window_apis=[r for r in apis if lo<=r["start"] and r["end"]<=hi]
        graph_calls=[r for r in window_apis if "GraphLaunch" in strings.get(r["nameId"],"")]
        if bool(graph_calls)!=expected_graph:
            raise ValueError("Steady-window CUDA graph launch evidence disagrees with verified mode")
        if not window_kernels:
            raise ValueError("No kernels inside measurement window")
        durations=[r["end"]-r["start"] for r in window_kernels]
        sync=[r for r in window_apis if "Synchronize" in strings.get(r["nameId"],"")]
        names=Counter(strings.get(r.get("demangledName"),str(r.get("shortName"))) for r in window_kernels)
        diagnostic_tables=[t for t in tables if "DIAGNOSTIC" in t or "ANALYSIS" in t]
        diagnostics={t:[dict(r) for r in db.execute('SELECT * FROM "'+t+'" LIMIT 100')] for t in diagnostic_tables}
        return dict(valid=True,worker_pid=worker_pid,nvtx_pid=nvtx_pid,nvtx_range=name,
                    window_ns=[lo,hi],scope="temporally contained events in completion window; boundary overlap excluded; not causal per-iteration attribution",
                    kernel_count=len(durations),kernel_total_ns=sum(durations),kernel_duration_ns=distribution(durations),
                    cuda_runtime_api_count=len(window_apis),cuda_runtime_api_total_ns=sum(r["end"]-r["start"] for r in window_apis),
                    synchronization_count=len(sync),synchronization_total_ns=sum(r["end"]-r["start"] for r in sync),
                    graph_launch_count=len(graph_calls),dominant_kernel_counts=names.most_common(10),
                    streams=sorted({r["streamId"] for r in window_kernels}),
                    whole_capture_worker_kernel_count=len(kernels),sqlite_tables=sorted(tables),diagnostics=diagnostics)
    finally:
        db.close()
