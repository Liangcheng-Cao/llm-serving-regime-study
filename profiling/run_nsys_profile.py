"""Nsight validation export. Invocation: python -m profiling.run_nsys_profile --arm A ..."""
import hashlib
from pathlib import Path
import subprocess
import sys

from .core import save, normalize
from .parsers.nsys import parse_summary, inspect_sqlite


def checked(command):
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    if result.returncode:
        raise RuntimeError(f"Export failed: {command}: {result.stdout[-3000:]}")
    return "\n".join(line.rstrip() for line in result.stdout.splitlines()).rstrip()+"\n"


def export_trace(output, run_id, manifest):
    report=output/"traces"/(run_id+".nsys-rep")
    if not report.exists():
        raise ValueError("Nsight report missing after capture shutdown")
    summary_dir=output/"summaries"/run_id
    summary_dir.mkdir(parents=True,exist_ok=False)
    database=summary_dir/"trace.sqlite"
    commands=[]
    command=["nsys","export","--type=sqlite","--output="+str(database),str(report)]
    commands.append(command)
    (summary_dir/"export.log").write_text(checked(command))
    parsed={}
    for report_name,kind in (("cuda_api_sum","api"),("cuda_gpu_kern_sum","kernel"),("nvtx_pushpop_trace",None),("cuda_gpu_trace",None)):
        command=["nsys","stats","--report",report_name,"--format","csv",str(database)]
        commands.append(command)
        text=checked(command)
        (summary_dir/(report_name+".csv")).write_text(text)
        if kind:
            parsed[kind]=parse_summary(text,kind)
    measured=[x for x in manifest["cohorts"] if not x["warmup"]]
    window=inspect_sqlite(database,measured[0]["worker_pid"],measured[0]["verified_mode"]!="eager")
    window["temporal_normalizations"]={key:normalize(window[key],generated_tokens=measured[0]["useful_decode_tokens"],
                 decode_iterations=70,requests=measured[0]["actual_batch"]) for key in
                 ("kernel_count","kernel_total_ns","cuda_runtime_api_count","cuda_runtime_api_total_ns",
                  "synchronization_count","synchronization_total_ns","graph_launch_count")}
    result=dict(valid=window["valid"],report=str(report),bytes=report.stat().st_size,
                sha256=hashlib.sha256(report.read_bytes()).hexdigest(),commands=commands,
                whole_capture_summaries=parsed,completion_window=window,
                normalization_note="Window counts are temporal diagnostics; full capture includes prefill. Causal per-token kernel attribution remains pending.")
    save(summary_dir/"trace_summary.json",result)
    return result


if __name__=="__main__":
    from .run_decode_experiment import main
    if "--profile" not in sys.argv:
        sys.argv.append("--profile")
    main()
