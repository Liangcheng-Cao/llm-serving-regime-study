"""Summarize fresh validation evidence only; never update historical service tables."""
import csv
import json
from pathlib import Path

from .core import Config, save, verify_server_info
from .run_decode_experiment import STUDY
from .parsers.nsys import inspect_sqlite


def summarize_run(manifest):
    if not manifest.get("valid") or manifest.get("status") != "valid":
        return None
    c=manifest["config"]
    measured=[x for x in manifest["cohorts"] if not x["warmup"]]
    if len(measured)!=c["repeats"] or not all(x.get("valid") for x in measured):
        raise ValueError("Incomplete valid run")
    mode={(x["verified_mode"],x["replay_bucket"]) for x in measured}
    if len(mode)!=1:
        raise ValueError("Mixed verified modes")
    verified,bucket=next(iter(mode))
    interval=manifest["summary"]["cohort_mean_interval_ms"]
    return dict(run_id=c["run_id"],batch_size=c["batch_size"],cap=c["max_decode_bs"],
                force_decode_eager=c["force_decode_eager"],output_len=c["output_len"],
                profiled=c["profiling"],cohorts=len(measured),verified_mode=verified,replay_bucket=bucket,
                mean_interval_ms=interval["mean"],median_cohort_interval_ms=interval["median"],
                sd_cohort_interval_ms=interval["sd"],cv=interval["cv"],
                useful_decode_tokens_s=c["batch_size"]*1000/interval["mean"],
                mean_cohort_e2e_s=manifest["summary"]["cohort_e2e_s"]["mean"])


def main():
    manifests={p.parent.name:json.loads(p.read_text()) for p in sorted((STUDY/"runs").glob("*/manifest.json"))}
    rows=[]; config_checks={}; invalid=[]
    for rid,m in manifests.items():
        row=summarize_run(m)
        if row is None:
            invalid.append(dict(run_id=rid,status=m.get("status"),failures=m.get("failures")))
            continue
        info=json.loads((STUDY/"runs"/rid/"server_info.json").read_text())
        config_checks[rid]=verify_server_info(Config(**m["config"]),info)
        rows.append(row)
    if not rows:
        raise ValueError("No valid runs")
    with (STUDY/"results.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    overhead=[]
    for arm in ("a","b"):
        base=next(r for r in rows if r["run_id"]==f"sanity-{arm}-128")
        prof=next(r for r in rows if r["run_id"]==f"trace-{arm}-128")
        overhead.append(dict(arm=arm.upper(),baseline=base["run_id"],profile=prof["run_id"],
                             baseline_ms=base["mean_interval_ms"],profile_ms=prof["mean_interval_ms"],
                             interval_change_pct=100*(prof["mean_interval_ms"]/base["mean_interval_ms"]-1),
                             e2e_change_pct=100*(prof["mean_cohort_e2e_s"]/base["mean_cohort_e2e_s"]-1),
                             limitation="3 baseline cohorts, 1 profiled cohort, distinct server starts; apparent perturbation includes drift"))
    traces={rid:m["trace"] for rid,m in manifests.items() if m.get("valid") and "trace" in m}
    quality={}
    for rid,t in traces.items():
        measured=next(x for x in manifests[rid]["cohorts"] if not x["warmup"])
        reviewed=inspect_sqlite(STUDY/"summaries"/rid/"trace.sqlite",measured["worker_pid"],measured["verified_mode"]!="eager")
        quality[rid]=reviewed["quality"]
        save(STUDY/"summaries"/rid/"quality_review.json",quality[rid])
    result=dict(scope="Fresh Phase 2B/2C pipeline validation, not mechanism interpretation",runs=rows,
                invalid_runs=invalid,overhead=overhead,resolved_config_checks=config_checks,trace_quality=quality,
                trace_validation={rid:dict(valid=t["valid"],bytes=t["bytes"],worker_pid=t["completion_window"]["worker_pid"],
                    kernel_count=t["completion_window"]["kernel_count"],graph_launch_count=t["completion_window"]["graph_launch_count"],
                    cuda_runtime_api_count=t["completion_window"]["cuda_runtime_api_count"]) for rid,t in traces.items()})
    save(STUDY/"validation_summary.json",result)
    print(json.dumps(dict(runs=rows,overhead=overhead,traces=result["trace_validation"]),indent=2))


if __name__=="__main__":
    main()
