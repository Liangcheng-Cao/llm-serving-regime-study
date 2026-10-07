"""Bounded independent-restart campaign. No automatic retries or concurrent GPU jobs."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from .core import Config, ARMS, save
from .run_decode_experiment import execute,STUDY
from .phase2d_metrics import extract

OUT=STUDY/"phase2d"


def jobs(stage):
    if stage=="primary":
        for block,order in enumerate(("CDAB","BADC","DCBA"),1):
            for i,arm in enumerate(order):
                for profile in ((False,True) if (block+i)%2 else (True,False)):
                    yield block,arm,profile,128
    elif stage=="secondary":
        for arm in "CD":yield 1,arm,False,256
        for block,order in enumerate(("CD","DC"),1):
            for arm in order:yield block,arm,True,256


async def main(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    for block,arm,profile,nout in jobs(stage):
        rid=f"d-{nout}-{arm.lower()}-b{block}-{'trace' if profile else 'plain'}"
        index=OUT/"trace_manifest.jsonl"
        if index.exists():
            completed={x["run_id"]:x for x in map(json.loads,index.read_text().splitlines())}
            if rid in completed:
                if completed[rid]["accepted"]:continue
                raise RuntimeError(f"Previous failure retained: {rid}; explicit audit required")
        b,cap,forced=ARMS[arm]
        config=Config(batch_size=b,max_decode_bs=cap,force_decode_eager=forced,output_len=nout,
                      run_id=rid,profiling=profile,repeats=1 if profile else 3)
        print("CAMPAIGN START",rid,flush=True)
        manifest=await execute(config,OUT)
        entry=dict(run_id=rid,arm=arm,block=block,profiled=profile,output_len=nout,
                   accepted=manifest["valid"],manifest=f"runs/{rid}/manifest.json",failure=None)
        if manifest["valid"] and profile:
            try:
                parsed=extract(OUT/"summaries"/rid/"trace.sqlite",manifest)
                save(OUT/"summaries"/rid/"phase2d_metrics.json",parsed)
                entry["metrics"]=f"summaries/{rid}/phase2d_metrics.json"
            except Exception as exc:
                entry.update(accepted=False,failure=str(exc))
        with index.open("a") as stream:stream.write(json.dumps(entry)+"\n")
        print("CAMPAIGN END",rid,entry["accepted"],entry["failure"],flush=True)
        if not entry["accepted"]:raise RuntimeError(f"Invalid run retained: {rid}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["primary","secondary"],required=True)
    args=p.parse_args()
    if args.stage=="secondary":
        entries=[json.loads(l) for l in (OUT/"trace_manifest.jsonl").read_text().splitlines()]
        for arm in "ABCD":
            if sum(x["accepted"] and x["profiled"] and x["output_len"]==128 and x["arm"]==arm for x in entries)<3:
                p.error("Secondary campaign requires three valid primary traces per arm")
    asyncio.run(main(args.stage))
