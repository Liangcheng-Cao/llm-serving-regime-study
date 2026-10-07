"""Run one fresh fixed-cohort experiment in WSL. Use python -m profiling.run_decode_experiment."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback

from .core import ARMS, Config, new_run_id, save, verify_cohort, distribution
from .instrument import prepare, digest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/gpu-mechanism-study"


def capture(args):
    try:
        p = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return dict(returncode=p.returncode, output=p.stdout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


def environment():
    import torch
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), client_timezone="Asia/Shanghai",
                hostname=socket.gethostname(), platform=platform.platform(), python=sys.version,
                packages={p:importlib.metadata.version(p) for p in ("torch","sglang","aiohttp")},
                torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                gpu=capture(["nvidia-smi"]), toolkit=capture(["/usr/local/cuda/bin/nvcc","--version"]),
                nsys=capture(["nsys","--version"]), git_commit=capture(["git","-C",str(ROOT),"rev-parse","HEAD"]),
                git_status=capture(["git","-C",str(ROOT),"status","--porcelain"]))


async def cohort(session, url, config):
    payload = dict(input_ids=[[1000+i]*config.input_len for i in range(config.batch_size)],
                   sampling_params=dict(temperature=0, max_new_tokens=config.output_len, ignore_eos=True), stream=True)
    stamps, finished = {}, {}
    start = time.perf_counter()
    async with session.post(url+"/generate", json=payload) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            obj = json.loads(raw)
            meta = obj["meta_info"]
            idx = obj.get("index", meta.get("id"))
            n = meta["completion_tokens"]
            stamps.setdefault(idx, {}).setdefault(n, time.perf_counter_ns())
            finished[idx] = n
            if meta.get("prompt_tokens") != config.input_len:
                raise ValueError("Prompt length contract failed")
    duration = time.perf_counter()-start
    if len(finished) != config.batch_size or any(n != config.output_len for n in finished.values()):
        raise ValueError("Output length/cohort cardinality contract failed")
    proxy = [(x[81]-x[11])/70/1e6 for x in stamps.values() if 81 in x and 11 in x]
    return dict(e2e_s=duration, output_tokens=config.batch_size*config.output_len,
                full_cohort_output_tokens_s=config.batch_size*config.output_len/duration,
                request_throughput_s=config.batch_size/duration,
                client_stable_interval_ms=distribution(proxy) if len(proxy)==config.batch_size else None,
                client_proxy_note="token 11→81 span/70; not network-chunk ITL")


def profile_command(server, prefix):
    return ["nsys","profile","--trace=cuda,nvtx","--cuda-trace-scope=process-tree",
            "--sample=none","--cpuctxsw=none","--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi","--capture-range-end=stop",
            "--kill=none","--output="+str(prefix),*server]


async def execute(config, output_dir):
    import aiohttp
    import sglang
    config.validate()
    run = output_dir / "runs" / config.run_id
    run.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, config=asdict(config), requested_mode=config.requested_mode,
                  valid=False, status="initializing", cohorts=[], failures=[], environment=environment(),
                  client_command=[sys.executable,*sys.argv], timing_scope="completion endpoints 10→80; 70 intervals")
    save(run/"manifest.json", result)
    proc = None
    scratch = None
    start = time.perf_counter()
    try:
        with socket.socket() as check:
            check.bind(("127.0.0.1",config.port))
        original = Path(sglang.__file__).parent
        scratch, sources = prepare(original, config.run_id)
        result["instrumentation"] = dict(scratch_path=str(scratch), sources=sources,
                                          observer_sha256=digest(Path(__file__).with_name("observer.py")))
        control, obs = run/"control.json", run/"observations.jsonl"
        save(control, dict(cohort="initialization", warmup=True, nvtx=config.nvtx))
        server = config.command(sys.executable)
        trace_prefix = output_dir / "traces" / config.run_id
        trace_prefix.parent.mkdir(parents=True, exist_ok=True)
        command = profile_command(server,trace_prefix) if config.profiling else server
        result.update(server_command=server, profiler_command=command if config.profiling else None)
        env = dict(os.environ, PYTHONPATH=str(scratch), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   NO_PROXY="127.0.0.1,localhost,::1", no_proxy="127.0.0.1,localhost,::1",
                   GPU_MECH_CONTROL=str(control), GPU_MECH_OBSERVATIONS=str(obs),
                   PATH=str(Path(sys.executable).parent)+":/usr/local/cuda/bin:"+os.environ["PATH"])
        for key in ("SGLANG_LOG_FORWARD_ITERS","SGLANG_LOG_DECODE_GRAPH_KEY","SGLANG_RECORD_STEP_TIME"):
            env.pop(key,None)
        with (run/"server.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    cwd=str(scratch), start_new_session=True)
        result["launch_pid"] = proc.pid
        save(run/"manifest.json",result)
        url = f"http://127.0.0.1:{config.port}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            for _ in range(300):
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited {proc.returncode}; inspect server.log")
                try:
                    async with session.get(url+"/health",timeout=aiohttp.ClientTimeout(total=2)) as r:
                        if r.status==200:
                            break
                except (aiohttp.ClientError,asyncio.TimeoutError):
                    pass
                await asyncio.sleep(1)
            else:
                raise TimeoutError("Server startup exceeded 300 health checks")
            result["startup_s"] = time.perf_counter()-start
            async with session.get(url+"/server_info") as r:
                r.raise_for_status()
                save(run/"server_info.json",await r.json())
            print(config.run_id,"server ready",round(result["startup_s"],2),flush=True)
            for index in range(config.warmup+config.repeats):
                warmup = index < config.warmup
                cid = f"{'warmup' if warmup else 'measure'}-{index}"
                save(control,dict(cohort=cid,warmup=warmup,nvtx=config.nvtx))
                if config.profiling and not warmup:
                    async with session.post(url+"/start_profile",json=dict(activities=["CUDA_PROFILER"],with_stack=False,record_shapes=False)) as r:
                        r.raise_for_status()
                client = await cohort(session,url,config)
                if config.profiling and not warmup:
                    async with session.post(url+"/stop_profile") as r:
                        r.raise_for_status()
                await asyncio.sleep(.2)  # excluded: allow tail observer flush and overlap retirement
                observations = [json.loads(line) for line in obs.read_text().splitlines()]
                candidates = [x for x in observations if x["cohort"]==cid and len(x["rows"])>=80]
                if len(candidates)!=1:
                    raise ValueError(f"Expected one complete observer cohort {cid}, got {len(candidates)}")
                verified = verify_cohort(config,candidates[0])
                result["cohorts"].append(dict(cohort=cid,warmup=warmup,**verified,**client))
                save(run/"manifest.json",result)
                print(cid,verified["verified_mode"],verified["replay_bucket"],round(verified["interval_ms"]["mean"],4),flush=True)
            measured = [x for x in result["cohorts"] if not x["warmup"]]
            result.update(valid=True,status="serving_valid",summary=dict(
                cohort_mean_interval_ms=distribution([x["interval_ms"]["mean"] for x in measured]),
                cohort_e2e_s=distribution([x["e2e_s"] for x in measured])))
    except Exception as exc:
        result.update(valid=False,status="invalid")
        result["failures"].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        print("INVALID",config.run_id,str(exc),flush=True)
    finally:
        if proc is not None:
            # Signal only the process group launched by this run; never a shared server.
            try:
                os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=10)
        result["total_run_s"] = time.perf_counter()-start
        if scratch:
            result["installed_sources_unchanged"] = all(digest(original/r)==x["original_sha256"] for r,x in sources.items())
        if config.profiling and result["valid"]:
            from .run_nsys_profile import export_trace
            try:
                result["trace"] = export_trace(output_dir,config.run_id,result)
                result["valid"] = result["trace"]["valid"]
                result["status"] = "valid" if result["valid"] else "invalid_trace"
            except Exception as exc:
                result.update(valid=False,status="invalid_trace")
                result["failures"].append(dict(type=type(exc).__name__,message=str(exc)))
        elif result["valid"]:
            result["status"]="valid"
        save(run/"manifest.json",result)
        with (output_dir/"experiment_manifest.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(run_id=config.run_id,valid=result["valid"],status=result["status"],
                                        manifest=str(run/"manifest.json")),allow_nan=False)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=ARMS,default="A")
    for name in ("model","revision","dtype","graph-policy"):
        parser.add_argument("--"+name)
    for name in ("batch-size","max-decode-bs","input-len","output-len","warmup","repeats","seed","port"):
        parser.add_argument("--"+name,type=int)
    parser.add_argument("--force-decode-eager",action="store_true",default=None)
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--no-nvtx",action="store_true")
    parser.add_argument("--run-id",default=None)
    parser.add_argument("--output-dir",type=Path,default=STUDY)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    batch,cap,forced=ARMS[args.arm]
    values=dict(batch_size=batch,max_decode_bs=cap,force_decode_eager=forced,
                run_id=args.run_id or new_run_id(),profiling=args.profile,nvtx=not args.no_nvtx)
    if args.profile:
        values["repeats"]=1
    for key in ("batch_size","max_decode_bs","force_decode_eager","input_len","output_len","warmup","repeats","seed","port","model","revision","dtype","graph_policy"):
        if getattr(args,key) is not None:
            values[key]=getattr(args,key)
    config=Config(**values).validate()
    if args.dry_run:
        print(json.dumps(dict(config=asdict(config),requested_mode=config.requested_mode,server_command=config.command(sys.executable)),indent=2))
        return
    if sys.platform!="linux":
        parser.error("Execute GPU runs inside WSL/Linux")
    if not asyncio.run(execute(config,args.output_dir.resolve()))["valid"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
