"""CPU-only configuration, verification and metric contracts."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
import re
import statistics as st
import uuid

ARMS = {"A": (9, 8, False), "B": (9, 16, False),
        "C": (8, 8, False), "D": (8, 8, True)}


def new_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:10]


@dataclass(frozen=True)
class Config:
    batch_size: int = 9
    graph_policy: str = "enabled"
    max_decode_bs: int = 8
    force_decode_eager: bool = False
    input_len: int = 128
    output_len: int = 128
    warmup: int = 3
    repeats: int = 3
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    dtype: str = "bfloat16"
    seed: int = 42
    port: int = 30149
    run_id: str = ""
    profiling: bool = False
    nvtx: bool = True

    def validate(self):
        for name in ("batch_size", "max_decode_bs", "input_len", "output_len", "warmup", "repeats"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.graph_policy != "enabled" or self.dtype != "bfloat16":
            raise ValueError("This initial pipeline fixes enabled graph policy and BF16")
        if self.output_len < 82:
            raise ValueError("Output must cover 70 steady intervals and client token 81")
        if not 1024 <= self.port <= 65535 or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid port or seed")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", self.run_id):
            raise ValueError("Unsafe or empty run ID")
        if self.profiling and self.repeats != 1:
            raise ValueError("Validation traces capture exactly one measured cohort")
        if self.profiling and not self.nvtx:
            raise ValueError("Trace validation requires the measurement NVTX range")
        return self

    @property
    def requested_mode(self):
        if self.force_decode_eager:
            return "forced_decode_eager"
        return "graph_eligible" if self.batch_size <= self.max_decode_bs else "natural_eager"

    def command(self, python):
        args = [python, "-m", "sglang.launch_server", "--model-path", self.model,
                "--revision", self.revision, "--dtype", self.dtype,
                "--attention-backend", "flashinfer", "--host", "127.0.0.1",
                "--port", str(self.port), "--mem-fraction-static", "0.669",
                "--chunked-prefill-size", "2048", "--disable-radix-cache",
                "--random-seed", str(self.seed), "--cuda-graph-max-bs-decode",
                str(self.max_decode_bs), "--decode-log-interval", "1000000"]
        if self.force_decode_eager:
            args.append("--disable-decode-cuda-graph")
        return args


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_server_info(config, info):
    expected = dict(model_path=config.model, revision=config.revision, dtype=config.dtype,
                    attention_backend="flashinfer", disable_radix_cache=True,
                    disable_overlap_schedule=False, disable_prefill_cuda_graph=False,
                    disable_decode_cuda_graph=config.force_decode_eager,
                    mem_fraction_static=.669, chunked_prefill_size=2048, random_seed=config.seed,
                    tp_size=1, dp_size=1, pp_size=1)
    wrong={key:dict(expected=value,actual=info.get(key)) for key,value in expected.items() if info.get(key)!=value}
    if wrong:
        raise ValueError(f"Resolved runtime config disagrees: {wrong}")
    return {**{key:info[key] for key in expected},
            **{key:info.get(key) for key in ("cuda_graph_max_bs_decode","cuda_graph_bs_decode","cuda_graph_config")}}


def distribution(values):
    values = sorted(values)
    if not values:
        raise ValueError("Empty measurements")
    def quantile(q):
        index = (len(values) - 1) * q
        lo = int(index)
        return values[lo] + (values[min(lo + 1, len(values)-1)]-values[lo])*(index-lo)
    mean = st.mean(values)
    sd = st.stdev(values) if len(values) > 1 else None
    return dict(count=len(values), mean=mean, median=st.median(values),
                p95=quantile(.95), sd=sd, cv=sd/mean if sd is not None and mean else None)


def normalize(total, *, generated_tokens, decode_iterations, requests):
    if any(type(n) is not int or n <= 0 for n in (generated_tokens, decode_iterations, requests)):
        raise ValueError("Normalization requires explicit positive work counts")
    if not math.isfinite(total) or total < 0:
        raise ValueError("Invalid total")
    return dict(total=total, per_generated_token=total/generated_tokens,
                per_decode_iteration=total/decode_iterations, per_request=total/requests)


def verify_cohort(config, observations):
    rows = observations["rows"]
    window = rows[9:80]
    if len(window) != 71:
        raise ValueError("Missing steady completion window")
    expected_graph = config.requested_mode == "graph_eligible"
    if any(r["batch"] != config.batch_size or r["graph"] != expected_graph for r in window):
        raise ValueError("Observed B/path disagrees with requested policy")
    if any(b["forward_iter"] != a["forward_iter"]+1 for a,b in zip(window,window[1:])):
        raise ValueError("Nonconsecutive completion iterations")
    intervals = [(b["ns"]-a["ns"])/1e6 for a,b in zip(window,window[1:])]
    if any(x <= 0 for x in intervals):
        raise ValueError("Nonpositive interval")
    buckets = set(observations["buckets"])
    if expected_graph:
        if len(buckets) != 1:
            raise ValueError("Missing or mixed actual replay buckets")
        bucket = next(iter(buckets))
        if not config.batch_size <= bucket <= config.max_decode_bs:
            raise ValueError("Invalid actual replay bucket")
        mode = "padded_cuda_graph_replay" if bucket > config.batch_size else "cuda_graph_replay"
    else:
        if buckets:
            raise ValueError("Unexpected graph replay in eager cohort")
        bucket, mode = None, "eager"
    return dict(valid=True, requested_mode=config.requested_mode, verified_mode=mode,
                actual_batch=config.batch_size, replay_bucket=bucket, decode_intervals=70,
                useful_decode_tokens=config.batch_size*70, interval_ms=distribution(intervals),
                intervals_ms=intervals, useful_tokens_s=config.batch_size*1000/st.mean(intervals),
                worker_pid=observations["pid"], window_forward_ids=[window[0]["forward_iter"],window[-1]["forward_iter"]])
