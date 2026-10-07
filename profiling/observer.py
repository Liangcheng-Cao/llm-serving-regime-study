"""Imported only by an isolated, hash-guarded SGLang copy.

Two NVTX calls per cohort window; no added CUDA synchronization. Completion
timestamps match the prior observer placement. I/O occurs only at cohort boundaries.
"""
import json
import os
from pathlib import Path
import time

_control = None
_rows = []
_buckets = set()
_range_open = False
_finished_cohort = None


def begin():
    global _control, _rows, _buckets, _range_open
    _control = json.loads(Path(os.environ["GPU_MECH_CONTROL"]).read_text())
    _rows, _buckets, _range_open = [], set(), False


def replay(bucket):
    _buckets.add(int(bucket))


def completion(batch, result):
    global _range_open, _control
    if _control is None:
        begin()
    if _control["cohort"] == _finished_cohort:
        _control = None
        return
    _rows.append(dict(ns=time.perf_counter_ns(), forward_iter=int(batch.forward_iter),
                      batch=len(batch.reqs), graph=bool(result.can_run_cuda_graph)))
    if _control["nvtx"]:
        import torch
        if len(_rows) == 10:
            phase = "WARMUP" if _control["warmup"] else "MEASUREMENT"
            torch.cuda.nvtx.range_push(f"GPU_MECH_{phase}_COMPLETION:{_control['cohort']}")
            _range_open = True
        elif len(_rows) == 80 and _range_open:
            torch.cuda.nvtx.range_pop()
            _range_open = False


def flush(batch):
    global _control, _range_open, _finished_cohort
    if _control is None or not batch.reqs or not all(r.finished() for r in batch.reqs):
        return
    if _range_open:
        import torch
        torch.cuda.nvtx.range_pop()
        _range_open = False
    payload = dict(cohort=_control["cohort"], warmup=_control["warmup"], rows=_rows,
                   buckets=sorted(_buckets), pid=os.getpid())
    with open(os.environ["GPU_MECH_OBSERVATIONS"], "a") as stream:
        stream.write(json.dumps(payload) + "\n")
    _finished_cohort = _control["cohort"]
    _control = None
