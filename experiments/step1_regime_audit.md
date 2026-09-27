# Decode execution boundary

**Question.** Does crossing the default graph ceiling actually select another execution path?

**Method.** Audit installed SGLang 0.5.19 control flow and verify native runtime flags using
one cohort at each B in {1,4,8,9,16}, 128 input tokens and 48 fixed output tokens.
Decode graph sizes were [1,2,4,8], padding enabled. This software default follows a GPU-memory
heuristic (less than 20 GiB selects a ceiling of 8); it is not a hardware capacity bound.

The inspected paths were `srt/arg_groups/memory_hook.py`, `cuda_graph_hook.py`,
`model_executor/model_runner.py`, `runner/decode_cuda_graph_runner.py`,
`runner_backend/full_cuda_graph_backend.py`, and `eager_runner.py` within the installed package.
The decode eligibility check permits padded batch sizes up to the captured maximum;
the selected backend invokes graph replay, otherwise ordinary model forward executes.
The distinct prefill graph branch does not rescue out-of-coverage decode batches.

**Result.** B=1,4,8 reported graph execution; B=9,16 reported eager execution,
with 48 native rows per cohort. These rows are not 48 independent repetitions.
Actual useful batch size and the returned graph flag were checked together.
The audit did not measure service rates.

**Verdict: PASS.** The installed runtime exposes an observable decode execution boundary.
Installed source hashes were verified unchanged in the original audit; installed sources and
machine-specific manifests are not redistributed in this package.
