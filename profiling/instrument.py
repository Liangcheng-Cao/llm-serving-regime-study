"""Minimal guarded observer injection; never modify installed SGLang."""
import hashlib
from pathlib import Path
import shutil
import tempfile

HASHES = {
    "srt/managers/scheduler_components/batch_result_processor.py": "0f0a3d92603c58bc555969b4f01daf2796f5aac9f38ad5a7155455ef5ca3ed4a",
    "srt/model_executor/runner/decode_cuda_graph_runner.py": "4f9b185ed058b8a6fb1e1d803ab17840a618af5d5816e534fbcdb6f67878072d",
    "srt/model_executor/model_runner.py": "b40efe0c7fb87d6f3baf5f9df3e755b9138247f82505183c564b4f8b1bc6663a",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError("Installed source anchor changed; re-audit before instrumenting")
    return text.replace(anchor, replacement, 1)


def prepare(original, run_id):
    for rel, expected in HASHES.items():
        if digest(original/rel) != expected:
            raise ValueError(f"Source hash changed: {rel}; re-audit required")
    # Source-only copy on Linux disk avoids slow WSL-mounted file operations.
    scratch = Path(tempfile.mkdtemp(prefix=f"gpu-mech-{run_id}-"))
    copied = scratch / "sglang"
    shutil.copytree(original, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(Path(__file__).with_name("observer.py"), copied / "_gpu_mech_observer.py")
    rel = "srt/managers/scheduler_components/batch_result_processor.py"
    path = copied / rel
    old = path.read_text()
    new = replace_once(old, "import logging\n", "import logging\nfrom sglang import _gpu_mech_observer as _gm\n")
    anchor = '''    def process_batch_result_decode(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()
'''
    new = replace_once(new, anchor, anchor + "        _gm.completion(batch, result)\n")
    tail = '''        self.metrics_reporter.report_decode_stats(
            can_run_cuda_graph,
            running_batch=batch,
            num_correct_drafts=result.num_correct_drafts,
        )
'''
    new = replace_once(new, tail, tail + "        _gm.flush(batch)\n")
    path.write_text(new)
    graph_rel = "srt/model_executor/runner/decode_cuda_graph_runner.py"
    graph_path = copied / graph_rel
    graph_old = graph_path.read_text()
    graph_new = replace_once(graph_old, "import logging\n", "import logging\nfrom sglang import _gpu_mech_observer as _gm\n")
    anchor = "            output = self.backend.replay(self._replay_graph_key, forward_batch)\n"
    graph_new = replace_once(graph_new, anchor, "            _gm.replay(self._replay_graph_key.size)\n" + anchor)
    graph_path.write_text(graph_new)
    runner_rel="srt/model_executor/model_runner.py"
    runner_path=copied/runner_rel
    runner=replace_once(runner_path.read_text(),"import logging\n","import logging\nfrom sglang import _gpu_mech_observer as _gm\n")
    anchor="    ) -> ModelRunnerOutput:\n        if has_forward_context():\n"
    runner=replace_once(runner,anchor,"    ) -> ModelRunnerOutput:\n        _gm.submission(forward_batch)\n        if has_forward_context():\n")
    runner_path.write_text(runner)
    sources = {r: dict(original_sha256=digest(original/r), instrumented_sha256=digest(copied/r))
               for r in (rel, graph_rel, runner_rel)}
    return scratch, sources
