import copy
from dataclasses import asdict, replace
import importlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from profiling.core import ARMS, Config, new_run_id, save, normalize, distribution, verify_cohort
from profiling.instrument import replace_once
from profiling.parsers.nsys import parse_summary, inspect_sqlite
from profiling.run_decode_experiment import profile_command


def observation(graph=False,batch=9,bucket=None):
    return dict(pid=42,buckets=[] if bucket is None else [bucket],rows=[
        dict(ns=i*1000000,forward_iter=i+100,batch=batch,graph=graph) for i in range(128)])


class ConfigurationTests(unittest.TestCase):
    def test_arms_and_phase_specific_control(self):
        self.assertEqual([Config(batch_size=b,max_decode_bs=c,force_decode_eager=f).requested_mode
                          for b,c,f in ARMS.values()],
                         ["natural_eager","graph_eligible","graph_eligible","forced_decode_eager"])
        cmd=Config(force_decode_eager=True).command("python")
        self.assertIn("--disable-decode-cuda-graph",cmd)
        self.assertNotIn("--disable-cuda-graph",cmd)
        self.assertIn("--revision",cmd)

    def test_invalid_config(self):
        for change in (dict(batch_size=0),dict(output_len=80),dict(warmup=-1),dict(port=20),
                       dict(run_id="../escape"),dict(graph_policy="automatic"),dict(profiling=True,repeats=2)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                replace(Config(run_id="test"),**change).validate()

    def test_unique_safe_ids(self):
        ids={new_run_id() for _ in range(200)}
        self.assertEqual(len(ids),200)
        for rid in ids:
            Config(run_id=rid).validate()

    def test_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"manifest.json"
            value=dict(valid=False,config=asdict(Config(run_id="test")),failures=["unverified"])
            save(path,value)
            self.assertEqual(json.loads(path.read_text()),value)
            with self.assertRaises(ValueError):
                save(path,dict(value=float("nan")))

    def test_profiler_targets_server(self):
        cmd=profile_command(["python","-m","sglang.launch_server"],Path("trace"))
        self.assertIn("--capture-range=cudaProfilerApi",cmd)
        self.assertEqual(cmd[-3:],["python","-m","sglang.launch_server"])


class VerificationTests(unittest.TestCase):
    def test_eager_and_padded_graph(self):
        eager=verify_cohort(Config(),observation())
        self.assertEqual(eager["decode_intervals"],70)
        self.assertEqual(eager["useful_decode_tokens"],630)
        self.assertEqual(eager["interval_ms"]["mean"],1)
        padded=verify_cohort(Config(max_decode_bs=16),observation(True,9,12))
        self.assertEqual(padded["verified_mode"],"padded_cuda_graph_replay")
        exact=verify_cohort(Config(batch_size=8),observation(True,8,8))
        self.assertEqual(exact["verified_mode"],"cuda_graph_replay")

    def test_reject_invalid_path_batch_window_bucket(self):
        cases=[]
        x=observation(); x["rows"][20]["graph"]=True; cases.append(x)
        x=observation(); x["rows"][20]["batch"]=8; cases.append(x)
        x=observation(); x["rows"][20]["forward_iter"]+=1; cases.append(x)
        x=observation(); x["rows"]=x["rows"][:79]; cases.append(x)
        x=observation(); x["buckets"]=[12]; cases.append(x)
        for x in cases:
            with self.assertRaises(ValueError):
                verify_cohort(Config(),x)
        with self.assertRaises(ValueError):
            verify_cohort(Config(max_decode_bs=16),observation(True))

    def test_normalization_uses_useful_tokens(self):
        x=normalize(1260,generated_tokens=630,decode_iterations=70,requests=9)
        self.assertEqual(x["per_generated_token"],2)
        self.assertEqual(x["per_decode_iteration"],18)
        self.assertEqual(x["per_request"],140)
        with self.assertRaises(ValueError):
            normalize(1,generated_tokens=0,decode_iterations=70,requests=9)

    def test_statistics(self):
        x=distribution([1,2,3])
        self.assertEqual(x["mean"],2)
        self.assertEqual(x["sd"],1)
        self.assertIsNone(distribution([1])["sd"])
        with self.assertRaises(ValueError):
            distribution([])

    def test_source_guard(self):
        for text in ("missing","anchor anchor"):
            with self.assertRaises(ValueError):
                replace_once(text,"anchor","replacement")

    def test_observer_tail_does_not_poison_next_cohort(self):
        import profiling.observer as observer
        observer=importlib.reload(observer)
        with tempfile.TemporaryDirectory() as folder:
            control=Path(folder)/"control.json"; out=Path(folder)/"obs.jsonl"
            with patch.dict(os.environ,{"GPU_MECH_CONTROL":str(control),"GPU_MECH_OBSERVATIONS":str(out)}):
                batch=SimpleNamespace(forward_iter=1,reqs=[SimpleNamespace(finished=lambda:True)])
                result=SimpleNamespace(can_run_cuda_graph=False)
                for cid in ["first","second"]:
                    save(control,dict(cohort=cid,warmup=False,nvtx=False))
                    observer.completion(batch,result); observer.flush(batch)
                    observer.completion(batch,result); observer.flush(batch)  # in-flight terminal row
                self.assertEqual([json.loads(l)["cohort"] for l in out.read_text().splitlines()],["first","second"])


class ParserTests(unittest.TestCase):
    def test_api_csv_preamble_quotes_and_sync(self):
        text='Processing...\n"Time (%)","Total Time (ns)","Num Calls","Name"\n50,100,2,cudaGraphLaunch\n50,100,1,cudaEventSynchronize\n'
        x=parse_summary(text,"api")
        self.assertEqual((x["count"],x["sync_count"],x["graph_launch_count"]),(3,1,2))
        self.assertEqual(x["total_ns"],200)

    def test_kernel_csv_with_comma(self):
        x=parse_summary('Total Time (ns),Instances,Name\n120,3,"kernel<float, int>"\n',"kernel")
        self.assertEqual(x["dominant"][0]["name"],"kernel<float, int>")

    def test_bad_schema_not_zero(self):
        for text in ("", "Time,Name\n10,kernel", "Total Time (ns),Instances,Name\n"):
            with self.assertRaises(ValueError):
                parse_summary(text,"kernel")

    def test_sqlite_scope_and_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"trace.sqlite"
            db=sqlite3.connect(path)
            pid=42<<24; tid=pid+43
            db.executescript('''CREATE TABLE StringIds(id INTEGER,value TEXT);
              CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
              CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,streamId INTEGER,demangledName INTEGER);
              CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER);''')
            db.executemany("INSERT INTO StringIds VALUES (?,?)",[(1,"kernel"),(2,"cudaGraphLaunch")])
            db.execute("INSERT INTO NVTX_EVENTS VALUES (100,200,?,NULL,?)",("GPU_MECH_MEASUREMENT_COMPLETION:test",tid))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (120,150,?,7,1)",(pid,))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (105,110,?,2)",(tid,))
            db.commit(); db.close()
            x=inspect_sqlite(path,42,True)
            self.assertEqual((x["kernel_count"],x["kernel_total_ns"],x["graph_launch_count"]),(1,30,1))
            with self.assertRaises(ValueError):
                inspect_sqlite(path,41,True)
            with self.assertRaises(ValueError):
                inspect_sqlite(path,42,False)


if __name__=="__main__":
    unittest.main()
