import unittest
import tempfile
import sqlite3
from pathlib import Path
from profiling.phase2d_metrics import union,clip,length,timeline,family,api_group
from profiling.run_phase2d import jobs
from profiling.core import distribution
from profiling.analyze_phase2d import aggregate,accepted


class IntervalTests(unittest.TestCase):
    def test_union_unsorted_nested_touching(self):
        self.assertEqual(union([(5,10),(0,6),(2,3),(10,12),(20,21)]),[(0,12),(20,21)])

    def test_overlap_not_double_counted(self):
        result=timeline([(0,10),(5,15)],(0,20))
        self.assertEqual(result["busy_ns"],15)
        self.assertEqual(result["overlap_excess_ns"],5)
        self.assertEqual(result["inactive_ns"],5)

    def test_gaps_separate_edges(self):
        result=timeline([(2,4),(7,9),(11,12)],(0,15))
        self.assertEqual(result["interior_gap_ns"],5)
        self.assertEqual(result["edge_gap_ns"],5)
        self.assertEqual(result["max_gap_ns"],3)
        self.assertAlmostEqual(result["interior_gap_fraction"],.5)

    def test_clipping(self):
        self.assertEqual(clip([(-2,2),(4,8),(10,13)],(0,10)),[(0,2),(4,8)])
        self.assertEqual(timeline([(-2,12)],(0,10))["busy_fraction"],1)
        with self.assertRaises(ValueError):clip([(0,1)],(1,1))

    def test_copy_overlap_union(self):
        kernels=[(0,5),(10,15)];copies=[(3,12)]
        self.assertEqual(length(union(kernels+copies)),15)
        self.assertEqual(timeline(kernels,(0,15))["interior_gap_ns"],5)
        self.assertEqual(timeline(kernels+copies,(0,15))["inactive_ns"],0)

    def test_empty(self):
        self.assertEqual(timeline([],(0,10))["inactive_ns"],10)
        self.assertIsNone(timeline([],(0,10))["interior_gap_fraction"])

    def test_multi_stream_serial_vs_overlap(self):
        self.assertEqual(timeline([(0,5),(5,10)],(0,10))["overlap_excess_ns"],0)
        self.assertEqual(timeline([(0,10),(0,10)],(0,10))["overlap_excess_ns"],10)

    def test_runtime_driver_launch_separation(self):
        self.assertEqual(api_group("cudaLaunchKernelExC"),"kernel_launch")
        self.assertEqual(api_group("cuLaunchKernel"),"other")
        self.assertEqual(api_group("cudaGraphLaunch"),"graph_launch")
        self.assertEqual(api_group("cudaStreamWaitEvent"),"other")

    def test_conservative_families(self):
        self.assertEqual(family("opaque_123"),"unknown")
        self.assertEqual(family("cutlass_fused_rmsnorm"),"normalization")
        self.assertEqual(family("BatchPrefillWithPagedKVCacheKernel"),"attention")
        self.assertEqual(family("sglang::store_kvcache"),"KV/cache")

    def test_cv_and_unit_denominator(self):
        self.assertEqual(distribution([8,10,12])["cv"],.2)
        self.assertAlmostEqual(560/70/8,1)

    def test_campaign_restarts_and_priority(self):
        initial=list(jobs("primary"))
        self.assertEqual(initial[0][1],"C")
        for arm in "ABCD":
            for profiled in (True,False):
                self.assertEqual(sum(a==arm and p==profiled for _,a,p,_ in initial),3)
        self.assertEqual(len(list(jobs("secondary"))),6)

    def test_trace_aggregation_missing_values_and_cv(self):
        rows=[dict(arm="C",metric=8,missing=None),dict(arm="C",metric=10,missing=None),dict(arm="C",metric=12,missing=None)]
        values=aggregate(rows,["arm"],["metric","missing"])
        self.assertEqual(len(values),1)
        self.assertEqual(values[0]["count"],3)
        self.assertEqual(values[0]["mean"],10)
        self.assertEqual(values[0]["cv"],.2)
        self.assertTrue(values[0]["warning_cv_gt_10pct"])

    def test_invalid_trace_filter_keeps_failures_out(self):
        rows=[dict(run_id="a",accepted=True),dict(run_id="b",accepted=False)]
        self.assertEqual([x["run_id"] for x in accepted(rows)],["a"])
        with self.assertRaises(ValueError):accepted(rows+rows)

    def test_structured_worker_region_and_normalization(self):
        from profiling.phase2d_metrics import extract
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"fixture.sqlite";db=sqlite3.connect(path)
            db.executescript('''CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,streamId INTEGER,correlationId INTEGER,demangledName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER);''')
            pid=42<<24;tid=pid+43
            db.executemany("INSERT INTO StringIds VALUES (?,?)",[(1,"cudaLaunchKernel"),(2,"cuLaunchKernel"),(3,"gemm_kernel")])
            db.executemany("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)",[
                (100,None,tid,"GPU_MECH_MEASUREMENT_SUBMIT_BEGIN:measure:B8"),
                (200,None,tid,"GPU_MECH_MEASUREMENT_SUBMIT_END:measure:B8"),
                (100,200,tid,"GPU_MECH_MEASUREMENT_COMPLETION:measure")])
            db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",[(110,120,tid,1,7),(111,119,tid,2,8)])
            db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",[
                (120,160,pid,1,7,3,1,1,1,32,1,1),(140,180,pid,2,8,3,1,1,1,32,1,1)])
            db.commit();db.close()
            m=dict(config=dict(batch_size=8),cohorts=[dict(warmup=False,worker_pid=42,cohort="measure",verified_mode="eager",replay_bucket=None)],
                   trace=dict(completion_window=dict(quality=dict(completeness_proven=False))))
            result=extract(path,m);metrics=result["metrics"]
            self.assertEqual(metrics["active_kernel_streams"],2)
            self.assertEqual(metrics["runtime_api_count"],1)
            self.assertEqual(metrics["driver_api_count"],1)
            self.assertAlmostEqual(metrics["gpu_busy_fraction"],.6)
            self.assertAlmostEqual(metrics["gpu_busy_us_per_token"],60/560/1000)
            self.assertEqual(result["useful_tokens"],560)
            self.assertEqual(metrics["kernel_count"],2)


if __name__=="__main__":unittest.main()
