import unittest
from profiling.phase2d_metrics import union,clip,length,timeline,family,api_group
from profiling.run_phase2d import jobs
from profiling.core import distribution


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


if __name__=="__main__":unittest.main()
