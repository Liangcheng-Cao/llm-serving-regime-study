import unittest
from profiling.phase2e_metrics import accounting,gap_distribution,sequence_similarity,correlate,gap_parts,kernel_summary,overlap_with,analyze_events,read_events


class MechanismTests(unittest.TestCase):
    def test_accounting_copy_in_gap_and_edges(self):
        a=accounting([(2,4),(8,10)],[(3,9)],(0,12),1)
        self.assertAlmostEqual(a['naive_residual_us_per_token'],0)
        self.assertEqual(a['corrected_residual_us_per_token'],0)
        self.assertEqual(a['kernel_span_residual_us_per_token'],0)
        self.assertAlmostEqual(a['nonkernel_outside_kernels_us_per_token'],.004)

    def test_accounting_negative_naive_residual(self):
        a=accounting([(0,2),(8,10)],[(1,9)],(0,10),2)
        self.assertAlmostEqual(a['naive_residual_us_per_token'],-.003)
        self.assertEqual(a['corrected_residual_us_per_token'],0)

    def test_accounting_no_kernels(self):
        a=accounting([],[(0,10)],(0,10),1)
        self.assertEqual(a['corrected_residual_us_per_token'],0)
        with self.assertRaises(ValueError):accounting([],[],(0,10),0)

    def test_gap_threshold_strict_and_weighted(self):
        s=gap_distribution([5000,10000,20000,65000])
        self.assertEqual(s['count_gt_5us'],3)
        self.assertAlmostEqual(s['time_fraction_gt_20us'],.65)
        self.assertEqual(s['median'],15)
        self.assertAlmostEqual(s['p90'],51.5)

    def test_empty_gaps(self):
        self.assertIsNone(gap_distribution([])['time_fraction_gt_5us'])

    def test_sequence_repetition_insertion_shift(self):
        a=['A','B']*100
        self.assertEqual(sequence_similarity(a,a)['similarity'],1)
        self.assertEqual(sequence_similarity(a,['X']+a)['lcs'],200)
        self.assertEqual(sequence_similarity(a,a[1:]+['A'])['lcs'],199)
        self.assertEqual(sequence_similarity([],[])['similarity'],1)
        self.assertEqual(sequence_similarity(['A'],['B'])['similarity'],0)

    def test_sequence_known_lcs(self):
        self.assertEqual(sequence_similarity('ABCBDAB','BDCABA')['lcs'],4)

    def test_unique_correlation_runtime_preferred(self):
        k=[dict(correlationId=1,start=15)]
        apis=[dict(correlationId=1,start=5,end=20,name='cudaLaunchKernel'),dict(correlationId=1,start=6,end=8,name='cuLaunchKernel')]
        kind,a=correlate(k,apis)[0]
        self.assertEqual(kind,'runtime_individual')
        self.assertEqual(15-a['end'],-5) # GPU can begin before API returns.

    def test_ambiguous_missing_invalid(self):
        a=dict(correlationId=1,start=5,end=8,name='cudaLaunchKernel')
        self.assertEqual(correlate([dict(correlationId=1,start=10)],[a,a])[0][0],'ambiguous')
        self.assertEqual(correlate([dict(correlationId=2,start=10)],[a])[0][0],'unmatched')
        self.assertEqual(correlate([dict(correlationId=1,start=1)],[a])[0][0],'invalid_order')

    def test_graph_many_to_one_not_individual(self):
        a=dict(correlationId=1,start=0,end=4,name='cudaGraphLaunch')
        self.assertEqual([r[0] for r in correlate([dict(correlationId=1,start=x) for x in (5,10)],[a])],['graph_replay']*2)

    def test_gap_partition_api_straddles_boundaries(self):
        for a,b,expected in [(2,5,(0,0,10)),(12,16,(2,4,4)),(5,15,(0,5,5)),(15,25,(5,5,0)),(25,30,(10,0,0))]:
            self.assertEqual(gap_parts((10,20),dict(start=a,end=b)),expected)

    def test_kernel_duration_excludes_boundary_crossing(self):
        k=[dict(name='gemm',start=-2,end=2),dict(name='gemm',start=3,end=7)]
        r=kernel_summary(k,(0,10),2)[0]
        self.assertEqual(r['observed_count'],2);self.assertEqual(r['complete_count'],1)
        self.assertEqual(r['mean_us'],.004);self.assertEqual(r['cumulative_us_per_token'],.003)

    def test_overlap_union_window(self):
        self.assertEqual(overlap_with([(0,5),(8,10)],[0,8],3,9),3)

    def test_end_to_end_event_association(self):
        ks=[dict(name='gemm',start=0,end=1000,streamId=1,correlationId=1),
            dict(name='gemm',start=5000,end=6000,streamId=1,correlationId=2)]
        apis=[dict(name='cudaLaunchKernel',start=-1000,end=-500,correlationId=1,globalTid=1),
              dict(name='cuLaunchKernel',start=2000,end=3000,correlationId=2,globalTid=1),
              dict(name='cudaEventSynchronize',start=5000,end=6000,correlationId=3,globalTid=1)]
        result,links,gaps=analyze_events(ks,apis,[],(0,7000),1)
        self.assertEqual(gaps,[4000]);self.assertEqual(links[0]['return_to_kernel_us'],2)
        self.assertEqual(result['association']['matched_gap_time_fraction'],1)
        self.assertEqual(result['correlation_multiplicity']['driver_individual']['maximum_kernels_per_api'],1)
        self.assertEqual(result['sync_events'][0]['overlap_fraction'],1)

    def test_lcs_matches_reference_dynamic_programming(self):
        import itertools
        sequences=[p for n in range(4) for p in itertools.product('AB',repeat=n)]
        for a in sequences:
            for b in sequences:
                row=[0]*(len(b)+1)
                for x in a:
                    new=[0]
                    for j,y in enumerate(b):new.append(row[j]+1 if x==y else max(row[j+1],new[-1]))
                    row=new
                self.assertEqual(sequence_similarity(a,b)['lcs'],row[-1])

    def test_sqlite_worker_scope_and_names(self):
        import sqlite3,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.sqlite';db=sqlite3.connect(path)
            db.executescript('''CREATE TABLE StringIds(id,value);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,nameId,correlationId);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,globalPid,demangledName,streamId,correlationId);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start,end,globalPid);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start,end,globalPid);''')
            db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'cudaLaunchKernel'),(2,'gemm')])
            # Reused correlationId in another process must not create ambiguity.
            for pid in [4,5]:
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0,1,?,1,8)',((pid<<24)|pid,))
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2,3,?,2,1,8)',(pid<<24,))
            db.commit();db.close()
            kernels,apis,other=read_events(path,dict(worker_pid=4,region_ns=[1,4]))
            self.assertEqual(len(kernels),1);self.assertEqual(len(apis),1)
            self.assertEqual(correlate(kernels,apis)[0][0],'runtime_individual')


if __name__=='__main__':unittest.main()
