"""Read-only event analysis of the schema and regions inspected in Phase 2D.

Correlations are process-scoped. Ambiguous matches are excluded; graph replay
correlations are explicitly many-to-one and never treated as individual launches.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import sqlite3
import statistics as st

from .core import distribution
from .phase2d_metrics import clip, union, length, family, api_group


def stats(values):
    values = sorted(values)
    if not values:
        return dict(count=0, mean=None, median=None, sd=None, cv=None, p90=None, p95=None, p99=None, maximum=None)
    result = distribution(values)
    def q(p):
        i = (len(values)-1)*p; lo = int(i); hi = min(lo+1,len(values)-1)
        return values[lo]+(i-lo)*(values[hi]-values[lo])
    result.update(p90=q(.90), p99=q(.99), maximum=values[-1])
    return result


def gap_distribution(gaps_ns):
    values = [v/1000 for v in gaps_ns if v>0]
    result = stats(values)
    total = sum(values)
    result['total_us'] = total
    for threshold in (5,10,20,50):
        result[f'time_fraction_gt_{threshold}us'] = sum(v for v in values if v>threshold)/total if total else None
        result[f'count_gt_{threshold}us'] = sum(v>threshold for v in values)
    return result


def accounting(kernels, other_work, region, tokens):
    if tokens<=0: raise ValueError('Positive token denominator required')
    km = union(clip(kernels,region)); work = union(clip(kernels+other_work,region))
    k = length(km); w = length(work); marked = region[1]-region[0]
    span = km[-1][1]-km[0][0] if km else 0
    gaps = sum(b[0]-a[1] for a,b in zip(km,km[1:]))
    edges = marked-k-gaps; nonkernel = w-k
    scale = tokens*1000
    return {key:value/scale for key,value in dict(
        marked_span_us_per_token=marked, kernel_span_us_per_token=span,
        kernel_union_us_per_token=k, gpu_work_union_us_per_token=w,
        interior_gaps_us_per_token=gaps, edge_gaps_us_per_token=edges,
        nonkernel_outside_kernels_us_per_token=nonkernel,
        naive_residual_us_per_token=marked-w-gaps,
        corrected_residual_us_per_token=marked-(w+gaps+edges-nonkernel),
        kernel_span_residual_us_per_token=span-k-gaps).items()}


def sequence_similarity(a,b):
    """Exact LCS length with a bitset DP; no heuristic removal of repeated names.

    Similarity is 2*LCS/(len(a)+len(b)). Start-time order across streams is a
    descriptive linearization, not a dependency order.
    """
    masks={}
    for i,item in enumerate(b): masks[item]=masks.get(item,0)|(1<<i)
    state=0
    for item in a:
        x=masks.get(item,0)|state
        state=x & ~(x-((state<<1)|1))
    lcs=state.bit_count()
    return dict(n_reference=len(a),n_comparison=len(b),lcs=lcs,
                similarity=2*lcs/(len(a)+len(b)) if a or b else 1.0,
                unmatched_reference=len(a)-lcs,unmatched_comparison=len(b)-lcs)


def correlate(kernels,apis):
    """A unique same-process/correlationId launch, preferring runtime over driver.

    Caller supplies one worker's records. No nearest-time guess and no fabricated
    cross-layer mapping. Duplicate launch records within a tier are ambiguous.
    """
    lookup=defaultdict(list)
    for a in apis: lookup[a['correlationId']].append(a)
    results=[]
    for k in kernels:
        candidates=lookup[k['correlationId']]
        tiers=[('runtime_individual',[a for a in candidates if api_group(a['name'])=='kernel_launch']),
               ('driver_individual',[a for a in candidates if a['name'].startswith(('cuLaunchKernel','cuLaunchCooperativeKernel'))]),
               ('graph_replay',[a for a in candidates if api_group(a['name'])=='graph_launch'])]
        chosen=None; kind='unmatched'
        for label,rows in tiers:
            if rows:
                kind=label if len(rows)==1 else 'ambiguous'
                chosen=rows[0] if len(rows)==1 else None
                break
        if chosen and chosen['start']>k['start']:
            kind='invalid_order';chosen=None
        results.append((kind,chosen))
    return results


def gap_parts(gap,api):
    """Partition a gap relative to a correlated launch's entry and return.

    These timestamps give association, not attribution of CPU work or causality.
    Negative return-to-kernel delays are retained elsewhere.
    """
    lo,hi=gap
    before=max(0,min(hi,api['start'])-lo)
    during=max(0,min(hi,api['end'])-max(lo,api['start']))
    after=max(0,hi-max(lo,api['end']))
    return before,during,after


def overlap_with(merged,starts,lo,hi):
    i=max(0,bisect_right(starts,lo)-1);total=0
    while i<len(merged) and merged[i][0]<hi:
        a,b=merged[i];total+=max(0,min(b,hi)-max(a,lo));i+=1
    return total


def kernel_summary(kernels,region,tokens):
    groups=defaultdict(list)
    for k in kernels: groups[k['name']].append(k)
    result=[]
    for name,rows in groups.items():
        durations=[(r['end']-r['start'])/1000 for r in rows if region[0]<=r['start'] and r['end']<=region[1]]
        s=stats(durations)
        result.append(dict(name=name,family=family(name),observed_count=len(rows),
                           complete_count=len(durations),mean_us=s['mean'],median_us=s['median'],
                           cumulative_us_per_token=length(clip([(r['start'],r['end']) for r in rows],region))/tokens/1000))
    return result


def read_events(database,parsed):
    db=sqlite3.connect(f'file:{database}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    worker=parsed['worker_pid'];lo,hi=parsed['region_ns']
    try:
        strings={r['id']:r['value'] for r in db.execute('SELECT * FROM StringIds')}
        def read(table,pidkey):
            return [dict(r) for r in db.execute('SELECT * FROM '+table) if ((r[pidkey]>>24)&0xffffff)==worker]
        api=read('CUPTI_ACTIVITY_KIND_RUNTIME','globalTid')
        for a in api:a['name']=strings[a['nameId']]
        kernels=read('CUPTI_ACTIVITY_KIND_KERNEL','globalPid')
        kernels=[k for k in kernels if k['start']<hi and k['end']>lo]
        for k in kernels:k['name']=strings[k['demangledName']]
        kernels.sort(key=lambda k:(k['start'],k['end'],k['streamId'],k['name']))
        other=read('CUPTI_ACTIVITY_KIND_MEMCPY','globalPid')+read('CUPTI_ACTIVITY_KIND_MEMSET','globalPid')
        other=[k for k in other if k['start']<hi and k['end']>lo]
        return kernels,api,other
    finally:db.close()


def analyze_events(kernels,apis,other,region,tokens):
    lo,hi=region
    intervals=[(k['start'],k['end']) for k in kernels]
    km=union(clip(intervals,region));gaps=[(a[1],b[0]) for a,b in zip(km,km[1:])]
    work=union(clip(intervals+[(x['start'],x['end']) for x in other],region))
    starts=[a for a,b in work]
    corr=correlate(kernels,apis)
    match_counts=Counter(x[0] for x in corr)
    multiplicities=defaultdict(Counter)
    for kind,a in corr:
        if a:multiplicities[kind][(a['correlationId'],a['globalTid'],a['start'])]+=1
    associations=[];delay=defaultdict(list)
    # A unique first kernel after each union gap; ties are ambiguous, excluded.
    at_start=defaultdict(list)
    for i,k in enumerate(kernels):
        at_start[k['start']].append(i)
        kind,a=corr[i]
        if a:
            delay[kind].append((k['start']-a['end'])/1000)
    for a,b in gaps:
        ids=at_start[b]
        if len(ids)!=1:continue
        i=ids[0];kind,api=corr[i]
        if kind not in ('runtime_individual','driver_individual'):continue
        before,during,after=gap_parts((a,b),api)
        associations.append(dict(gap_start_ns=a,gap_end_ns=b,kernel_name=kernels[i]['name'],
            stream_id=kernels[i]['streamId'],correlation_id=kernels[i]['correlationId'],
            match_kind=kind,api_name=api['name'],api_start_ns=api['start'],api_end_ns=api['end'],
            gap_us=(b-a)/1000,before_launch_us=before/1000,during_launch_us=during/1000,
            after_launch_us=after/1000,return_to_kernel_us=(b-api['end'])/1000,
            launch_start_in_gap=a<=api['start']<b))
    sync=[]
    for api in apis:
        if not(lo<=api['start']<hi and api_group(api['name'])=='sync'):continue
        end=min(hi,api['end']);duration=end-api['start']
        overlap=overlap_with(work,starts,api['start'],end)
        sync.append(dict(name=api['name'],start_ns=api['start'],end_ns=api['end'],
                         position_fraction=(api['start']-lo)/(hi-lo),duration_us=duration/1000,
                         overlap_gpu_work_us=overlap/1000,overlap_fraction=overlap/duration if duration else None,
                         boundary_1pct=api['start']<lo+.01*(hi-lo) or api['start']>=hi-.01*(hi-lo)))
    total_gap=sum(b-a for a,b in gaps)/1000
    assoc_gap=sum(x['gap_us'] for x in associations)
    association=dict(gap_count=len(gaps),matched_gap_count=len(associations),
        matched_gap_count_fraction=len(associations)/len(gaps) if gaps else None,
        matched_gap_time_fraction=assoc_gap/total_gap if total_gap else None,
        launch_start_in_gap_fraction=st.mean(x['launch_start_in_gap'] for x in associations) if associations else None,
        **{f'{part}_fraction_of_matched_gap':sum(x[part+'_us'] for x in associations)/assoc_gap if assoc_gap else None
           for part in ('before_launch','during_launch','after_launch')})
    selected_api=[a for a in apis if lo<=a['start']<hi]
    return dict(accounting=accounting(intervals,[(x['start'],x['end']) for x in other],region,tokens),
        gaps=gap_distribution([b-a for a,b in gaps]),correlation_counts=dict(match_counts),
        correlation_multiplicity={kind:dict(unique_apis=len(counts),maximum_kernels_per_api=max(counts.values()),
            apis_with_multiple_kernels=sum(v>1 for v in counts.values())) for kind,counts in multiplicities.items()},
        return_to_kernel_delay_us={kind:stats(v) for kind,v in delay.items()},
        negative_return_delay_counts={kind:sum(x<0 for x in v) for kind,v in delay.items()},
        association=association,sync_events=sync,kernels=kernel_summary(kernels,region,tokens),
        selected_record_count=len(kernels)+len(other)+len(selected_api),
        selected_api_records=len(selected_api),selected_kernel_records=len(kernels)),associations,[b-a for a,b in gaps]
