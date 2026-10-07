"""Reanalyze existing Phase 2D SQLite files; no server launch or input mutation."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics as st

from .phase2e_metrics import read_events, analyze_events, sequence_similarity, stats
from .phase2d_metrics import family, LABELS
from .analyze_phase2d import write_csv, aggregate

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/gpu-mechanism-study/phase2d'
OUT=ROOT/'outputs/gpu-mechanism-study/phase2e'


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'summaries').mkdir(exist_ok=True)
    entries=[json.loads(x) for x in (SOURCE/'trace_manifest.jsonl').read_text().splitlines()]
    summaries=[];flat=[];kernels=[];syncs=[];sequences={};allgaps={};provenance=[]
    for entry in entries:
        if not entry['accepted'] or not entry['profiled']:continue
        rid=entry['run_id'];base={k:entry[k] for k in ('run_id','arm','block','output_len')}
        parsed=json.loads((SOURCE/entry['metrics']).read_text())
        database=SOURCE/'summaries'/rid/'trace.sqlite'
        events,apis,other=read_events(database,parsed)
        result,associations,gaps=analyze_events(events,apis,other,parsed['region_ns'],parsed['useful_tokens'])
        result.update(base);result['region_ns']=parsed['region_ns'];result['tokens']=parsed['useful_tokens']
        (OUT/'summaries'/f'{rid}.json').write_text(json.dumps(result,indent=2)+'\n')
        # Event-level associations are retained compressed locally; compact summaries are committed.
        import gzip,csv
        if associations:
            with gzip.open(OUT/'summaries'/f'{rid}-associations.csv.gz','wt',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(associations[0]));writer.writeheader();writer.writerows(associations)
        summaries.append(result);allgaps[rid]=gaps
        row={**base,**result['accounting'],**{'gap_'+k:v for k,v in result['gaps'].items()},
             **{'association_'+k:v for k,v in result['association'].items()},
             'selected_record_count':result['selected_record_count']}
        for kind,s in result['return_to_kernel_delay_us'].items():
            row.update({f'{kind}_delay_{k}':v for k,v in s.items()})
        for kind,count in result['correlation_counts'].items():row['correlation_'+kind]=count
        flat.append(row)
        kernels.extend(dict(**base,**k) for k in result['kernels'])
        syncs.extend(dict(**base,**s) for s in result['sync_events'])
        primary=max({k['streamId'] for k in events},key=lambda sid:sum(k['streamId']==sid for k in events))
        sequences[rid]={'all_names':[k['name'] for k in events],
                        'all_families':[family(k['name']) for k in events],
                        'dominant_stream_names':[k['name'] for k in events if k['streamId']==primary]}
        provenance.append(dict(**base,sqlite=str(database.relative_to(ROOT)),sqlite_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
            phase2d_metrics_sha256=hashlib.sha256((SOURCE/entry['metrics']).read_bytes()).hexdigest(),
            worker_pid=parsed['worker_pid'],region_ns=parsed['region_ns'],useful_tokens=parsed['useful_tokens']))
        print('Parsed',rid, result['correlation_counts'],flush=True)
    comparisons=[]
    for label,left,right,length in [('B8','C','D',128),('B9','B','A',128),('graph_padding','C','B',128),('B8_output256','C','D',256)]:
        ls=[s for s in summaries if s['arm']==left and s['output_len']==length]
        rs=[s for s in summaries if s['arm']==right and s['output_len']==length]
        for a in ls:
            for b in rs:
                for kind in sequences[a['run_id']]:
                    comparisons.append(dict(pair=label,reference=a['run_id'],comparison=b['run_id'],sequence=kind,
                        **sequence_similarity(sequences[a['run_id']][kind],sequences[b['run_id']][kind])))
    metrics=sorted(set(k for row in flat for k in row)-{'run_id','arm','block','output_len'})
    aggregates=aggregate(flat,['output_len','arm'],metrics)
    kagg=aggregate(kernels,['output_len','arm','name','family'],['observed_count','complete_count','mean_us','median_us','cumulative_us_per_token'])
    kernel_pairs=[]
    for pair,left,right in [('B8','C','D'),('B9','B','A'),('graph_padding','C','B')]:
        names={k['name'] for k in kernels if k['output_len']==128 and k['arm'] in (left,right)}
        for name in sorted(names):
            row=dict(pair=pair,name=name,family=family(name))
            for metric in ['observed_count','mean_us','median_us','cumulative_us_per_token']:
                values={a:[k[metric] for k in kernels if k['name']==name and k['arm']==a and k['output_len']==128] for a in (left,right)}
                for label,a in [('reference',left),('comparison',right)]:
                    s=stats(values[a]);row.update({f'{metric}_{label}_{key}':s[key] for key in ('count','mean','sd')})
                lm=st.mean(values[left]) if values[left] else None;rm=st.mean(values[right]) if values[right] else None
                row[metric+'_ratio']=rm/lm if lm and rm is not None else None
                row[metric+'_delta']=rm-lm if lm is not None and rm is not None else None
            kernel_pairs.append(row)
    import csv
    overhead=list(csv.DictReader((SOURCE/'profiler_perturbation.csv').open()))
    dtrace={r['run_id']:r for r in csv.DictReader((SOURCE/'trace_metrics.csv').open())}
    perturbation=[]
    for s in summaries:
        h=next((h for h in overhead if h['run_id']==s['run_id']),None)
        if h is None:continue
        d=dtrace[s['run_id']]
        perturbation.append(dict(run_id=s['run_id'],arm=s['arm'],output_len=s['output_len'],
            cadence_change_pct=float(h['cadence_change_pct']),selected_record_count=s['selected_record_count'],
            runtime_api_count=float(d['runtime_api_count']),kernel_launch_count=float(d['kernel_launch_count'])))
    primary=[p for p in perturbation if p['output_len']==128]
    correlations={metric:st.correlation([p[metric] for p in primary],[p['cadence_change_pct'] for p in primary])
                  for metric in ['selected_record_count','runtime_api_count','kernel_launch_count']}
    for filename,rows in [('trace_metrics.csv',flat),('aggregate_metrics.csv',aggregates),('kernel_metrics.csv',kernels),
                          ('kernel_aggregate.csv',kagg),('kernel_comparisons.csv',kernel_pairs),('sequence_similarity.csv',comparisons),
                          ('profiler_event_association.csv',perturbation),('sync_events.csv',syncs)]:write_csv(OUT/filename,rows)
    (OUT/'input_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    result=dict(traces=summaries,sequence_comparisons=comparisons,aggregate=aggregates,kernel_aggregate=kagg,
                kernel_comparisons=kernel_pairs,profiler_event_association=perturbation,profiler_pearson_primary=correlations)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    make_figures(summaries,allgaps)
    print('Completed',len(summaries),'traces',len(comparisons),'sequence comparisons',flush=True)


def make_figures(summaries,gaps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    import shutil
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for source,dest in [('01_unprofiled_decode','A_service_effect'),('04_launch_calls','C_launch_calls'),('05_gpu_busy_fraction','D_busy_fraction')]:
        for ext in ('png','svg'):shutil.copyfile(SOURCE/'figures'/f'{source}.{ext}',folder/f'{dest}.{ext}')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','svg'):
            p=folder/f'{name}.{ext}';fig.savefig(p,dpi=170)
            if ext=='svg':p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
        plt.close(fig)
    labels=['B8 graph','B8 eager','B9 graph','B9 eager'];order='CDBA'
    fig,ax=plt.subplots(figsize=(9,4.8))
    busy=[st.mean(s['accounting']['gpu_work_union_us_per_token'] for s in summaries if s['arm']==a and s['output_len']==128) for a in order]
    gap=[st.mean(s['accounting']['interior_gaps_us_per_token'] for s in summaries if s['arm']==a and s['output_len']==128) for a in order]
    x=np.arange(4);ax.bar(x-.18,busy,.36,label='Observed GPU-work union',color='#177e89');ax.bar(x+.18,gap,.36,label='Interior kernel gaps',color='#d96b4c')
    ax.set(xticks=x,xticklabels=labels,ylabel='µs / useful decode-token equivalent',title='B. Recorded work and gaps (not an additive latency model)')
    ax.legend();fig.text(.5,.01,'Profiled windows only; copies can overlap kernel gaps',ha='center',fontsize=9);fig.subplots_adjust(bottom=.18)
    save(fig,'B_work_and_gaps')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for a,color in [('C','#177e89'),('D','#d96b4c')]:
        first=True
        for s in summaries:
            if s['arm']!=a or s['output_len']!=128:continue
            values=np.sort(np.array(gaps[s['run_id']])/1000)
            axes[0].plot(values,np.arange(1,len(values)+1)/len(values),color=color,alpha=.65,label=LABELS[a] if first else None)
            axes[1].plot(values,np.cumsum(values)/sum(values),color=color,alpha=.65);first=False
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('Positive inter-kernel gap (µs)');ax.set_ylim(0,1.02)
    axes[0].set_ylabel('Fraction of gap count');axes[1].set_ylabel('Fraction of cumulative gap time');axes[0].legend()
    fig.suptitle('E. B8 gap distributions — three separate traces per arm');save(fig,'E_gap_ecdf')
    fig,ax=plt.subplots(figsize=(11,4.7));ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
    ax.text(6,4.65,'F. Conceptual execution model — not literal trace data',ha='center',fontsize=15)
    for y,label in [(3.7,'Eager host'),(2.8,'Eager GPU'),(1.6,'Graph host'),(.7,'Graph GPU')]:ax.text(0,y,label,va='center')
    for i in range(4):
        x=2+i*2.2
        ax.add_patch(Rectangle((x,3.45),.8,.5,color='#d96b4c'));ax.text(x+.4,3.7,'launch',ha='center',va='center',fontsize=9)
        ax.add_patch(Rectangle((x+.3,2.55),.9,.5,color='#177e89'));ax.text(x+.75,2.8,'kernel',ha='center',va='center',fontsize=9,color='white')
        if i<3:ax.text(x+1.65,2.8,'gap',ha='center',va='center',fontsize=9)
    ax.add_patch(Rectangle((2,1.35),1.7,.5,color='#d96b4c'));ax.text(2.85,1.6,'replay submission',ha='center',va='center',fontsize=9)
    for i in range(4):
        x=2.3+i*1.0;ax.add_patch(Rectangle((x,.45),.9,.5,color='#177e89'));ax.text(x+.45,.7,'kernel',ha='center',va='center',fontsize=9,color='white')
    ax.text(6,0,'Host dispatch, dependencies and bookkeeping may contribute; no one-call/one-gap causal mapping implied.',ha='center',fontsize=9)
    save(fig,'F_conceptual_timeline')


if __name__=='__main__':main()
