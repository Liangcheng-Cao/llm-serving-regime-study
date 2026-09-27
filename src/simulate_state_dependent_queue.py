"""Equal-share state-dependent service queue. No history in the state or service law."""
import csv, json, math
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]
curve = [{k: float(v) for k, v in x.items() if k in ['B', 'mu_tok_s', 'T_ms']} for x in csv.DictReader((R / 'data/processed/service_curve.csv').open())]
xs = np.array([x['B'] for x in curve])
ys = np.array([x['mu_tok_s'] for x in curve])
t8 = next((x['T_ms'] / 1000 for x in curve if x['B'] == 8))

def rate(n, law):
    """Aggregate useful tokens/s. Beyond B=16, np.interp holds aggregate capacity fixed; the removed law extrapolates B=8 per-job speed. No history labels enter this function."""
    if n == 0:
        return 0.0
    if law == 'removed' and n > 8:
        return n / t8
    return float(np.interp(n, xs, ys))

def simulate(initial, law):
    """Advance to arrivals, completions or the 60-second horizon, then drain. Initial jobs count toward departures but have no future-job sojourn. At ties, completions precede arrivals."""
    jobs = [dict(w=float(w), arrival=None, id=f'i{i}') for i, w in enumerate(initial)]
    t = 0.0
    next_arrival = 0
    departures = []
    segments = []
    events = [dict(t=0.0, N=len(jobs), kind='initial')]
    n0 = len(jobs)
    drain = False

    def complete():
        nonlocal jobs
        remaining = []
        for j in jobs:
            if j['w'] <= 1e-07:
                departures.append(dict(t=t, arrival=j['arrival'], id=j['id']))
            else:
                remaining.append(j)
        jobs = remaining
    complete()
    for tick in range(20000):
        while next_arrival < 600 and abs(t - next_arrival / 10) < 1e-08:
            jobs.append(dict(w=128.0, arrival=next_arrival / 10, id=f'f{next_arrival}'))
            next_arrival += 1
            events.append(dict(t=t, N=len(jobs), kind='arrival'))
        if t >= 60 and (not drain):
            drain = True
            n_end = len(jobs)
        if drain and (not jobs):
            break
        n = len(jobs)
        mu = rate(n, law)
        per_job = mu / n if n else 0
        arr = next_arrival / 10 if next_arrival < 600 else math.inf
        finish = t + min((j['w'] for j in jobs)) / per_job if n else math.inf
        boundary = 60.0 if t < 60 - 1e-10 else math.inf
        end = min(arr, finish, boundary)
        assert math.isfinite(end) and end > t - 1e-10
        dt = max(0.0, end - t)
        if t < 60 and dt > 0:
            segments.append(dict(start=t, end=end, N=n, mu_tok_s=mu))
        for j in jobs:
            j['w'] -= per_job * dt
        t = end
        before = len(jobs)
        complete()
        if before != len(jobs):
            events.append(dict(t=t, N=len(jobs), kind='completion'))
        if t > 90:
            raise RuntimeError('Unexpected drain length')
    else:
        raise RuntimeError('Event loop limit')
    dt = np.array([q['end'] - q['start'] for q in segments])
    ns = np.array([q['N'] for q in segments])
    assert abs(sum(dt) - 60) < 1e-07
    order = np.argsort(ns)
    median = int(ns[order][np.searchsorted(np.cumsum(dt[order]), 30)])
    c = sum((q['t'] <= 60 + 1e-09 for q in departures))
    assert n_end - n0 == 600 - c
    sojourn = [q['t'] - q['arrival'] for q in departures if q['arrival'] is not None]
    assert len(sojourn) == 600
    first_low = next((q['start'] for q in segments if q['N'] <= 8), None)
    first_high = next((q['start'] for q in segments if q['N'] > 8), None)
    return dict(law=law, N_initial=n0, initial_remaining=initial, mean_N=float(np.dot(ns, dt) / 60), median_N=median, N_end=n_end, completions=c, arrivals=600, net_drift=(n_end - n0) / 60, sojourn_mean_s=float(np.mean(sojourn)), sojourn_median_s=float(np.median(sojourn)), fraction_gt8=float(dt[ns > 8].sum() / 60), time_gt8_s=float(dt[ns > 8].sum()), time_outside_measured_B_s=float(dt[ns > 16].sum()), max_N=int(max(ns)), first_time_le8=first_low, first_time_gt8=first_high, segments=segments, events=events)

def main():
    state = json.loads((R / 'data/inputs/initial_states.json').read_text())
    assert state['all_lifecycle_checks_pass']
    outputs = []
    for x in state['states']:
        n = x['N_client']
        cases = dict(uniform=[128 * (j + 0.5) / n for j in range(n)], empirical=[j['remaining_tokens'] for j in x['jobs']], full128=[128] * n)
        for treatment, w in cases.items():
            for law in ['measured', 'removed']:
                out = simulate(w, law)
                out.update(rep=x['rep'], origin_label=x['history'], initialization=treatment)
                outputs.append(out)
                if x['rep'] == 1:
                    print({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items() if k in ['origin_label', 'initialization', 'law', 'mean_N', 'median_N', 'net_drift', 'completions', 'sojourn_median_s', 'fraction_gt8', 'time_outside_measured_B_s', 'max_N']})
    (R / 'data/processed/simulation_summary.json').write_text(json.dumps([{k: v for k, v in q.items() if k not in ['segments', 'events']} for q in outputs], indent=2))
    print('Completed', len(outputs), 'deterministic scenarios; all flow accounting checks passed.')
if __name__ == '__main__':
    main()
