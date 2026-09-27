"""Rebuild the frozen service curve from 30 cohorts, without GPU measurements."""
import csv
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1]

def main():
    groups = {}
    with (R / 'data/inputs/calibration_intervals.csv').open() as f:
        for x in csv.DictReader(f):
            groups.setdefault((int(x['B']), int(x['rep'])), []).append(float(x['interval_ms']))
    assert len(groups) == 30 and all((len(v) == 70 for v in groups.values()))
    rows = []
    for b in sorted({k[0] for k in groups}):
        means = np.array([np.mean(v) for (batch, rep), v in groups.items() if batch == b])
        t = float(means.mean())
        mu = b * 1000 / t
        rows.append(dict(B=b, T_ms=t, mu_tok_s=mu, mu_req_s=mu / 128, cohorts=len(means), cohort_CV_pct=float(means.std(ddof=1) / t * 100), regime='graph' if b <= 8 else 'eager'))
    with (R / 'data/processed/service_curve.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('Rebuilt 10 service points from 30 cohorts and 2,100 intervals.')
if __name__ == '__main__':
    main()
