"""Check conservation, an analytic service case and frozen numerical evidence."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import simulate_state_dependent_queue as model

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / 'data' / 'processed' / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    old_rate = model.rate
    try:
        model.rate = lambda n, law: 256. * n
        result = model.simulate([64, 128], 'analytic_check')
        assert math.isclose(result['sojourn_mean_s'], .5, abs_tol=1e-8)
        assert math.isclose(result['sojourn_median_s'], .5, abs_tol=1e-8)
        assert result['N_end'] == 4 and result['completions'] == 598
    finally:
        model.rate = old_rate

    reference = json.loads((ROOT / 'data/inputs/frozen_simulation_reference.json').read_text())
    output = json.loads((ROOT / 'data/processed/simulation_summary.json').read_text())
    assert len(output) == len(reference) == 60
    key = lambda x: (x['rep'], x['origin_label'], x['initialization'], x['law'])
    frozen = {key(x): x for x in reference}
    for actual in output:
        expected = frozen[key(actual)]
        assert actual.keys() == expected.keys()
        for field, value in actual.items():
            target = expected[field]
            if isinstance(value, (float, int)):
                assert math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-8), (key(actual), field)
            else:
                assert value == target, (key(actual), field)
        assert actual['N_end'] - actual['N_initial'] == actual['arrivals'] - actual['completions']
        if actual['initialization'] in ['empirical', 'uniform']:
            assert actual['time_outside_measured_B_s'] == 0

    curve = {int(x['B']): x for x in rows('service_curve.csv')}
    assert math.isclose(float(curve[8]['mu_req_s']), 19.166444088881068, abs_tol=1e-10)
    assert math.isclose(float(curve[9]['mu_req_s']), 6.936825948314298, abs_tol=1e-10)
    cells = {(int(x['cap']), int(x['B'])): x for x in rows('step2_summary.csv')}
    cohorts = rows('step2_repetitions.csv')
    for (cap, batch), cell in cells.items():
        values = [float(x['mean_ms']) for x in cohorts if int(x['cap']) == cap and int(x['batch']) == batch]
        assert len(values) == 10
        assert math.isclose(np.mean(values), float(cell['mean_ms']), abs_tol=1e-10)
    assert round(float(cells[8, 9]['mean_ms']), 3) == 10.892
    assert round(float(cells[16, 9]['mean_ms']), 3) == 3.655
    assert round(float(cells[8, 9]['mean_ms']) / float(cells[16, 9]['mean_ms']), 2) == 2.98

    real = rows('step4_summary.csv')
    for name in ['step3_summary.csv', 'step4_summary.csv', 'step4_blocks.csv']:
        for x in rows(name):
            assert int(x['N_end']) - int(x['N_start']) == int(x['A']) - int(x['C'])
    for x in rows('model_comparison.csv'):
        sim = next(v for v in output if v['rep'] == int(x['rep']) and v['origin_label'] == x['history']
                   and v['initialization'] == 'empirical' and v['law'] == 'measured')
        obs = next(v for v in real if v['rep'] == x['rep'] and v['history'] == x['history'] and v['cap'] == '8')
        assert math.isclose(sim['mean_N'], float(x['sim_mean_N']), abs_tol=1e-8)
        assert math.isclose(float(obs['mean_N']), float(x['real_mean_N']), abs_tol=1e-10)
        assert math.isclose(sim['sojourn_median_s'], float(x['sim_sojourn_median']), abs_tol=1e-8)
    print('PASS: analytic case, 60 frozen scenarios, conservation, causal contrast and service cliff.')


if __name__ == '__main__':
    main()
