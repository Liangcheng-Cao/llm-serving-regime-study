"""Recreate four descriptive figures from the released numeric evidence."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, RED = '#246bb0', '#bf493d'


def rows(name):
    with (ROOT / 'data' / 'processed' / name).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    (ROOT / 'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT / 'figures' / name, dpi=180)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    curve = rows('service_curve.csv')
    fig, ax = plt.subplots(figsize=(7.4, 4.5), layout='constrained')
    for regime, color in [('graph', BLUE), ('eager', RED)]:
        points = [x for x in curve if x['regime'] == regime]
        ax.plot([int(x['B']) for x in points],
                [float(x['mu_req_s']) for x in points], 'o-', color=color, label=regime)
    ax.plot([8, 9], [19.1664440889, 6.9368259483], '--', color='gray')
    ax.axvline(8.5, color='gray', alpha=.3)
    ax.axhline(10, color='gray', linestyle=':', label='arrival rate = 10 req/s')
    ax.annotate('19.17 → 6.94 req/s', xy=(8.55, 13), xytext=(10, 17),
                arrowprops={'arrowstyle': '->', 'color': 'gray'})
    ax.set(title='Measured service law: a sharp execution cliff', xlabel='Actual useful batch B',
           ylabel='Capacity proxy (128-token requests/s)', xticks=[1, 2, 4, 6, 8, 9, 10, 12, 14, 16])
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=.15)
    save(fig, 'service_curve.png')

    cohorts = rows('step2_repetitions.csv')
    fig, ax = plt.subplots(figsize=(7.4, 4.5), layout='constrained')
    for i, (batch, cap) in enumerate([(8, 8), (8, 16), (9, 8), (9, 16)]):
        values = [float(x['mean_ms']) for x in cohorts
                  if int(x['batch']) == batch and int(x['cap']) == cap]
        color = RED if (batch, cap) == (9, 8) else BLUE
        ax.scatter(i + np.linspace(-.09, .09, len(values)), values, color=color, alpha=.65, s=24)
        ax.plot([i-.22, i+.22], [np.mean(values)]*2, color=color, linewidth=3)
        ax.text(i, max(values)+.5, f'{np.mean(values):.3f}', ha='center')
    ax.set(xticks=range(4), xticklabels=['B=8\ncap8 graph', 'B=8\ncap16 graph',
                                       'B=9\ncap8 eager', 'B=9\ncap16 graph'],
           ylabel='Mean completed-decode interval (ms)', ylim=(0, 13),
           title='Same useful B=9: 2.98× interval difference')
    ax.axvline(1.5, color='gray', alpha=.3)
    ax.text(.02, .97, 'Dots: 10 cohorts/cell; bars: means\nB=8 is the negative control',
            transform=ax.transAxes, va='top', fontsize=8)
    ax.grid(axis='y', alpha=.15)
    save(fig, 'same_batch_graph_vs_eager.png')

    pulse = rows('pulse_series.csv')
    fig, ax = plt.subplots(figsize=(7.4, 4.5), layout='constrained')
    for cap, color in [(8, RED), (16, BLUE)]:
        for rep in range(1, 6):
            values = [x for x in pulse if int(x['cap']) == cap and int(x['rep']) == rep]
            ax.step([float(x['t']) for x in values], [int(x['B']) for x in values],
                    where='post', color=color, alpha=.8 if rep == 1 else .2,
                    linewidth=1.5 if rep == 1 else .8,
                    label=f'graph max {cap}' if rep == 1 else None)
    ax.axvline(8, color='gray', linestyle='--', label='+6 request pulse')
    ax.axhline(8, color='gray', linestyle=':', linewidth=.8)
    ax.set(xlim=(4, 20), xlabel='Time since arrival trace start (s)', ylabel='Actual decode batch B',
           title='Finite pulse, sustained queue amplification')
    ax.legend(fontsize=8)
    ax.text(.98, .97, 'All five pairs; repetition 1 emphasized\n0.1-s display samples; arrival rate 10 req/s',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    ax.grid(alpha=.15)
    save(fig, 'pulse_response.png')

    comparison = rows('model_comparison.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout='constrained')
    for ax, real_key, sim_key, ylabel in [
        (axes[0], 'real_mean_N', 'sim_mean_N', 'Mean outstanding N (60 s)'),
        (axes[1], 'real_server_residency_median', 'sim_sojourn_median', 'Median duration (s)'),
    ]:
        for h, offset in [('LOW', 0), ('HIGH', 2.5)]:
            group = [x for x in comparison if x['history'] == h]
            for shift, key, color in [(0, real_key, BLUE), (1, sim_key, RED)]:
                vals = [float(x[key]) for x in group]
                pos = offset + shift
                ax.scatter(pos + np.linspace(-.08, .08, len(vals)), vals, color=color, s=25)
                ax.plot([pos-.18, pos+.18], [np.mean(vals)]*2, color=color, linewidth=2)
        ax.set(xticks=[0, 1, 2.5, 3.5], xticklabels=['Real\nLOW', 'Model\nLOW', 'Real\nHIGH', 'Model\nHIGH'],
               ylabel=ylabel, ylim=(0, None))
        ax.grid(axis='y', alpha=.15)
    axes[0].set_title('State separation reproduced')
    axes[1].set_title('Related duration metrics')
    fig.suptitle('Minimal classical queue vs real cap8 system', fontsize=13)
    fig.supxlabel('Five paired initial states; dots are repetitions, not confidence intervals.\nReal: server residency; model: future-job sojourn.', fontsize=9)
    save(fig, 'real_vs_queue_model.png')
    print('Recreated four figures.')


if __name__ == '__main__':
    main()
