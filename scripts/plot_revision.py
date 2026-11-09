"""Editable manuscript figures, rendered only from saved, audited results.

November 9 layout: separate the executed workflow from evaluation questions;
separate predictor-access effects from verification effects. No inference runs.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Results/revision_20261106'
FIG = ROOT / 'paper/figures'
BLUE = '#3875a5'
NAVY = '#224f76'
TEAL = '#147d79'
ORANGE = '#c46a32'
INK = '#243846'
GRAY = '#778691'
PALE_BLUE = '#edf4fa'
PALE_TEAL = '#eaf5f2'
PALE_ORANGE = '#fcf1e7'
GRID = '#e1e7ec'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.2,
    'axes.titlesize': 8.6, 'axes.labelsize': 7.2,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 7.0,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .6, 'savefig.facecolor': 'white',
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(FIG / f'{name}.{suffix}', dpi=240,
                    bbox_inches='tight', pad_inches=.035)
    plt.close(fig)


def method_figure():
    # The method figure names executed procedures and contains no numerals.
    from draw_method_schematic import draw
    draw()


def results_figure():
    main = pd.read_csv(ROOT / 'Results/Tables/main_results.csv').set_index('method')
    original_pairs = pd.read_csv(ROOT / 'Results/Tables/comparisons.csv')
    revision = pd.read_csv(OUT / 'ablation_results.csv').set_index('method')
    pairs = pd.read_csv(OUT / 'paired_comparisons.csv')
    cost = pd.read_csv(OUT / 'cpu_latency_summary.csv').set_index('method')
    old_verify = pd.read_csv(OUT / 'verification_summary_original.csv').iloc[0]
    new_verify = pd.read_csv(OUT / 'verification_summary_revision.csv').set_index('arm')
    fig = plt.figure(figsize=(7.2, 4.05))

    def heading(x, y, letter, title, subtitle):
        fig.text(x, y, letter, fontsize=9, fontweight='bold', color=NAVY)
        fig.text(x+.028, y, title, fontsize=8.3, fontweight='bold')
        fig.text(x+.028, y-.032, subtitle, fontsize=6.5, color=GRAY)

    def clean(ax, zero=False):
        ax.set_axisbelow(True)
        ax.grid(axis='x', color=GRID, lw=.6)
        ax.spines[['left', 'top', 'right']].set_visible(False)
        ax.spines['bottom'].set_color('#9ba9b4')
        ax.tick_params(axis='y', length=0, pad=5)
        ax.tick_params(axis='x', length=2.5, pad=3)
        if zero:
            ax.axvline(0, color=GRAY, lw=.85, ls=(0, (2, 2)), zorder=1)

    def interval(ax, d, y, color, marker='o', multiplier=1, size=3.8):
        value = d['delta'] if 'delta' in d else d['brier']
        ax.errorbar(value*multiplier, y,
            xerr=[[(value-d.ci_low)*multiplier], [(d.ci_high-value)*multiplier]],
            fmt=marker, ms=size, color=color, elinewidth=1.0,
            capsize=2, capthick=.85, zorder=3)

    legend = [
        Line2D([], [], marker='s', color=BLUE, lw=0, markersize=4, label='Conventional'),
        Line2D([], [], marker='D', color=NAVY, lw=0, markersize=4, label=r'Selected $B^\star$'),
        Line2D([], [], marker='o', color=TEAL, lw=0, markersize=4, label='LLM + predictor access'),
        Line2D([], [], marker='o', color=ORANGE, lw=0, markersize=4, label='LLM − predictor access'),
    ]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.52, 1.015),
               ncol=4, frameon=False, fontsize=6.7, columnspacing=1.4, handletextpad=.4)

    heading(.018, .925, 'A', 'Predictive performance', 'Historical test · marginal 95% CIs')
    heading(.542, .925, 'B', 'Predictor-access comparison', 'Access ablation · paired 95% CIs')
    heading(.018, .438, 'C', 'Brier score and warm latency', 'Warm runs · CPU models / GPU LLMs')
    heading(.542, .438, 'D', 'Verification comparison', 'A2 − A1 · paired 95% CIs')

    # Main comparison, visually grouped; the validation-selected baseline is fixed.
    ax = fig.add_axes([.15, .605, .285, .266])
    methods = ['Prior', 'LR', 'RF', 'GBM', 'Ensemble', 'D0', 'A1', 'A2', 'D_match']
    labels = ['Prior', 'Logistic', 'RF', 'LightGBM', r'$B^\star$', 'Direct (D0)',
              'Planner (A1)', 'Verifier (A2)', 'Matched evidence']
    ys = [0, 1, 2, 3, 4, 5.7, 7.1, 8.1, 9.1]
    ax.axhspan(-.5, 4.5, facecolor=PALE_BLUE, zorder=0)
    ax.axhspan(6.6, 9.6, facecolor=PALE_TEAL, zorder=0)
    for m, y in zip(methods, ys):
        color = NAVY if m == 'Ensemble' else BLUE if m in methods[:5] else GRAY if m == 'D0' else TEAL
        marker = 'D' if m == 'Ensemble' else 's' if m in methods[:5] else '^' if m == 'A2' else 'o'
        interval(ax, main.loc[m], y, color, marker)
        ax.text(1.04, y, f'{main.loc[m, "brier"]:.4f}', transform=ax.get_yaxis_transform(),
                va='center', fontsize=6.3, color=color)
    ax.axvline(main.loc['Ensemble', 'brier'], color=NAVY, ls=(0, (2, 2)), lw=.7)
    ax.set(yticks=ys, yticklabels=labels, ylim=(9.8, -.7), xlim=(.205, .358),
           xticks=[.22, .26, .30, .34], xlabel='Brier score ↓')
    ax.tick_params(axis='y', labelsize=6.6)
    ax.set_xlabel('Brier score ↓', labelpad=3)
    clean(ax)
    primary = original_pairs.query("left == 'A2' and right == 'Ensemble'").iloc[0]
    matched = original_pairs.query("left == 'A1' and right == 'D_match'").iloc[0]
    for y, label, d in [(.510, r'A2 − $B^\star$', primary), (.479, 'A1 − D$_{match}$', matched)]:
        fig.text(.045, y, f'{label}: {d.delta:+.5f}  [{d.ci_low:+.5f}, {d.ci_high:+.5f}]',
                 fontsize=6.5, color=INK)

    # One scale for the two access effects, with the underlying scores explicit.
    ax = fig.add_axes([.692, .660, .274, .185])
    access_rows = []
    for y, left, right, label in [
        (0, 'A1_noB', 'A1_withB', 'Planner (A1)'),
        (1, 'A2_noB', 'A2_withB', 'Verifier (A2)'),
    ]:
        d = pairs.query('left == @left and right == @right').iloc[0]
        interval(ax, d, y, ORANGE, 'o' if y == 0 else '^', size=5)
        ax.text(d.delta, y-.21, f'{d.delta:+.4f}', fontsize=7.3, color=ORANGE,
                fontweight='bold', ha='center')
        access_rows.append({'label': label, **d.to_dict()})
    ax.set(yticks=[0, 1], yticklabels=['Planner (A1)', 'Verifier (A2)'],
           ylim=(1.6, -.6), xlim=(-.005, .132), xticks=[0, .04, .08, .12])
    ax.set_xlabel(r'Δ Brier: without − with $B^\star$', labelpad=4)
    clean(ax, zero=True)
    fig.text(.57, .548, 'Brier with → without access', fontsize=6.5, color=GRAY)
    fig.text(.57, .518,
             f'A1   {revision.loc["A1_withB", "brier"]:.4f} → {revision.loc["A1_noB", "brier"]:.4f}'
             f'       A2   {revision.loc["A2_withB", "brier"]:.4f} → {revision.loc["A2_noB", "brier"]:.4f}',
             fontsize=7.0)
    fig.text(.57, .484, 'Higher observed loss without predictor access.',
             fontsize=6.6, color=ORANGE)

    # All five measured conventional models, plus the four access-ablation systems.
    ax = fig.add_axes([.104, .095, .365, .285])
    latency_rows = []
    baseline_positions = {
        'Prior': ('Prior', (4, 5)), 'LR': ('Logistic', (-20, 6)),
        'GBM': ('LightGBM', (-22, -14)), 'RF': ('RF', (-14, 10)),
        'Ensemble': (r'$B^\star$', (10, -12)),
    }
    for method, (label, offset) in baseline_positions.items():
        x, y = cost.loc[method, 'p50_ms']/1000, main.loc[method, 'brier']
        color, marker = (NAVY, 'D') if method == 'Ensemble' else (BLUE, 's')
        ax.scatter(x, y, c=color, marker=marker, s=18, zorder=4)
        ax.annotate(label, (x, y), xytext=offset, textcoords='offset points', fontsize=6.4,
                    color=color, arrowprops=dict(arrowstyle='-', color=color, lw=.4))
        latency_rows.append(dict(method=method, brier=float(y), median_seconds=float(x), source='cpu_latency_summary.csv'))
    offsets = {'A1_withB': (-30, -11), 'A2_withB': (-32, 13),
               'A1_noB': (-42, 11), 'A2_noB': (4, -12)}
    for method, offset in offsets.items():
        d = revision.loc[method]
        color = ORANGE if method.endswith('noB') else TEAL
        x, y = d.latency_p50_ms/1000, d.brier
        ax.scatter(x, y, c=color, marker='o' if method.startswith('A1') else '^', s=24, zorder=4)
        ax.annotate(method[:2], (x, y), xytext=offset, textcoords='offset points', fontsize=6.7,
                    color=color, arrowprops=dict(arrowstyle='-', color=color, lw=.5))
        latency_rows.append(dict(method=method, brier=float(y), median_seconds=float(x), source='ablation_results.csv'))
    ax.set(xscale='log', xlim=(.00065, 330), ylim=(.205, .35),
           yticks=[.23, .27, .31, .35], ylabel='Brier score ↓')
    ax.set_xlabel('Median warm system latency (s, log) →', labelpad=3)
    ax.xaxis.set_major_locator(FixedLocator([.001, .01, .1, 1, 10, 100]))
    ax.xaxis.set_minor_locator(NullLocator())
    clean(ax)
    ax.grid(axis='y', color=GRID, lw=.6)

    # Loss magnitude and uncertainty, rather than counting votes for/against a verifier.
    ax = fig.add_axes([.70, .165, .19, .195])
    verify_rows = [
        ('Main', original_pairs.query("left == 'A2' and right == 'A1'").iloc[0], old_verify, GRAY),
        (r'With $B^\star$', pairs.query("left == 'A2_withB' and right == 'A1_withB'").iloc[0], new_verify.loc['with_predictor'], TEAL),
        (r'Without $B^\star$', pairs.query("left == 'A2_noB' and right == 'A1_noB'").iloc[0], new_verify.loc['without_predictor'], ORANGE),
    ]
    for y, (label, d, count, color) in enumerate(verify_rows):
        interval(ax, d, y, color, multiplier=1000, size=4.1)
        suffix = '†' if count.fallback_associated_changes else ''
        ax.text(1.16, y, f'{int(count.changed)}/471{suffix}\n{int(count.improved)} / {int(count.worsened)}',
                transform=ax.get_yaxis_transform(), fontsize=6.3, ha='center', va='center', color=color)
    ax.set(yticks=[0, 1, 2], yticklabels=[v[0] for v in verify_rows],
           ylim=(2.5, -.5), xlim=(-8.2, 5.1), xticks=[-8, -4, 0, 4])
    ax.tick_params(axis='y', labelsize=6.8)
    ax.set_xlabel('Δ Brier ($\\times 10^{-3}$)', labelpad=4)
    clean(ax, zero=True)
    ax.text(1.16, 1.08, 'Changed\nhelp / harm', transform=ax.transAxes,
            fontsize=6.0, ha='center', va='bottom', color=GRAY)
    fig.text(.605, .054, '← verification helps', fontsize=6.4, color=TEAL)
    fig.text(.80, .054, 'verification harms →', fontsize=6.4, color=ORANGE)
    fig.text(.57, .018, '† Includes one fallback-associated change (harm).', fontsize=6.2, color=GRAY)

    save(fig, 'results')
    # A compact provenance record makes figure values independently inspectable.
    values = {
        'data_revision': 'revision_20261106', 'layout_revision': '20261109',
        'sample': {'requests': 157, 'questions': 120, 'LLM_repeats': 3},
        'main_performance': main.loc[methods, ['brier', 'ci_low', 'ci_high']].reset_index().to_dict('records'),
        'main_paired_callouts': [primary.to_dict(), matched.to_dict()],
        'predictor_access': access_rows, 'latency': latency_rows,
        'verification': [{'label': label, **d.to_dict(),
            'changed': int(c.changed), 'improved': int(c.improved), 'worsened': int(c.worsened),
            'fallback_associated_changes': int(c.fallback_associated_changes)}
            for label, d, c, _ in verify_rows],
        'scope': 'Exploratory comparisons on the same historical test; no new model calls or timing measurements.',
    }
    def finite(value):
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    (FIG / 'plotted_values.json').write_text(json.dumps(finite(values), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method-only', action='store_true')
    args = parser.parse_args()
    method_figure()
    if not args.method_only:
        results_figure()
