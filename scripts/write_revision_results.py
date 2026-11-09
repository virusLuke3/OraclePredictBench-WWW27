"""Populate manuscript numbers only after the entire access ablation is complete."""
import json
from pathlib import Path
import re

import pandas as pd
from oraclepredict.common import ROOT, dump, digest, utcnow

OUT=ROOT/'Results/revision_20261106'


def signed(x,digits=4):
    return f'{x:+.{digits}f}'


def tex_ci(row,digits=4):
    return rf'${signed(row.delta,digits)}$ (95\% CI $[{signed(row.ci_low,digits)},{signed(row.ci_high,digits)}]$)'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['outputs']==1884
    results=pd.read_csv(OUT/'ablation_results.csv').set_index('method')
    pairs=pd.read_csv(OUT/'paired_comparisons.csv')
    temporal=pd.read_csv(OUT/'calendar_sensitivity.csv')
    verify=pd.read_csv(OUT/'verification_summary_revision.csv').set_index('arm')
    costs=json.loads((OUT/'physical_cost_revision.json').read_text())
    offline=json.loads((OUT/'offline_preparation_cost.json').read_text())
    cpu=pd.read_csv(OUT/'cpu_latency_summary.csv').set_index('method')
    stability=pd.read_csv(OUT/'run_stability.csv')
    def pair(a,b):return pairs.query('left==@a and right==@b').iloc[0]
    a1=pair('A1_noB','A1_withB');a2=pair('A2_noB','A2_withB')
    original=pd.read_csv(ROOT/'Results/Tables/main_results.csv').set_index('method')
    vwith=verify.loc['with_predictor'];vno=verify.loc['without_predictor']
    run_with=stability.query("method=='A1_withB'").brier
    run_no=stability.query("method=='A1_noB'").brier
    no_access_intervals=temporal.query("left in ['A1_noB','A2_noB'] and right in ['A1_withB','A2_withB']")
    if (no_access_intervals.ci_low>0).all():
        temporal_text='All six calendar-block intervals for predictor removal remain above zero.'
    elif (no_access_intervals.ci_high<0).all():
        temporal_text='All six calendar-block intervals for predictor removal remain below zero.'
    else:
        n=int(((no_access_intervals.ci_low<=0)&(no_access_intervals.ci_high>=0)).sum())
        temporal_text=f'For the fresh access intervention, {n} of six calendar-block intervals span zero.'
    if a1.ci_low>0 and a2.ci_low>0:
        abstract='Fresh predictor removal increases planner Brier by '+tex_ci(a1)+'.'
        conclusion='The access intervention supports dependence on the conventional predictor; these results do not establish an orchestration advantage beyond that predictor.'
    else:
        abstract='A fresh access intervention changes planner Brier by '+tex_ci(a1)+'.'
        conclusion='The access intervention and matched-evidence controls delimit the contribution of orchestration within this cohort.'
    macros={
        'AbstractAblation':abstract,
        'AblationFinding':('Fresh removal changes A1 Brier by '+tex_ci(a1)+' and A2 by '+tex_ci(a2)+
            '. This tests access to the predictor, including the induced change in tool policy. '
            f'Fresh A1 controls call $B^\\star$ in {summary["audit"]["case_arms_with_predictor_calls"]["with_predictor"]}/471 trajectories, '
            'versus none without access. Across repeats, A1 Brier ranges from '
            f'{run_with.min():.4f} to {run_with.max():.4f} with access and '
            f'{run_no.min():.4f} to {run_no.max():.4f} without it.'),
        'VerificationFinding':(f'In fresh controls, {int(vwith.changed)}/471 outputs change '
            f'({int(vwith.improved)} improve, {int(vwith.worsened)} worsen), with A2--A1 '
            +tex_ci(pair('A2_withB','A1_withB'),5)+'; without the predictor, '
            f'{int(vno.changed)}/471 change ({int(vno.improved)} improve, {int(vno.worsened)} worsen), with '
            +tex_ci(pair('A2_noB','A1_noB'),5)+'.'),
        'TemporalFinding':temporal_text,
        'ExecutionFinding':(f'The fresh ablation uses {costs["physical_calls"]:,} calls with '
            f'{costs["failed_calls"]} transport failures and {int(results.fallback_outputs.sum())} fallback outputs '
            f'across 1,884 predictions; its inference allocation is {costs["allocated_GPU_hours"]:.2f} GPU-hours.'),
        'PreparationFinding':(f'A separate exact replay of shared offline preparation takes {offline["elapsed_seconds"]:.2f}\\,s; '
            f'fresh $B^\\star$ warm service takes {cpu.loc["Ensemble","p50_ms"]:.2f}\\,ms median, including packet preparation.'),
        'ConclusionAblation':conclusion,
    }
    fallback_changes=int(vwith.fallback_associated_changes+vno.fallback_associated_changes)
    if fallback_changes:
        change_words='change involves' if fallback_changes==1 else 'changes involve'
        macros['VerificationFinding']+=(f' Of these, {fallback_changes} {change_words} a prior fallback '
            '(marked in Figure~\\ref{fig:results}D); trace analysis separates fallback from valid revisions.')
    text='% Generated from completed saved predictions; do not edit numeric claims manually.\n'
    text+='\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items())+'\n'
    (ROOT/'paper/generated_revision.tex').write_text(text)
    lines=[r'\begin{tabular}{@{}lrr@{}}',r'\toprule',r'System & Brier & $\Delta$ vs. $B^\star$ [95\% CI] \\',r'\midrule']
    names={'Ensemble':r'$B^\star$','D0':'D0','A1':'A1','A2':'A2','D_match':r'D$_{match}$'}
    for method in names:
        d=original.loc[method]
        value='reference' if method=='Ensemble' else rf'${d.delta_vs_best:+.4f}\ [{d.delta_ci_low:+.4f},{d.delta_ci_high:+.4f}]$'
        lines.append(rf'{names[method]} & {d.brier:.4f} & {value} \\')
    lines.append(r'\midrule')
    names={'A1_withB':r'A1, with $B^\star$','A2_withB':r'A2, with $B^\star$',
           'A1_noB':r'A1, no $B^\star$','A2_noB':r'A2, no $B^\star$'}
    for method in names:
        d=results.loc[method]
        lines.append(rf'{names[method]} & {d.brier:.4f} & ${d.delta_vs_best:+.4f}\ [{d.delta_ci_low:+.4f},{d.delta_ci_high:+.4f}]$ \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'paper/tables/revision_exact.tex').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'review/revision_20261106/manuscript_numbers.json',dict(macros=macros,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [OUT/'summary.json',OUT/'ablation_results.csv',OUT/'paired_comparisons.csv',OUT/'calendar_sensitivity.csv',OUT/'verification_summary_revision.csv',OUT/'physical_cost_revision.json',OUT/'offline_preparation_cost.json',OUT/'cpu_latency_summary.csv',OUT/'run_stability.csv',ROOT/'Results/Tables/main_results.csv']},
        generated_at=utcnow()))
    print('Wrote manuscript macros and exact table.')


if __name__=='__main__':main()
