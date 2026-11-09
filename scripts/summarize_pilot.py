"""Summarize completed validation calls; never substitute for real inference."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from oraclepredict.common import ROOT,PROCESSED,dump,utcnow

def main():
    frame=pd.read_parquet(ROOT/'runs/agents_pilot.parquet')
    labels=pd.read_parquet(PROCESSED/'validation_labels.parquet')
    scored=frame.merge(labels[['sample_id','label']],on='sample_id',validate='many_to_one')
    assert len(scored)==len(frame) and scored.sample_id.nunique()==32
    scored['brier']=(scored.p_used-scored.label)**2
    metrics=scored.groupby('method').agg(N=('sample_id','nunique'),brier=('brier','mean'),fallback=('fallback_used','sum'),latency_p50_ms=('total_latency_ms','median'))
    metrics.to_csv(ROOT/'review/pilot_metrics.csv')
    calls=[json.loads(p.read_text()) for p in (ROOT/'runs/agents_pilot/calls').rglob('*.json')]
    execution=json.loads((ROOT/'runs/agents_pilot_execution.json').read_text());cfg=execution['cfg']
    test=pd.read_parquet(ROOT/'metadata/split_manifest.parquet').query("split=='test'")
    estimate=execution['elapsed_seconds']*len(test)*len(cfg['seeds'])/32
    budget={'created_at':utcnow(),'pilot_requests':32,'physical_calls':len(calls),'failed_physical_calls':sum(c['error'] is not None for c in calls),'input_tokens':sum((c.get('response',{}).get('usage') or {}).get('prompt_tokens',0) for c in calls),'output_tokens':sum((c.get('response',{}).get('usage') or {}).get('completion_tokens',0) for c in calls),'pilot_elapsed_seconds':execution['elapsed_seconds'],'formal_cases':len(test),'formal_repeats':len(cfg['seeds']),'projected_formal_seconds':estimate,'projected_allocated_GPU_hours':estimate*4/3600,'dollar_cost':None,'dollar_cost_reason':'No billing tariff supplied; allocated wall time is not measured GPU busy utilization','hard_limits':{'calls_per_case_repeat_including_retry':18,'output_tokens_per_call':cfg['max_tokens'],'tool_calls':cfg['tool_calls_max'],'concurrency':cfg['concurrency']},'adaptation':'Same scientific prompts, model weights and sampling settings. Runtime migration and preprocessing correction explicitly recorded; old test outcomes already seen.'}
    dump(ROOT/'metadata/pilot_budget.json',budget)
    frame.loc[frame.fallback_used,['sample_id','method','repeat','error_type']].to_csv(ROOT/'review/pilot_failures.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    for ax,col,title in zip(axes,['brier','latency_p50_ms'],['Validation Brier (lower is better)','Median system latency (ms)']):
        ax.bar(metrics.index,metrics[col],color='#336b94');ax.set_title(title,fontsize=10);ax.tick_params(labelsize=9)
    fig.suptitle('32-request executed validation pilot; not formal test',fontsize=11)
    fig.savefig(ROOT/'Results/Figures/pilot_results.pdf');fig.savefig(ROOT/'Results/Figures/pilot_results.svg');plt.close(fig)
    text=f'''# Executed validation pilot

32 chronologically preselected validation requests; one repeat per method. The
completed run issued {len(calls)} physical HTTP calls, of which
{budget['failed_physical_calls']} failed, and produced {int(frame.fallback_used.sum())}
fallback outputs among {len(frame)} method outputs. Elapsed time was
{execution['elapsed_seconds']:.1f} seconds. These are feasibility observations,
not the main test results.

```
{metrics.to_string()}
```

Provider: {cfg['provider']}. Hardware: four {cfg['hardware_GPU_type']} GPUs,
concurrency {cfg['concurrency']}. The same weights, prompts, sampling settings and
tool budgets are used for all four methods. Original SGLang pilot/outage records
remain under `runs/attempt01_outage`; the corrected rerun is explicitly retrospective.

Linear extrapolation gives {estimate/60:.1f} minutes and
{estimate*4/3600:.2f} allocated GPU-hours for {len(test)} requests × {len(cfg['seeds'])}
repeats. This estimate excludes cold start and is not measured formal cost.
No dollar tariff or GPU busy utilization is fabricated. Actual token counts,
call traces, failures and the pilot figure are retained.
'''
    (ROOT/'review/pilot_report.md').write_text(text)
    print(json.dumps(budget,indent=2))

if __name__=='__main__':main()
