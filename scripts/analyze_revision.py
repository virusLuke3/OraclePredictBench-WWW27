"""Rebuild November revision analyses from immutable per-request records.

No model calls, refitting, selection or prompt changes occur in this script.
Run with --original-only while the new inference experiment is still running.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oraclepredict.common import ROOT, PROCESSED, dump, digest, utcnow
from oraclepredict.evaluate import cluster_bootstrap, calendar_supergroups

OUT = ROOT / 'Results/revision_20261106'
RUN = ROOT / 'runs/revision_20261106'
REV = ROOT / 'review/revision_20261106'


def interval(draws):
    low, high = np.quantile(draws, [.025, .975])
    return {'ci_low': float(low), 'ci_high': float(high)}


def verification(directory, labels, name):
    rows = []; calls = []; tools = []
    baseline = pd.read_parquet(ROOT/'runs/baseline_test.parquet').query("method=='Ensemble'").set_index('sample_id').p_used
    for path in sorted(directory.glob('*.json')):
        case = json.loads(path.read_text())
        records = {r['method']: r for r in case['records']}
        a1 = next(v for k, v in records.items() if k == 'A1' or k.startswith('A1_'))
        a2 = next(v for k, v in records.items() if k == 'A2' or k.startswith('A2_'))
        y = int(labels.loc[case['sample_id'], 'label'])
        delta = (a2['p_used']-y)**2 - (a1['p_used']-y)**2
        findings = case.get('verifier') or {}
        p_baseline = float(baseline.loc[case['sample_id']])
        before = abs(a1['p_used']-p_baseline); after = abs(a2['p_used']-p_baseline)
        changed = abs(a1['p_used']-a2['p_used']) > 1e-12
        fallback_in_pair = bool(a1['fallback_used'] or a2['fallback_used'])
        revision_called = any(c['role']=='revise' for c in case['actual_calls'])
        rows.append(dict(sample_id=case['sample_id'], repeat=case['repeat'],
            arm=case.get('arm', 'original_with_predictor'), label=y,
            p_A1=a1['p_used'], p_A2=a2['p_used'], delta_loss=delta,
            changed=changed,
            improved=delta < -1e-12, worsened=delta > 1e-12,
            needs_revision=findings.get('needs_revision', False),
            revision_called=revision_called,
            valid_revision_change=changed and revision_called and not fallback_in_pair,
            changed_with_fallback=changed and fallback_in_pair,
            p_baseline=p_baseline, distance_to_baseline_before=before, distance_to_baseline_after=after,
            moved_toward_baseline=after < before-1e-12,
            verifier_claimed_error_types=json.dumps([e.get('type') for e in findings.get('errors',[])]),
            A1_fallback=a1['fallback_used'], A2_fallback=a2['fallback_used'],
            verification_extra_ms=a2['total_latency_ms']-a1['total_latency_ms'],
            findings=json.dumps(findings, sort_keys=True)))
        for number, call in enumerate(case['actual_calls']):
            usage = (call.get('response') or {}).get('usage', {})
            calls.append(dict(sample_id=case['sample_id'], repeat=case['repeat'],
                arm=case.get('arm', 'original_with_predictor'), number=number,
                role=call['role'], attempt=call['attempt'], error=call.get('error'),
                latency_ms=call['latency_ms'], input_tokens=usage.get('prompt_tokens'),
                output_tokens=usage.get('completion_tokens'), usage_known=bool(usage)))
        for item in case['trace']:
            payload = item['payload']
            tools.append(dict(sample_id=case['sample_id'], repeat=case['repeat'],
                arm=case.get('arm', 'original_with_predictor'), tool=payload['tool'],
                payload_sha256=payload['sha256']))
    frame = pd.DataFrame(rows); call_frame = pd.DataFrame(calls)
    tool_frame = pd.DataFrame(tools)
    frame.to_csv(OUT/f'verification_{name}.csv', index=False)
    call_frame.to_csv(OUT/f'physical_calls_{name}.csv', index=False)
    tool_frame.to_csv(OUT/f'tool_calls_{name}.csv', index=False)
    summaries = []
    for arm, f in frame.groupby('arm'):
        valid=f[f.valid_revision_change]; fallback=f[f.changed_with_fallback]
        assert int(f.changed.sum()) == len(valid)+len(fallback)
        summaries.append(dict(arm=arm, N=int(f.sample_id.nunique()), outputs=len(f),
            changed=int(f.changed.sum()), changed_requests=int(f.loc[f.changed,'sample_id'].nunique()),
            improved=int(f.improved.sum()), worsened=int(f.worsened.sum()),
            moved_toward_baseline=int(f.moved_toward_baseline.sum()),
            valid_revision_changes=len(valid), valid_revision_improved=int(valid.improved.sum()),
            valid_revision_worsened=int(valid.worsened.sum()),
            fallback_associated_changes=len(fallback), fallback_associated_improved=int(fallback.improved.sum()),
            fallback_associated_worsened=int(fallback.worsened.sum()),
            valid_revision_delta_contribution=float(valid.delta_loss.sum()/len(f)),
            fallback_delta_contribution=float(fallback.delta_loss.sum()/len(f)),
            requested_revision=int(f.needs_revision.sum()), revision_called=int(f.revision_called.sum()),
            A1_fallback=int(f.A1_fallback.sum()), A2_fallback=int(f.A2_fallback.sum()),
            mean_delta_brier=float(f.groupby('sample_id').delta_loss.mean().mean()),
            median_extra_seconds=float(f.verification_extra_ms.median()/1000)))
    pd.DataFrame(summaries).to_csv(OUT/f'verification_summary_{name}.csv', index=False)
    costs = dict(physical_calls=len(call_frame), successful_calls=int(call_frame.error.isna().sum()),
        failed_calls=int(call_frame.error.notna().sum()), retry_calls=int(call_frame.attempt.gt(0).sum()),
        known_input_tokens=int(call_frame.input_tokens.sum()), known_output_tokens=int(call_frame.output_tokens.sum()),
        calls_with_unknown_usage=int((~call_frame.usage_known).sum()),
        dollar_cost=None, usage_note='Known tokens are a lower bound when failed-call usage is unavailable; not zero.',
        shared_prefix_note='Each physical call counted once; per-method costs include the prefix they require.')
    dump(OUT/f'physical_cost_{name}.json', costs)
    return frame, summaries, costs


def audit_new_cases(manifest, base):
    from oraclepredict.data import FEATURES
    expected = base.query("method=='Ensemble'").set_index('sample_id').p_used
    frozen = REV/'ablation_frozen_protocol.json'
    protocol_hash = digest(frozen)
    counts = {'with_predictor': 0, 'without_predictor': 0}
    predictor_calls = {'with_predictor': 0, 'without_predictor': 0}
    cases_with_predictor = {'with_predictor': 0, 'without_predictor': 0}
    forbidden = {'label','resolved_price','resolution_class','v_resolved','disputed','settlement_time'}
    for path in (RUN/'agents_test').glob('*.json'):
        c = json.loads(path.read_text()); arm = c['arm']; sid = c['sample_id']
        assert c['protocol_sha256'] == protocol_hash
        assert sid in set(manifest.sample_id)
        assert set(c['packet']['features']) == set(FEATURES)
        assert not set(c['packet']) & forbidden
        assert set(c['packet']) == {'evidence_id','as_of','features','training_prior'}
        cases_with_predictor[arm] += int(any(t['payload']['tool']=='predict_tabular' for t in c['trace']))
        for entry in c['trace']:
            t = entry['payload']
            assert t['as_of'] == c['packet']['as_of']
            if t['tool'] == 'predict_tabular':
                assert arm == 'with_predictor'
                assert abs(t['result']['p_reject']-expected.loc[sid]) < 1e-12
                predictor_calls[arm] += 1
        counts[arm] += 1
    assert counts == {'with_predictor': 471, 'without_predictor': 471}, counts
    return dict(case_arm_records=counts, predictor_calls=predictor_calls,
        case_arms_with_predictor_calls=cases_with_predictor,
        labels_only_joined_after_inference=True, features_allowlisted=True,
        control_predictor_matches_frozen_baseline=True, protocol_sha256=protocol_hash)


def main(original_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(PROCESSED/'test_labels.parquet').set_index('sample_id')
    manifest = pd.read_parquet(ROOT/'metadata/split_manifest.parquet').query("split=='test'").sort_values(['proposal_time','sample_id'])
    original, old_summary, old_cost = verification(ROOT/'runs/agents_test', labels, 'original')
    # Verify the trace-based decomposition against the previously frozen score.
    old_pairs = pd.read_csv(ROOT/'Results/Tables/comparisons.csv')
    old_delta = old_pairs.query("left=='A2' and right=='A1'").delta.iloc[0]
    assert abs(old_summary[0]['mean_delta_brier']-old_delta) < 1e-12
    if original_only:
        print(json.dumps({'verification':old_summary,'cost':old_cost}, indent=2)); return
    cfg = json.loads((ROOT/'configs/revision_20261106/ablation.json').read_text())
    agent = pd.read_parquet(RUN/'agents_test.parquet')
    base = pd.read_parquet(ROOT/'runs/baseline_test.parquet')
    audit = audit_new_cases(manifest, base)
    assert len(agent) == 1884 and not agent.duplicated(['sample_id','repeat','method']).any()
    for method, f in agent.groupby('method'):
        assert set(f.sample_id) == set(manifest.sample_id)
        assert f.groupby('sample_id').size().eq(3).all()
        assert f.p_used.between(0,1).all()
    verification_frame, verification_summary, costs = verification(RUN/'agents_test', labels, 'revision')
    scored = pd.concat([base.query("method=='Ensemble'"), agent], ignore_index=True).merge(labels, on='sample_id', validate='many_to_one')
    scored['loss'] = (scored.p_used-scored.label)**2
    scored.to_parquet(OUT/'predictions_scored.parquet', index=False)
    methods = ['Ensemble','A1_withB','A2_withB','A1_noB','A2_noB']
    loss = scored.groupby(['sample_id','method']).loss.mean().unstack().loc[manifest.sample_id, methods]
    loss.to_csv(OUT/'per_request_loss.csv')
    boot = cluster_bootstrap(loss, manifest.group_id, cfg['bootstrap_draws'], cfg['bootstrap_seed'])
    np.savez_compressed(OUT/'bootstrap_indices.npz', indices=boot['indices'], N=boot['N'],
                       group_labels=np.asarray(boot['group_labels'], dtype=str))
    pd.DataFrame(boot['means'], columns=methods).to_parquet(OUT/'bootstrap_means.parquet', index=False)
    cpu_latency = pd.read_csv(OUT/'cpu_latency_summary.csv').set_index('method').loc['Ensemble']
    results = []
    for j, method in enumerate(methods):
        d = scored.query('method==@method')
        diff = boot['means'][:,j]-boot['means'][:,0]
        results.append(dict(method=method, N=len(manifest), G=boot['G'], repeats=int(d.repeat.nunique()),
            brier=float(loss[method].mean()), **interval(boot['means'][:,j]),
            delta_vs_best=float(loss[method].mean()-loss.Ensemble.mean()),
            delta_ci_low=interval(diff)['ci_low'], delta_ci_high=interval(diff)['ci_high'],
            latency_p50_ms=float(cpu_latency.p50_ms if method=='Ensemble' else d.total_latency_ms.median()),
            latency_p95_ms=float(cpu_latency.p95_ms if method=='Ensemble' else d.total_latency_ms.quantile(.95)),
            preparation_p50_ms=float(cpu_latency.preparation_p50_ms if method=='Ensemble' else d.preparation_ms.median()),
            latency_source='cpu_latency_replay' if method=='Ensemble' else 'access_ablation',
            timed_outputs=int(cpu_latency.timed_outputs) if method=='Ensemble' else len(d),
            fallback_outputs=int(d.fallback_used.sum()),
            mean_llm_calls=float(d.llm_calls.mean()),
            mean_tool_calls=float(d.tool_calls.mean()),
            mean_model_ms=float(d.model_latency_ms.mean()) if method!='Ensemble' else None))
    pd.DataFrame(results).to_csv(OUT/'ablation_results.csv', index=False)
    pairs = [(a,b,'access_or_verification') for a,b in cfg['comparisons']]
    pairs += [(a,'Ensemble','vs_frozen_baseline') for a in methods[1:]]
    comps = []; sensitivity = []
    for a,b,kind in pairs:
        i,j=methods.index(a),methods.index(b)
        comps.append(dict(left=a,right=b,family=kind,N=len(manifest),G=boot['G'],
            delta=float(loss[a].mean()-loss[b].mean()),**interval(boot['means'][:,i]-boot['means'][:,j])))
    pd.DataFrame(comps).to_csv(OUT/'paired_comparisons.csv', index=False)
    for days in cfg['calendar_days']:
        groups = calendar_supergroups(manifest, days)
        cb = cluster_bootstrap(loss, groups, cfg['bootstrap_draws'], cfg['bootstrap_seed'])
        for a,b,kind in pairs:
            i,j=methods.index(a),methods.index(b)
            sensitivity.append(dict(calendar_days=days,left=a,right=b,G=cb['G'],
                delta=float(loss[a].mean()-loss[b].mean()),**interval(cb['means'][:,i]-cb['means'][:,j])))
    pd.DataFrame(sensitivity).to_csv(OUT/'calendar_sensitivity.csv', index=False)
    scored.groupby(['method','repeat']).agg(N=('sample_id','size'),brier=('loss','mean'),fallback=('fallback_used','sum')).to_csv(OUT/'run_stability.csv')
    for v in verification_summary:
        suffix = 'withB' if v['arm']=='with_predictor' else 'noB'
        expected_delta = next(c['delta'] for c in comps if c['left']=='A2_'+suffix and c['right']=='A1_'+suffix)
        assert abs(v['mean_delta_brier']-expected_delta)<1e-12
    execution = json.loads((RUN/'agents_test_execution.json').read_text())
    costs.update(elapsed_seconds=execution['elapsed_seconds'], allocated_GPU_hours=execution['allocated_GPU_hours'],
        initialization_seconds=execution['initialization_seconds'], cold_start_included=False)
    dump(OUT/'physical_cost_revision.json', costs)
    summary = dict(analyzed_at=utcnow(),N=len(manifest),G=boot['G'], outputs=len(agent),
        test_previously_seen=True, inference='Exploratory paired percentile intervals; no new significance claims',
        comparisons=comps,verification=verification_summary,costs=costs,audit=audit,
        hashes={str(p.relative_to(ROOT)):digest(p) for p in [RUN/'agents_test.parquet',PROCESSED/'test_labels.parquet',REV/'ablation_frozen_protocol.json',Path(__file__)]})
    dump(OUT/'summary.json', summary); dump(REV/'ablation_execution_audit.json',audit)
    print(pd.DataFrame(results)[['method','brier','delta_vs_best','fallback_outputs']].to_string(index=False))
    print(pd.DataFrame(comps).to_string(index=False))
    print(json.dumps(verification_summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--original-only',action='store_true')
    main(parser.parse_args().original_only)
