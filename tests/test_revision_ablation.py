"""Scientific safeguards for the new tool-removal intervention."""
import json
from pathlib import Path

import pandas as pd
import pytest

from oraclepredict.common import ROOT
from oraclepredict.revision_ablation import NoPredictorStore, ArmCaller


def test_predictor_removal_is_enforced_by_server():
    features = pd.read_parquet(ROOT/'Data/Processed_Data/features.parquet').set_index('sample_id')
    manifest = pd.read_parquet(ROOT/'metadata/split_manifest.parquet').set_index('sample_id')
    sid = features.index[0]
    store = NoPredictorStore(features,manifest.proposal_time,.5)
    assert store.predictor is None
    session = store.session(sid)
    with pytest.raises(ValueError,match='Predictor access disabled'):
        session.execute('predict_tabular',{})
    assert session.payloads == []
    result = session.execute('get_actor_history',{'role':'proposer'})
    assert set(result['result']) == {x for x in features if x.startswith('proposer_prior_')}
    with pytest.raises(ValueError):session.execute('get_proposal',{'case_id':'different'})
    with pytest.raises(ValueError):session.execute('get_proposal',{'as_of':'2100-01-01'})


def test_arm_prompts_are_isolated_and_control_unchanged(tmp_path):
    cfg = json.loads((ROOT/'configs/revision_20261106/ablation.json').read_text())
    a = ArmCaller(cfg,17,tmp_path/'a','without_predictor')
    b = ArmCaller(cfg,17,tmp_path/'b','with_predictor')
    assert b.prompts['A1'] == (ROOT/'prompts/A1.txt').read_text()
    assert 'predict_tabular' not in a.prompts['A1']
    for key in ['common','verifier','revise']:
        assert a.prompts[key] == b.prompts[key] == (ROOT/f'prompts/{key}.txt').read_text()
