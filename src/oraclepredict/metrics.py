import numpy as np
from autogluon.core.metrics import make_scorer

def binary_brier(y_true,y_pred,sample_weight=None,**kwargs):
    y=np.asarray(y_true,dtype=float).reshape(-1);p=np.asarray(y_pred,dtype=float)
    if p.ndim==2 and p.shape[1]==2:p=p[:,1]
    if p.ndim!=1 or p.shape!=y.shape or not len(y):raise ValueError('Invalid shape')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probability')
    if not np.isin(y,[0,1]).all():raise ValueError('Invalid label')
    return float(np.average((p-y)**2,weights=sample_weight))

BRIER=make_scorer(name='binary_brier',score_func=binary_brier,optimum=0,greater_is_better=False,needs_proba=True)
