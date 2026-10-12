"""Gaussian predictions; reused v5 constants are explicitly fingerprinted."""
from __future__ import annotations
import json
import numpy as np
from v5.normal_integral import integrate
from v3.sign_limit import sign_limit
from v6.tracking import ROOT,prepare,atomic_json,stream


def main() -> None:
    c=dict(id='v6_0_gaussian_limits',seed=106150,draws=200000,pairs=10,ratios=[16000.,1.])
    out=prepare(c,['src/v6/limits.py','src/v6/tracking.py','src/v5/normal_integral.py','src/v3/sign_limit.py','src/provenance.py','provenance/v6/influence_v5_input.json'])
    if (out/'summary.json').exists():print('Retained Gaussian calibration');return
    const=json.loads((ROOT/'provenance/v6/influence_v5_input.json').read_text());k=c['pairs'];rng=stream(c,0,'gaussian')
    z=rng.normal(size=(c['draws'],k));u=rng.normal(size=(c['draws'],k));records=[];runs={}
    for ratio in c['ratios']:
        t=np.sqrt(const['ratio_constant']*ratio)
        matrices={'oracle':z,'pooled':z+t*u.sum(axis=1)[:,None]/np.sqrt(k),
                  'separate':z+np.sqrt(k)*t*u,'shared_single':z+np.sqrt(k)*t*u[:,0,None]}
        for method,w in matrices.items():
            order=np.argsort(-np.abs(w),axis=1);positive=np.take_along_axis(w,order,axis=1)>0
            r=np.cumprod(positive,axis=1).sum(axis=1);fdp=r/(k+r);runs[f'{method}_{int(ratio)}']=r.astype('uint8')
            scale=t if method=='pooled' else np.sqrt(k)*t if method=='shared_single' else 0.
            integral=integrate(scale,256)
            records.append(dict(ratio=ratio,method=method,fdr=float(fdp.mean()),mcse=float(fdp.std(ddof=1)/np.sqrt(len(fdp))),
                predicted_integral=integral,normalized_variance=1+(t*t if method=='pooled' else k*t*t if method!='oracle' else 0.),
                normalized_offdiag_covariance=t*t if method=='pooled' else k*t*t if method=='shared_single' else 0.))
    np.savez_compressed(out/'individual_runs.npz',**runs)
    atomic_json(out/'summary.json',dict(config=c,records=records,scope='200000 Gaussian vectors, reused across two ratios and four schemes; not toroidal data sets. Predictions condition on reused numerical v5 constants.'))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
