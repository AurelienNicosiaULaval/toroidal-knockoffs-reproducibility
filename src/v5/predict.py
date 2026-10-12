"""Independent normal-limit predictions and derivative-resolution check."""
from __future__ import annotations
import json
from fractions import Fraction
import numpy as np
from v5.population import population
from v5.tracking import ROOT,prepare,stream,atomic_json


def main() -> None:
    path=ROOT/'results/v5/v5_0_influence/summary.json';saved=json.loads(path.read_text())
    config=dict(id='v5_0_limit_prediction',seed=930831,draws=1000000,chunk=20000,
                n_over_m=[0.,1.,32000/14,16000.],derivative_step=.0005)
    out=prepare(config,['src/v5/predict.py','src/v5/population.py','src/v5/tracking.py',
                       'src/v3/population_pair.py','src/provenance.py',str(path.relative_to(ROOT))])
    if (out/'summary.json').exists():return
    h=config['derivative_step'];lower=population(1.,1.-h,1.,80,32);upper=population(1.,1.+h,1.,80,32)
    slope=(upper['score'][1]-lower['score'][1])/(2*h)
    atomic_json(out/'derivative.json',dict(lower=lower,upper=upper,slope=slope,
        difference_from_original=slope-saved['slope'],scope='Step-halving check, not an error certificate'))
    scales=np.sqrt(saved['ratio_constant']*np.array(config['n_over_m']));runs=[]
    for batch,start in enumerate(range(0,config['draws'],config['chunk'])):
        size=min(config['chunk'],config['draws']-start)
        rng=stream(config,batch,'normal');z=rng.normal(size=(size,10));common=rng.normal(size=(size,1))
        counts=[]
        for scale in scales:
            score=z+scale*common;order=np.argsort(-np.abs(score),axis=1)
            positive=np.take_along_axis(score,order,axis=1)>0
            counts.append(np.cumprod(positive,axis=1).sum(axis=1))
        runs.append(np.array(counts,dtype=np.int8).T)
    runs=np.concatenate(runs);fdp=runs/(10+runs)
    np.savez_compressed(out/'individual_runs.npz',run=runs)
    exact=sum(Fraction(k,10+k)*Fraction(1,2**(k+1)) for k in range(10))+Fraction(1,2**11)
    atomic_json(out/'summary.json',dict(config=config,scales=scales.tolist(),
        predicted_fdr=fdp.mean(axis=0).tolist(),mcse=(fdp.std(axis=0,ddof=1)/np.sqrt(len(fdp))).tolist(),
        exact_fair_sign_fdr=float(exact),exact_fraction=str(exact),
        scope='Gaussian local-in-estimation limit. Monte Carlo errors condition on the calibrated ratio constant.'))


if __name__=='__main__':main()
