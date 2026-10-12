"""Covariate-only calibration of the first-coefficient variance cost of an extra harmonic."""
from __future__ import annotations
import json
import time
import numpy as np
from v4.core import sample_pairs,pair_knockoffs
from v5.harmonic_score import features
from v5.tracking import ROOT,prepare,atomic_json,stream


def main() -> None:
    config=dict(id='v5_0_harmonic_variance',seed=930830,batches=20,rows=50000,gamma=1.,kappa=1.)
    out=prepare(config,['src/v5/calibrate_risk.py','src/v5/harmonic_score.py','src/v5/tracking.py',
                         'src/v4/core.py','src/toroidal.py','src/provenance.py'])
    if (out/'summary.json').exists():return
    start=time.perf_counter();sums=[];grams=[]
    for batch in range(config['batches']):
        x=sample_pairs(stream(config,batch,'original'),config['rows'],1,1.,1.)
        xt=pair_knockoffs(stream(config,batch,'knockoff'),x,1.,1.)
        d=np.c_[features(x),features(xt)]
        sums.append(d.sum(axis=0));grams.append(d.T@d)
    sums=np.array(sums);grams=np.array(grams)
    first=np.array([0,1,4,5,8,9,12,13])
    def calculate(use: np.ndarray) -> tuple[np.ndarray,np.ndarray,float]:
        n=config['rows']*len(use);mean=sums[use].sum(axis=0)/n
        covariance=grams[use].sum(axis=0)/n-np.outer(mean,mean)
        restricted=np.linalg.inv(covariance[np.ix_(first,first)])
        extended=np.linalg.inv(covariance)[np.ix_(first,first)]
        minimum=float(np.linalg.eigvalsh(extended-restricted).min())
        return np.diag(extended)/np.diag(restricted),covariance,minimum
    ratios,covariance,minimum=calculate(np.arange(len(sums)))
    jack=np.array([calculate(np.delete(np.arange(len(sums)),j))[0] for j in range(len(sums))])
    se=np.sqrt((len(sums)-1)/len(sums)*np.sum((jack-jack.mean(axis=0))**2,axis=0))
    np.savez_compressed(out/'batch_moments.npz',sum=sums,gram=grams,covariance=covariance)
    atomic_json(out/'summary.json',dict(config=config,variance_ratios=ratios.tolist(),mcse=se.tolist(),
        order=['original1_cos','original1_sin','original2_cos','original2_sin',
               'witness1_cos','witness1_sin','witness2_cos','witness2_sin'],
        minimum_variance_difference_eigenvalue=minimum,seconds=time.perf_counter()-start,
        scope='Asymptotic coefficient variances under a correctly specified first-harmonic mean, not power ratios.'))


if __name__=='__main__':main()
