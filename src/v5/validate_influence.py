"""Independent Monte-Carlo check of the derivative by a likelihood-score identity."""
from __future__ import annotations
import json
import numpy as np
from scipy.special import i0e,i1e
from toroidal import vec
from v4.core import sample_pairs,pair_knockoffs
from v5.tracking import ROOT,prepare,atomic_json,stream


def main() -> None:
    oracle_path=ROOT/'results/v5/v5_0_influence/population_80_32_+0.json'
    summary_path=ROOT/'results/v5/v5_0_influence/summary.json'
    oracle=json.loads(oracle_path.read_text());summary=json.loads(summary_path.read_text())
    c=dict(id='v5_0_influence_identity',seed=930832,batches=20,rows=50000,gamma=1.,kappa=1.)
    out=prepare(c,['src/v5/validate_influence.py','src/v5/tracking.py','src/v4/core.py',
                  'src/toroidal.py','src/provenance.py',str(oracle_path.relative_to(ROOT)),str(summary_path.relative_to(ROOT))])
    if (out/'summary.json').exists():return
    covariance=np.array(oracle['covariance']);coefficient=np.array(oracle['coefficient']);mean=np.array(oracle['mean'])
    contrast=np.array([0.,0.,-1.,0.,0.,0.,1.,0.]);precision=np.linalg.solve(covariance,contrast)
    A=i1e(1.)/i0e(1.);estimates=[]
    for batch in range(c['batches']):
        x=sample_pairs(stream(c,batch,'original'),c['rows'],1,1.,1.)
        xt=pair_knockoffs(stream(c,batch,'witness'),x,1.,1.)
        d=np.c_[vec(x).reshape(len(x),4),vec(xt).reshape(len(x),4)]-mean
        residual=np.cos(2*x[:,0])-oracle['y_mean']-d@coefficient
        psi=d@precision;score=np.cos(x[:,0]-x[:,1])-A
        values=-psi*residual*score
        estimates.append(float(values.mean()))
        atomic_json(out/f'batches/{batch:03d}.json',dict(batch=batch,mean=float(values.mean()),
            variance=float(values.var(ddof=1)),rows=len(values)))
    value=float(np.mean(estimates));se=float(np.std(estimates,ddof=1)/np.sqrt(len(estimates)))
    atomic_json(out/'summary.json',dict(derivative=value,mcse=se,quadrature_derivative=summary['slope'],
        standardized_difference=(value-summary['slope'])/se,identity='a = -E[psi r (cos U - A(g))]',
        scope='Independent stochastic check, not a certified sign bound'))


if __name__=='__main__':main()
