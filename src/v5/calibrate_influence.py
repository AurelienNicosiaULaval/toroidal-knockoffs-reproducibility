"""Estimate the oracle sandwich variance and translate pilot size into shared-score error."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import i0e,i1e
from toroidal import vec
from v4.core import sample_pairs,pair_knockoffs
from v5.population import population
from v5.tracking import ROOT,prepare,atomic_json,stream

SOURCES=['src/v5/calibrate_influence.py','src/v5/population.py','src/v5/tracking.py',
         'src/v3/population_pair.py','src/v4/core.py','src/toroidal.py','src/provenance.py']


def calibrate(c: dict) -> None:
    out=prepare(c,SOURCES)
    if (out/'summary.json').exists():
        print('Retained completed calibration');return
    start=time.perf_counter();quadratures=[]
    for grid,hermite in c['quadratures']:
        results=[]
        for step in [-c['derivative_step'],0.,c['derivative_step']]:
            path=out/f'population_{grid}_{hermite}_{step:+g}.json'
            if path.exists():
                result=json.loads(path.read_text())
            else:
                result=population(c['gamma'],c['gamma']+step,c['kappa'],grid,hermite)
                atomic_json(path,result)
            results.append(result)
        slope=(results[2]['score'][1]-results[0]['score'][1])/(2*c['derivative_step'])
        quadratures.append(dict(grid=grid,hermite=hermite,slope=slope,oracle=results[1]))
        print('Quadrature',grid,hermite,'slope',slope,flush=True)
    target=quadratures[-1]['oracle'];gamma=np.array(target['covariance'])
    coefficient=np.array(target['coefficient']);mean=np.array(target['mean'])
    contrast=np.zeros(8)
    for offset,sign in [(2,1),(6,-1)]:
        block=coefficient[offset:offset+2]
        if np.linalg.norm(block)<1e-8:
            raise ArithmeticError('Null norm derivative is not numerically resolved')
        contrast[offset:offset+2]=sign*block/np.linalg.norm(block)
    precision_contrast=np.linalg.solve(gamma,contrast)
    pieces=[]
    for batch in range(c['batches']):
        path=out/f'batches/{batch:03d}.json'
        if path.exists():
            values=json.loads(path.read_text())
        else:
            x=sample_pairs(stream(c,batch,'original'),c['batch_rows'],1,c['gamma'],c['kappa'])
            xt=pair_knockoffs(stream(c,batch,'knockoff'),x,c['gamma'],c['kappa'])
            design=np.c_[vec(x).reshape(len(x),4),vec(xt).reshape(len(x),4)]-mean
            residual=np.cos(2*x[:,0])-target['y_mean']-design@coefficient
            influence=design@precision_contrast
            values=dict(batch=batch,rows=len(x),residual_mean=float(residual.mean()),
                residual_variance=float(np.mean(residual**2)),
                influence_squared=float(np.mean(influence**2)),
                influence_residual_squared=float(np.mean(influence**2*residual**2)))
            atomic_json(path,values)
        pieces.append(values)
    def tau(rows: list[dict]) -> float:
        mse=np.mean([r['residual_variance'] for r in rows])
        psi2=np.mean([r['influence_squared'] for r in rows])
        weighted=np.mean([r['influence_residual_squared'] for r in rows])
        return float(weighted+((c['pairs']-1)*mse+c['noise_sd']**2)*psi2)
    variance=tau(pieces)
    jackknife=np.array([tau(pieces[:j]+pieces[j+1:]) for j in range(len(pieces))])
    se=float(np.sqrt((len(pieces)-1)/len(pieces)*np.sum((jackknife-jackknife.mean())**2)))
    a=quadratures[-1]['slope'];A=i1e(c['gamma'])/i0e(c['gamma'])
    information=1-A/c['gamma']-A*A;V=1/(c['pairs']*information)
    coefficient_ratio=a*a*V/variance
    atomic_json(out/'summary.json',dict(config=c,quadrature_slopes=[{k:r[k] for k in ['grid','hermite','slope']} for r in quadratures],
        slope=a,pilot_variance=V,tau_squared=variance,tau_squared_mcse=se,
        information=information,ratio_constant=coefficient_ratio,
        ratio_constant_mcse=coefficient_ratio*se/variance,
        exact_influence_second_moment=float(contrast@precision_contrast),
        empirical_influence_second_moment=float(np.mean([r['influence_squared'] for r in pieces])),
        mean_pair_residual_variance=float(np.mean([r['residual_variance'] for r in pieces])),
        scope='Monte Carlo variance error only; quadrature and derivative errors are convergence checks, not certified bounds.',
        seconds=time.perf_counter()-start))
    print('tau_squared',variance,'MCSE',se,'ratio_constant',coefficient_ratio,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('configuration');args=p.parse_args()
    calibrate(json.loads(Path(args.configuration).read_text()))
