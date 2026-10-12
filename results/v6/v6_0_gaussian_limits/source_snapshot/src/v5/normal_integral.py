"""Deterministic evaluation of the shared-normal leading-run FDR by a two-dimensional integral."""
from __future__ import annotations
import json
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from v5.tracking import ROOT,prepare,atomic_json


def integrate(scale: float, nodes: int, nulls: int=10) -> dict:
    if scale<0 or nodes<2 or nulls!=10:
        raise ValueError('This calculation is registered for ten nulls and ten strong signals')
    z,w=leggauss(nodes);L=10.;M=12.*np.sqrt(1+scale*scale)
    latent=L*z;latent_weight=L*w*np.exp(-latent**2/2)/np.sqrt(2*np.pi)
    magnitude=(z+1)*M/2;magnitude_weight=w*M/2
    mu=scale*latent[:,None];r=magnitude[None,:]
    negative_density=np.exp(-(-r-mu)**2/2)/np.sqrt(2*np.pi)
    upper=ndtr(mu-r);middle=np.maximum(0.,ndtr(r-mu)-ndtr(-r-mu))
    conditional=.5*ndtr(scale*latent)**nulls
    for count in range(1,nulls):
        integrand=negative_density*upper**count*middle**(nulls-1-count)
        conditional+=nulls*math.comb(nulls-1,count)*count/(nulls+count)*(integrand@magnitude_weight)
    return dict(scale=scale,nodes=nodes,fdr=float(latent_weight@conditional),
        truncation_error_upper=float(ndtr(-L)+.5*nulls*ndtr(-M/np.sqrt(1+scale*scale))),
        scope='Analytic tail bound only; quadrature discretization error is not interval-certified')


def main() -> None:
    source=ROOT/'results/v5/v5_0_limit_prediction/summary.json';prediction=json.loads(source.read_text())
    config=dict(id='v5_0_normal_integral',scales=prediction['scales'],nodes=[64,128,256])
    out=prepare(config,['src/v5/normal_integral.py','src/v5/tracking.py','src/provenance.py',str(source.relative_to(ROOT))])
    if (out/'summary.json').exists():return
    rows=[integrate(t,n) for t in config['scales'] for n in config['nodes']]
    atomic_json(out/'summary.json',dict(config=config,records=rows))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
