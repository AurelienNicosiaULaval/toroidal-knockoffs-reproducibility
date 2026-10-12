"""Exact Bessel moment formulas, independently checked by angular quadrature.

X2 ~ VM(kappa), X1=X2+U, U~VM(gamma) independently; Y=cos(2 X1).
Working Q is a product law: knockoffs are independent of X and have zero
population regression coefficients. X2 is a conditional null.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.special import ive


def moments(gamma:float,kappa:float):
    a,b,c=ive(np.arange(1,4),gamma)/ive(0,gamma)
    mu,nu,rho=ive(np.arange(1,4),kappa)/ive(0,kappa)
    cov=np.array([[(1+b*nu)/2-a*a*mu*mu,a*((1+nu)/2-mu*mu)],
                  [a*((1+nu)/2-mu*mu),(1+nu)/2-mu*mu]])
    cross=np.array([(a*mu+c*rho)/2-a*mu*b*nu,b*((mu+rho)/2-mu*nu)])
    coefficient=np.linalg.solve(cov,cross)
    return dict(gamma=gamma,kappa=kappa,coefficient=coefficient.tolist(),
        population_score=np.abs(coefficient).tolist(),minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(cov)[0]))


def quadrature(gamma,kappa,grid=512):
    t=np.arange(grid)*2*np.pi/grid;x1,x2=np.meshgrid(t,t,indexing='ij')
    weight=np.exp(kappa*np.cos(x2)+gamma*np.cos(x1-x2));weight=weight.ravel()/weight.sum()
    f=np.c_[np.cos(x1.ravel()),np.cos(x2.ravel())];y=np.cos(2*x1.ravel())
    mean=weight@f;cov=(f*weight[:,None]).T@f-np.outer(mean,mean)
    return np.linalg.solve(cov,(weight*y)@f-mean*(weight@y))


if __name__=='__main__':
    rows=[]
    for gamma,kappa in [(.1,.01),(.5,.5),(1.,1.),(2.,1.)]:
        r=moments(gamma,kappa);direct=quadrature(gamma,kappa)
        r['independent_grid_coefficient']=direct.tolist()
        r['max_difference']=float(np.max(np.abs(direct-r['coefficient'])))
        if r['max_difference']>1e-10:raise AssertionError('Independent moment calculations disagree')
        rows.append(r)
    Path('results/v3/validation/independent_counterexample.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
