"""Vanishing-ridge compatible pseudolikelihood with checked stationarity.

Fixed-dimensional theorem: tau_m > 0, tau_m -> 0; centred CLT when
sqrt(m)*tau_m -> 0. No high-dimensional rate or finite FDR claim.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e,i1e
from fit_model import loss_grad,unpack
from toroidal import Model,vec


def hessian(par: np.ndarray, x: np.ndarray, edges: np.ndarray, penalty: float) -> np.ndarray:
    m,p=x.shape;v=vec(x);model=unpack(par,p,edges);eta=model.fields(x);r=np.linalg.norm(eta,axis=2)
    a=i1e(r)/i0e(r);t=np.divide(a,r,out=np.full_like(r,.5),where=r>1e-10)
    u=np.divide(eta,r[:,:,None],out=np.zeros_like(eta),where=r[:,:,None]>1e-10)
    radial=1-t-a*a
    covariance=t[:,:,None,None]*np.eye(2)+(radial-t)[:,:,None,None]*u[:,:,:,None]*u[:,:,None,:]
    H=penalty*np.eye(len(par))
    for j in range(p):
        indices=[2*j,2*j+1];pieces=[np.broadcast_to(np.eye(2),(m,2,2))]
        for e,(left,right) in enumerate(edges):
            if j not in (left,right):continue
            k=right if j==left else left;v0=v[:,k,0];v1=v[:,k,1];D=np.zeros((m,2,4))
            if j==left:D[:,0,0]=v0;D[:,0,1]=v1;D[:,1,2]=v0;D[:,1,3]=v1
            else:D[:,0,0]=v0;D[:,0,2]=v1;D[:,1,1]=v0;D[:,1,3]=v1
            pieces.append(D);indices.extend(range(2*p+4*e,2*p+4*e+4))
        D=np.concatenate(pieces,axis=2);CD=np.einsum('nab,nbk->nak',covariance[:,j],D)
        block=D.reshape(2*m,-1).T@CD.reshape(2*m,-1)/m
        H[np.ix_(indices,indices)]+=block
    return H


def fit_verified(x: np.ndarray, edges: np.ndarray, penalty: float | None=None) -> tuple[Model,dict]:
    x=np.asarray(x,float);edges=np.asarray(edges,int).reshape(-1,2)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Finite pilot matrix, at least two rows, required')
    m,p=x.shape;tau=float(m**-2 if penalty is None else penalty)
    if not np.isfinite(tau) or tau<=0:raise ValueError('Positive ridge required for this existence theorem')
    Model(np.zeros((p,2)),edges,np.zeros((len(edges),2,2)))
    target=min(1e-8,m**-2);start=time.perf_counter();v=vec(x)
    result=minimize(loss_grad,np.zeros(2*p+4*len(edges)),args=(v,edges,tau),jac=True,
                    method='L-BFGS-B',options={'gtol':target/4,'ftol':1e-15,'maxiter':2000,'maxls':50})
    par=result.x;newton_steps=0
    # Newton polishing removes a fixed stopping-tolerance gap in the asymptotics.
    for _ in range(12):
        value,gradient=loss_grad(par,v,edges,tau)
        if np.linalg.norm(gradient)<=target:break
        H=hessian(par,x,edges,tau);step=np.linalg.solve(H,gradient);rate=1.
        descent=float(gradient@step)
        for _ in range(40):
            trial=par-rate*step;newvalue,_=loss_grad(trial,v,edges,tau)
            if newvalue<=value-1e-4*rate*descent+2e-14*max(1.,abs(value)):break
            rate*=.5
        else:raise ArithmeticError('Newton line search failed')
        par=trial;newton_steps+=1
    value,gradient=loss_grad(par,v,edges,tau);norm=float(np.linalg.norm(gradient))
    if norm>target:raise ArithmeticError(f'Unverified stationarity: {norm} > {target}')
    return unpack(par,p,edges),dict(m=m,penalty=tau,gradient_l2=norm,tolerance=target,
        objective=value,global_objective_gap_upper=norm**2/(2*tau),
        iterations=int(result.nit),newton_steps=newton_steps,seconds=time.perf_counter()-start)
