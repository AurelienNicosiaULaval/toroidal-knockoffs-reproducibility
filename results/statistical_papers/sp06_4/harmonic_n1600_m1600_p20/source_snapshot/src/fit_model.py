"""Convex joint toroidal pseudolikelihood on a supplied graph.
The l2 penalty is optional. The fixed-dimensional theorem concerns a compact
constraint or a vanishing penalty, not a fixed regularization parameter.
"""
from __future__ import annotations
import numpy as np
from scipy.special import i0e,i1e
from scipy.optimize import minimize
from toroidal import Model,vec


def unpack(par,p,edges):
    return Model(par[:2*p].reshape(p,2),edges,par[2*p:].reshape(-1,2,2))


def loss_grad(par,v,edges,penalty=0.):
    n,p,_=v.shape;h=par[:2*p].reshape(p,2);M=par[2*p:].reshape(-1,2,2)
    eta=np.broadcast_to(h,v.shape).copy()
    for (j,k),mat in zip(edges,M):
        eta[:,j]+=v[:,k]@mat.T;eta[:,k]+=v[:,j]@mat
    r=np.linalg.norm(eta,axis=2);a=i1e(r)/i0e(r)
    loss=np.mean(np.sum(np.log(2*np.pi)+np.log(i0e(r))+r-np.sum(eta*v,axis=2),axis=1))
    scale=np.divide(a,r,out=np.full_like(r,.5),where=r>1e-10)
    g=scale[:,:,None]*eta-v
    gh=g.mean(0).ravel();gm=np.empty((len(edges),2,2))
    for z,(j,k) in enumerate(edges):gm[z]=(g[:,j].T@v[:,k]+v[:,j].T@g[:,k])/n
    grad=np.r_[gh,gm.ravel()]
    return float(loss+.5*penalty*np.dot(par,par)),grad+penalty*par


def fit(x,edges,penalty=0.,start=None,maxiter=1000):
    x=np.asarray(x,float);edges=np.asarray(edges,int).reshape(-1,2);n,p=x.shape
    if not np.all(np.isfinite(x)) or n<2:raise ValueError('Nonfinite or insufficient input')
    v=vec(x);start=np.zeros(2*p+4*len(edges)) if start is None else np.asarray(start,float)
    result=minimize(loss_grad,start,args=(v,edges,penalty),jac=True,method='L-BFGS-B',
        options={'maxiter':maxiter,'ftol':1e-11,'gtol':1e-6,'maxls':40})
    if not result.success:raise RuntimeError(f'Pseudolikelihood failure: {result.message}')
    return unpack(result.x,p,edges),result


def conditional_nll(P,x):
    eta=P.fields(x);v=vec(x);r=np.linalg.norm(eta,axis=2)
    return np.log(2*np.pi)+np.log(i0e(r))+r-np.sum(eta*v,axis=2)
