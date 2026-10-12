"""Exact edge-wise knockoffs for known first-harmonic pairwise circular laws.

p(theta) proportional exp(sum h_j^T v_j + sum_edges v_j^T M_jk v_k).
Independent subjects; one atomic feature per angle, not per sine/cosine.
Exact here refers to the probabilistic construction, not floating point arithmetic.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve
from scipy.special import i0e, i1e
from scipy.optimize import brentq


def vec(x):
    return np.stack([np.cos(x), np.sin(x)], axis=-1)


def wrap(x):
    return (x+np.pi)%(2*np.pi)-np.pi


def rot(a):
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])


@dataclass
class Model:
    h: np.ndarray
    edges: np.ndarray
    M: np.ndarray

    def __post_init__(self):
        self.h=np.asarray(self.h,dtype=float)
        self.edges=np.asarray(self.edges,dtype=int).reshape(-1,2)
        self.M=np.asarray(self.M,dtype=float).reshape(-1,2,2)
        if self.h.ndim!=2 or self.h.shape[1]!=2 or len(self.M)!=len(self.edges):
            raise ValueError('Wrong model dimensions')
        if len(self.edges) and (self.edges.min()<0 or self.edges.max()>=len(self.h)):
            raise ValueError('Invalid graph indices')
        if any(j>=k for j,k in self.edges) or len(set(map(tuple,self.edges)))!=len(self.edges):
            raise ValueError('Unique, ascending, undirected edges required')
        if not np.all(np.isfinite(self.h)) or not np.all(np.isfinite(self.M)):
            raise ValueError('Nonfinite model parameter')

    @property
    def p(self): return len(self.h)

    def fields(self,x):
        v=vec(x); f=np.broadcast_to(self.h,v.shape).copy()
        for (j,k),M in zip(self.edges,self.M):
            f[:,j]+=v[:,k]@M.T
            f[:,k]+=v[:,j]@M
        return f

    def log_unnormalized(self,x):
        v=vec(x); out=np.einsum('npj,pj->n',v,self.h)
        for (j,k),M in zip(self.edges,self.M):
            out+=np.einsum('ni,ij,nj->n',v[:,j],M,v[:,k])
        return out

    def matrix(self):
        J=np.zeros((2*self.p,2*self.p))
        for (j,k),M in zip(self.edges,self.M):
            J[2*j:2*j+2,2*k:2*k+2]=M
            J[2*k:2*k+2,2*j:2*j+2]=M.T
        return J


class EdgeSampler:
    """Block factorization costs O(|E|), sampling O(n(p+|E|)).

    Per edge A_e = [[s I,M],[M^T,s I]], s=||M||_op.
    A_e is PSD; any zero eigenvalues are intentionally retained as zero.
    Node-wise scalar self-terms are constant on the circle.
    """
    def __init__(self,model: Model):
        self.model=model; self.blocks=[]
        for (j,k),M in zip(model.edges,model.M):
            s=float(np.linalg.svd(M,compute_uv=False)[0])
            if s==0: continue
            A=np.block([[s*np.eye(2),M],[M.T,s*np.eye(2)]])
            lam,U=np.linalg.eigh(A)
            if lam.min() < -1e-10*max(1,s): raise ArithmeticError('Unexpected non-PSD block')
            lam=np.maximum(lam,0)
            L=U*np.sqrt(lam)
            self.blocks.append((j,k,L[:2],L[2:]))

    def sample(self,rng,x):
        x=np.asarray(x); v=vec(x)
        if x.ndim!=2 or x.shape[1]!=self.model.p: raise ValueError('Wrong input dimension')
        f=np.broadcast_to(self.model.h,v.shape).copy()
        for j,k,Bj,Bk in self.blocks:
            z=v[:,j]@Bj+v[:,k]@Bk+rng.normal(size=(len(x),Bj.shape[1]))
            f[:,j]+=z@Bj.T; f[:,k]+=z@Bk.T
        return rng.vonmises(np.arctan2(f[:,:,1],f[:,:,0]),np.linalg.norm(f,axis=2))


class SpectralSampler:
    """Earlier global, scalar diagonal loading; PSD boundary allowed."""
    def __init__(self,model:Model,factor=1.01):
        self.model=model; J=model.matrix();lam,U=np.linalg.eigh(J)
        self.d=max(0.,-lam.min())*factor
        self.A=J+self.d*np.eye(2*model.p)
        self.L=U*np.sqrt(np.maximum(lam+self.d,0))

    def sample(self,rng,x):
        n,p=x.shape;v=vec(x).reshape(n,2*p)
        f=(v@self.A+rng.normal(size=(n,2*p))@self.L.T+self.model.h.ravel()).reshape(n,p,2)
        return rng.vonmises(np.arctan2(f[:,:,1],f[:,:,0]),np.linalg.norm(f,axis=2))


def make_tree(kind,p=40,gamma=1.):
    if kind=='chain': parents=np.arange(p-1)
    elif kind=='star': parents=np.zeros(p-1,dtype=int)
    elif kind=='binary': parents=(np.arange(1,p)-1)//2
    else: raise ValueError('Unknown tree')
    edges=np.column_stack([parents,np.arange(1,p)])
    return Model(np.zeros((p,2)),edges,np.tile(gamma*np.eye(2),(p-1,1,1)))


def sample_tree(rng,model,n):
    """Exact iid sampling when h=0, tree rooted at 0, M=gamma R(phase)."""
    if not np.all(model.h==0) or len(model.edges)!=model.p-1: raise ValueError('Not a zero-field tree')
    x=np.empty((n,model.p));x[:,0]=rng.uniform(-np.pi,np.pi,n)
    for (j,k),M in zip(model.edges,model.M):
        gamma=float(np.linalg.norm(M[:,0]));phase=np.arctan2(M[1,0],M[0,0])
        if not np.allclose(M,gamma*rot(phase)):raise ValueError('Not a phase-isotropic edge')
        x[:,k]=wrap(x[:,j]-phase+rng.vonmises(0,gamma,n))
    return x


def inv_bessel_ratio(r):
    r=float(np.clip(r,0.,1-1e-12))
    if r<1e-12:return 0.
    hi=max(2.,1./(1-r))
    return brentq(lambda k:i1e(k)/i0e(k)-r,0.,hi,xtol=1e-12)


def fit_tree(x,edges):
    """Known tree, h=0. Independent edge increments give closed separable MLE.
    Fits both phase and concentration, NOT graph selection or node fields.
    """
    delta=x[:,edges[:,0]]-x[:,edges[:,1]]
    c=np.exp(1j*delta).mean(0); r=abs(c)
    M=np.stack([inv_bessel_ratio(a)*rot(b) for a,b in zip(r,np.angle(c))])
    return Model(np.zeros((x.shape[1],2)),edges.copy(),M)


def knockstat(x,xt,y,penalty=5.):
    n,p=x.shape;F=np.column_stack([vec(x).reshape(n,-1),vec(xt).reshape(n,-1)])
    F-=F.mean(0); y=np.asarray(y); yc=y-y.mean(axis=0)
    coef=solve(F.T@F+penalty*np.eye(4*p),F.T@yc,assume_a='pos',check_finite=False)
    if coef.ndim==1:
        return np.linalg.norm(coef[:2*p].reshape(p,2),axis=1)-np.linalg.norm(coef[2*p:].reshape(p,2),axis=1)
    c=coef.reshape(2,p,2,-1)
    return np.linalg.norm(c[0],axis=(1,2))-np.linalg.norm(c[1],axis=(1,2))


def select(w,q=.1):
    for t in np.sort(np.unique(abs(w[w!=0]))):
        if (1+np.sum(w<=-t))/max(1,np.sum(w>=t))<=q:
            return np.flatnonzero(w>=t)
    return np.array([],dtype=int)


def swap_log_ratio(P:Model,Q:Model,x,xt):
    """One-subject one-angle log Radon-Nikodym ratio; truth P needed."""
    return np.sum((vec(x)-vec(xt))*(P.fields(x)-Q.fields(x)),axis=2)


def error_radii(P:Model,Q:Model):
    """Conservative per-node uniform bound on the natural-field error."""
    delta=np.linalg.norm(P.h-Q.h,axis=1)
    pm={tuple(e):m for e,m in zip(P.edges,P.M)};qm={tuple(e):m for e,m in zip(Q.edges,Q.M)}
    for e in pm.keys()|qm.keys():
        err=np.linalg.norm(pm.get(e,np.zeros((2,2)))-qm.get(e,np.zeros((2,2))),2)
        delta[list(e)]+=err
    return delta
