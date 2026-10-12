"""Exactly simulated designs used in the manuscript, with fixed graph/parameters."""
import numpy as np
from toroidal import Model,make_tree,sample_tree,wrap


def design(name,p=40):
    if name in ('chain','binary','star'):return make_tree(name,p,1.)
    if name=='cycle':
        P=make_tree('chain',p,1.)
        return Model(P.h,np.vstack([P.edges,[0,p-1]]),np.concatenate([P.M,[np.eye(2)]]))
    if name=='cliques':
        if p%5:raise ValueError('p must be a multiple of 5')
        rng=np.random.default_rng(19590331)
        edges=np.array([(b+j,b+k) for b in range(0,p,5) for j in range(5) for k in range(j+1,5)])
        return Model(rng.normal(0,.12,(p,2)),edges,rng.normal(0,.14,(len(edges),2,2)))
    raise ValueError(name)


def rejection(rng,P,n):
    bound=np.linalg.norm(P.h,axis=1).sum()+sum(np.linalg.norm(M,2) for M in P.M)
    accepted=[];count=0;proposed=0
    while count<n:
        b=max(100,2*(n-count));u=rng.uniform(-np.pi,np.pi,(b,P.p));proposed+=b
        a=np.log(rng.uniform(size=b))<P.log_unnormalized(u)-bound
        accepted.append(u[a]);count+=a.sum()
        if proposed>max(n*50000,10_000_000):raise RuntimeError('Rejection resource limit')
    return np.concatenate(accepted)[:n],proposed


def sample_design(rng,name,P,n):
    if name in ('chain','binary','star'):return sample_tree(rng,P,n)
    if name=='cycle':
        chain=Model(P.h,P.edges[:-1],P.M[:-1]); chunks=[]; count=0
        while count<n:
            u=sample_tree(rng,chain,max(200,3*(n-count)))
            a=np.log(rng.uniform(size=len(u)))<np.cos(u[:,0]-u[:,-1])-1
            chunks.append(u[a]);count+=int(a.sum())
        return np.concatenate(chunks)[:n]
    if name=='cliques':
        out=np.empty((n,P.p))
        for b in range(0,P.p,5):
            ii=np.flatnonzero((P.edges[:,0]>=b)&(P.edges[:,1]<b+5))
            sub=Model(P.h[b:b+5],P.edges[ii]-b,P.M[ii]);out[:,b:b+5]=rejection(rng,sub,n)[0]
        return out
    raise ValueError(name)
