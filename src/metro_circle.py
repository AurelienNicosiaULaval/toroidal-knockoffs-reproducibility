"""Circular random-walk Metro, a specialized implementation of Bates et al.

This is not a replacement claim for all Metro implementations. Symmetric von
Mises random-walk proposals, fixed response-free concentration, a fixed min-fill
elimination order, and damping <1 are used. Rejected proposals are retained as
auxiliary variables in every later acceptance calculation. Omitting their
probabilities is NOT a valid knockoff algorithm.

The implementation is written for this project from the recursion in Bates,
Candes, Janson and Wang (2021), Appendix A. Acceptance probabilities are kept in
log space without clipping away from zero. An exhaustive discrete-state test
checks the recursion independently of Monte-Carlo calibration.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from toroidal import Model, vec, wrap


def elimination_order(p: int, edges: np.ndarray):
    """Deterministic min-fill; tie-break by degree then vertex index."""
    adj = [set() for _ in range(p)]
    for j, k in edges:
        adj[int(j)].add(int(k)); adj[int(k)].add(int(j))
    left = set(range(p)); order=[]; fronts=[]
    while left:
        def score(j):
            ns=sorted(adj[j] & left)
            fill=sum(v not in adj[u] for u,v in combinations(ns,2))
            return fill,len(ns),j
        j=min(left,key=score); ns=sorted(adj[j] & left)
        order.append(j); fronts.append(ns)
        for u,v in combinations(ns,2):
            adj[u].add(v);adj[v].add(u)
        left.remove(j)
        for u in ns:adj[u].discard(j)
    inverse=np.argsort(order)
    fronts=[[int(inverse[k]) for k in ns] for ns in fronts]
    return np.asarray(order), fronts


class CircularMetro:
    def __init__(self, model: Model, tau: float=1., damping:float=.99,
                 max_frontier:int=16, order=None):
        if not np.isfinite(tau) or tau<0:raise ValueError('tau must be nonnegative')
        if not 0<damping<1:raise ValueError('Use damping strictly between zero and one')
        self.original_model=model; self.tau=tau;self.damping=damping
        p=model.p
        if order is None:self.order,self.front=elimination_order(p,model.edges)
        else:
            # Deliberately use a full frontier for externally specified orders.
            self.order=np.asarray(order,dtype=int)
            if sorted(self.order.tolist())!=list(range(p)):raise ValueError('Invalid order')
            self.front=[list(range(j+1,p)) for j in range(p)]
        self.inv=np.argsort(self.order); self.width=max(map(len,self.front),default=0)
        if self.width>max_frontier:
            raise ValueError(f'Induced frontier {self.width} exceeds resource cap {max_frontier}')
        edges=[]; mats=[]
        for (j,k),M in zip(model.edges,model.M):
            a,b=int(self.inv[j]),int(self.inv[k])
            edges.append((min(a,b),max(a,b)));mats.append(M if a<b else M.T)
        self.model=Model(model.h[self.order],np.asarray(edges).reshape(-1,2),np.asarray(mats).reshape(-1,2,2))
        self.neighbors=[[] for _ in range(p)]
        for (j,k),M in zip(self.model.edges,self.model.M):
            self.neighbors[j].append((k,M));self.neighbors[k].append((j,M.T))
        self.fmask=[sum(1<<k for k in f) for f in self.front]
        self.affected=[[k for k in range(j) if j in self.front[k]] for j in range(p)]

    def sample(self,rng:np.random.Generator,x:np.ndarray,proposals=None,forced=None):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=self.model.p:raise ValueError('Wrong X shape')
        X=x[:,self.order]; n,p=X.shape
        P=wrap(X+rng.vonmises(0,self.tau,size=X.shape)) if proposals is None else np.asarray(proposals)[:,self.order]
        V=vec(X); VP=vec(P); cache=[{} for _ in range(p)]
        decisions=np.zeros((n,p),bool); log_actual=np.zeros((n,p)); queries=0
        def loga(j, flags):
            nonlocal queries
            key=flags & self.fmask[j]
            if key in cache[j]:return cache[j][key]
            queries+=1
            field=np.broadcast_to(self.model.h[j],(n,2)).copy()
            for k,M in self.neighbors[j]:
                vk=VP[:,k] if flags & (1<<k) else V[:,k]
                field += vk@M.T
            ratio=np.sum((VP[:,j]-V[:,j])*field,axis=1)
            for k in self.affected[j]:
                a1=loga(k,flags | (1<<j));a0=loga(k,flags)
                f1=np.where(decisions[:,k],a1,np.log1p(-np.exp(a1)))
                f0=np.where(decisions[:,k],a0,np.log1p(-np.exp(a0)))
                ratio+=f1-f0
            out=np.log(self.damping)+np.minimum(ratio,0.)
            if not np.all(np.isfinite(out)):raise FloatingPointError('Nonfinite Metro log probability')
            cache[j][key]=out
            return out
        for j in range(p):
            a=loga(j,0);log_actual[:,j]=a
            decisions[:,j]=(np.log(rng.uniform(size=n))<a) if forced is None else np.asarray(forced)[:,self.order[j]]
        xt=np.where(decisions,P,X)[:,self.inv]
        self.last_diagnostics=dict(width=self.width,queries=queries,acceptance=float(decisions.mean()),
            min_log_acceptance=float(log_actual.min()))
        branch=np.sum(np.where(decisions,log_actual,np.log1p(-np.exp(log_actual))),axis=1)
        return xt,branch
