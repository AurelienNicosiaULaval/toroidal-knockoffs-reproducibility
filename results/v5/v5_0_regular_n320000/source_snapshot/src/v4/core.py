"""Small tested core for v4: exact generators, scores, and batch thresholds."""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve
from scipy.special import i0e,i1e
from scipy.optimize import brentq
from toroidal import vec


def ridge_scores(x:np.ndarray,xt:np.ndarray,y:np.ndarray,penalty:float=5.)->np.ndarray:
    n,p=x.shape
    f=np.concatenate((vec(x).reshape(n,2*p),vec(xt).reshape(n,2*p)),axis=1)
    f-=f.mean(axis=0);yc=y-y.mean()
    coefficient=solve(f.T@f+penalty*np.eye(4*p),f.T@yc,assume_a='pos',check_finite=False)
    return np.linalg.norm(coefficient[:2*p].reshape(p,2),axis=1)-np.linalg.norm(coefficient[2*p:].reshape(p,2),axis=1)


def batch_metrics(scores:np.ndarray,active:np.ndarray,q:float=.1)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Exact knockoff+ rule, including tied magnitudes and all-zero rows."""
    w=np.asarray(scores,float);truth=np.broadcast_to(np.asarray(active,bool),w.shape)
    order=np.argsort(-np.abs(w),axis=1,kind='stable');sorted_w=np.take_along_axis(w,order,axis=1)
    absolute=np.abs(sorted_w);pos=sorted_w>0;neg=sorted_w<0
    positives=pos.cumsum(axis=1);negatives=neg.cumsum(axis=1)
    ends=np.c_[absolute[:,:-1]>absolute[:,1:],np.ones(len(w),bool)]
    valid=((1+negatives)/np.maximum(1,positives)<=q)&(absolute>0)&ends
    k=np.max(np.where(valid,np.arange(w.shape[1]),-1),axis=1);safe=np.maximum(k,0)
    true_pos=(pos&np.take_along_axis(truth,order,axis=1)).cumsum(axis=1)
    rows=np.arange(len(w));count=np.where(k>=0,positives[rows,safe],0);true=np.where(k>=0,true_pos[rows,safe],0)
    fdp=(count-true)/np.maximum(1,count);power=true/np.maximum(1,truth.sum(axis=1))
    return fdp,power,count==0


def sample_pairs(rng:np.random.Generator,n:int,pairs:int,gamma:float,kappa:float)->np.ndarray:
    x2=rng.vonmises(0,kappa,(n,pairs));u=rng.vonmises(0,gamma,(n,pairs))
    x=np.empty((n,2*pairs));x[:,0::2]=(x2+u+np.pi)%(2*np.pi)-np.pi;x[:,1::2]=x2
    return x


def pair_knockoffs(rng:np.random.Generator,x:np.ndarray,gamma:float,kappa:float)->np.ndarray:
    if gamma<0:raise ValueError('Nonnegative interaction required')
    v=vec(x);field=gamma*(v[:,0::2]+v[:,1::2])+np.sqrt(gamma)*rng.normal(size=(len(x),x.shape[1]//2,2))
    full=np.empty_like(v);full[:,0::2]=field;full[:,1::2]=field+np.array([kappa,0.])
    return rng.vonmises(np.arctan2(full[:,:,1],full[:,:,0]),np.linalg.norm(full,axis=2))


def fit_common_gamma(x:np.ndarray)->float:
    mean=float(np.cos(x[:,0::2]-x[:,1::2]).mean())
    if mean<=0:return 0.
    upper=max(2.,1/(1-mean))
    return float(brentq(lambda g:i1e(g)/i0e(g)-mean,0,upper,xtol=1e-13))


def signal_prior(rng:np.random.Generator,rows:int,p:int,count:int,amplitude:float)->tuple[np.ndarray,np.ndarray]:
    if not 1<=count<=p:raise ValueError('Nonempty support required')
    order=np.argsort(rng.uniform(size=(rows,p)),axis=1);active=np.zeros((rows,p),bool)
    active[np.arange(rows)[:,None],order[:,:count]]=True
    angle=rng.uniform(-np.pi,np.pi,(rows,p));b=amplitude*np.stack((np.cos(angle),np.sin(angle)),axis=2)
    return b*active[:,:,None],active
