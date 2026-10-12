"""Pair-specific MLEs and the unchanged augmentation with heterogeneous interactions."""
from __future__ import annotations
import numpy as np
from toroidal import vec
from v4.core import fit_common_gamma


def fit_separate(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float)
    if x.ndim!=2 or len(x)<1 or x.shape[1]%2 or not np.isfinite(x).all():
        raise ValueError('Finite observations arranged in nonempty pairs are required')
    return np.array([fit_common_gamma(x[:,j:j+2]) for j in range(0,x.shape[1],2)])


def witnesses(rng: np.random.Generator,x: np.ndarray,gamma: np.ndarray,kappa: float) -> np.ndarray:
    x=np.asarray(x,float);g=np.asarray(gamma,float)
    if x.ndim!=2 or x.shape[1]%2 or g.shape!=(x.shape[1]//2,):
        raise ValueError('One interaction per pair is required')
    if not np.isfinite(x).all() or not np.isfinite(g).all() or np.any(g<0) or not np.isfinite(kappa):
        raise ValueError('Finite angles, fields and nonnegative interactions required')
    v=vec(x);g=g[None,:,None]
    field=g*(v[:,0::2]+v[:,1::2])+np.sqrt(g)*rng.normal(size=(len(x),x.shape[1]//2,2))
    full=np.empty_like(v);full[:,0::2]=field;full[:,1::2]=field+np.array([kappa,0.])
    return rng.vonmises(np.arctan2(full[:,:,1],full[:,:,0]),np.linalg.norm(full,axis=2))
