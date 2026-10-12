"""Fixed harmonic blocks and streaming sufficient statistics for identical ridge fits."""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve


def features(x: np.ndarray, harmonics: int = 2) -> np.ndarray:
    x=np.asarray(x,float)
    if x.ndim!=2 or not np.all(np.isfinite(x)) or harmonics<1:
        raise ValueError('Finite n by p angles and a positive harmonic count required')
    return np.stack([f(k*x) for k in range(1,harmonics+1) for f in (np.cos,np.sin)],axis=2).reshape(len(x),-1)


class Moments:
    """Accumulate centered ridge inputs without retaining all n design rows."""
    def __init__(self, dimension: int, responses: int):
        self.n=0
        self.sum=np.zeros(dimension)
        self.y_sum=np.zeros(responses)
        self.gram=np.zeros((dimension,dimension))
        self.cross=np.zeros((dimension,responses))

    def update(self, design: np.ndarray, y: np.ndarray) -> None:
        if y.ndim==1:
            y=y[:,None]
        if design.shape[0]!=y.shape[0] or design.shape[1]!=len(self.sum):
            raise ValueError('Incompatible design/response shapes')
        self.n+=len(design)
        self.sum+=design.sum(axis=0);self.y_sum+=y.sum(axis=0)
        self.gram+=design.T@design;self.cross+=design.T@y

    def scores(self, p: int, retained: int, available: int=2, penalty: float=5.) -> np.ndarray:
        if not 1<=retained<=available or penalty<0 or self.n==0:
            raise ValueError('Invalid score dimensions or ridge penalty')
        index=np.concatenate([np.arange(j*2*available,j*2*available+2*retained) for j in range(2*p)])
        gram=self.gram[np.ix_(index,index)]-np.outer(self.sum[index],self.sum[index])/self.n
        cross=self.cross[index]-np.outer(self.sum[index],self.y_sum)/self.n
        coefficient=solve(gram+penalty*np.eye(len(index)),cross,assume_a='pos',check_finite=False)
        blocks=coefficient.reshape(2*p,2*retained,-1)
        return np.linalg.norm(blocks[:p],axis=1)-np.linalg.norm(blocks[p:],axis=1)
