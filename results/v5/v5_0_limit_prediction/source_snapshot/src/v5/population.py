"""Version-5 quadrature exposing covariance and mean; based on retained v3 integration."""
from __future__ import annotations
import numpy as np
from numpy.polynomial.hermite import hermgauss
from v3.population_pair import vm_moments


def population(gamma: float, work: float, kappa: float, grid: int, hermite: int) -> dict:
    if work < 0:
        raise ValueError('Working interaction must be nonnegative')
    angles=np.arange(grid)*2*np.pi/grid
    x=np.stack(np.meshgrid(angles,angles,indexing='ij'),axis=-1).reshape(-1,2)
    v=np.stack((np.cos(x),np.sin(x)),axis=-1)
    energy=kappa*v[:,1,0]+gamma*np.cos(x[:,0]-x[:,1])
    probability=np.exp(energy-energy.max());probability/=probability.sum()
    y=np.cos(2*x[:,0]);f=v.reshape(-1,4)
    nodes,w=hermgauss(hermite)
    z=np.stack(np.meshgrid(nodes,nodes,indexing='ij'),axis=-1).reshape(-1,2)*np.sqrt(2)
    weights=np.outer(w,w).ravel()/np.pi
    mean=np.zeros(8);second=np.zeros((8,8));cross=np.zeros(8)
    for start in range(0,len(x),128):
        vv=v[start:start+128];ff=f[start:start+128]
        pp=probability[start:start+128];yy=y[start:start+128]
        field=work*(vv[:,0]+vv[:,1])[:,None,:]+np.sqrt(work)*z[None,:,:]
        eta=np.stack((field,field+np.array([kappa,0.])),axis=2)
        mu,cov=vm_moments(eta);mt=mu.reshape(len(vv),len(z),4)
        em=np.einsum('g,ngk->nk',weights,mt)
        ett=np.einsum('n,g,ngi,ngj->ij',pp,weights,mt,mt)
        for j in range(2):
            ett[2*j:2*j+2,2*j:2*j+2]+=np.einsum('n,g,ngab->ab',pp,weights,cov[:,:,j])
        full=np.c_[ff,em];mean+=pp@full;cross+=(pp*yy)@full
        second[:4,:4]+=(ff*pp[:,None]).T@ff
        original_witness=(ff*pp[:,None]).T@em
        second[:4,4:]+=original_witness;second[4:,:4]+=original_witness.T
        second[4:,4:]+=ett
    covariance=second-np.outer(mean,mean);y_mean=float(probability@y)
    coefficient=np.linalg.solve(covariance,cross-mean*y_mean)
    norms=np.linalg.norm(coefficient.reshape(4,2),axis=1)
    return dict(gamma=gamma,work=work,kappa=kappa,grid=grid,hermite=hermite,
                mean=mean.tolist(),covariance=covariance.tolist(),coefficient=coefficient.tolist(),
                y_mean=y_mean,score=(norms[:2]-norms[2:]).tolist())
