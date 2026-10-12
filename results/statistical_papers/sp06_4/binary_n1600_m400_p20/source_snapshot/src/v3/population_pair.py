"""Independent angular/Gauss-Hermite quadrature for a two-node model.
No simulated selections: returns population least-squares targets.
"""
from __future__ import annotations
import numpy as np
from scipy.special import i0e, i1e
from numpy.polynomial.hermite import hermgauss


def vm_moments(eta):
    r=np.linalg.norm(eta,axis=-1);u=np.divide(eta,r[...,None],out=np.zeros_like(eta),where=r[...,None]>1e-14)
    a=i1e(r)/i0e(r);t=np.divide(a,r,out=np.full_like(r,.5),where=r>1e-14)
    radial=1-t-a*a;uu=u[..., :,None]*u[...,None,:]
    mean=a[...,None]*u
    covariance=t[...,None,None]*np.eye(2)+(radial-t)[...,None,None]*uu
    return mean,covariance


def population(gamma_true=1.,gamma_work=1.,h1=.8,h2=.3,grid=64,hermite=20):
    angles=np.arange(grid)*2*np.pi/grid
    x=np.stack(np.meshgrid(angles,angles,indexing='ij'),axis=-1).reshape(-1,2)
    v=np.stack([np.cos(x),np.sin(x)],axis=-1)
    energy=h1*v[:,0,0]+h2*v[:,1,0]+gamma_true*np.cos(x[:,0]-x[:,1])
    weights=np.exp(energy-energy.max());weights/=weights.sum()
    y=np.cos(2*x[:,0]);f=v.reshape(-1,4)
    nodes,w=hermgauss(hermite);g=np.stack(np.meshgrid(nodes,nodes,indexing='ij'),axis=-1).reshape(-1,2)*np.sqrt(2)
    wg=np.outer(w,w).ravel()/np.pi
    mean=np.zeros(8);second=np.zeros((8,8));crossy=np.zeros(8)
    # Integrate conditionally on X, summing the latent grid in batches.
    for start in range(0,len(x),128):
        vv=v[start:start+128];ff=f[start:start+128];ww=weights[start:start+128];yy=y[start:start+128]
        field=gamma_work*(vv[:,0]+vv[:,1])[:,None,:]+np.sqrt(gamma_work)*g[None,:,:]
        eta=np.stack([field+np.array([h1,0]),field+np.array([h2,0])],axis=2)
        mu,cov=vm_moments(eta);mt=mu.reshape(len(vv),len(g),4)
        em=np.einsum('g,ngk->nk',wg,mt)
        ett=np.einsum('n,g,ngi,ngj->ij',ww,wg,mt,mt)
        for j in range(2):ett[2*j:2*j+2,2*j:2*j+2]+=np.einsum('n,g,ngab->ab',ww,wg,cov[:,:,j])
        full=np.c_[ff,em];mean+=ww@full;crossy+=(ww*yy)@full
        second[:4,:4]+=(ff*ww[:,None]).T@ff
        cr=(ff*ww[:,None]).T@em;second[:4,4:]+=cr;second[4:,:4]+=cr.T;second[4:,4:]+=ett
    covariance=second-np.outer(mean,mean);coef=np.linalg.solve(covariance,crossy-mean*(weights@y))
    norms=np.linalg.norm(coef.reshape(4,2),axis=1);score=norms[:2]-norms[2:]
    return dict(gamma_true=gamma_true,gamma_work=gamma_work,grid=grid,hermite=hermite,
                coef=coef.tolist(),score=score.tolist(),minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(covariance)[0]))


if __name__=='__main__':
    import json,time
    from pathlib import Path
    rows=[];t=time.perf_counter()
    for size,gh in [(48,16),(64,24),(96,40)]:
        for g in [.95,1.,1.05]:
            r=population(gamma_work=g,grid=size,hermite=gh);rows.append(r);print(g,size,gh,r['score'],flush=True)
    Path('results/v3/validation/nonlinear_population_grid.json').write_text(json.dumps(dict(records=rows,seconds=time.perf_counter()-t),indent=2)+'\n')
