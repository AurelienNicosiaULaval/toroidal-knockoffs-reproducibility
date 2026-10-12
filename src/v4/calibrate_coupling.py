"""Covariate-only calibration of local-alternative power over five loadings."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.special import i0e,i1e
from toroidal import make_tree,sample_tree
from loading import AllocatedSampler,endpoint_allocations
from v4.core import signal_prior,batch_metrics
from v4.tracking import ROOT,prepare,atomic_json,stream

SOURCES=['src/v4/calibrate_coupling.py','src/v4/core.py','src/v4/tracking.py',
         'src/toroidal.py','src/loading.py','src/provenance.py']


def weights(graph:str,p:int)->np.ndarray:
    if graph=='star':return np.r_[np.sqrt(p-1),np.ones(p-1)]
    if graph=='chain':return np.sin(np.pi*np.minimum(np.arange(1,p+1),np.arange(p,0,-1))/(p+1))
    raise ValueError('Only the registered exact tree designs are supported')


def exact_sigma(graph:str,p:int,gamma:float)->np.ndarray:
    distance=np.abs(np.arange(p)[:,None]-np.arange(p))
    if graph=='star':
        distance=np.full((p,p),2);distance[0,:]=1;distance[:,0]=1;np.fill_diagonal(distance,0)
    a=i1e(gamma)/i0e(gamma)
    return np.kron(a**distance/2,np.eye(2))


def calibrate(c:dict)->None:
    out=prepare(c,SOURCES)
    if (out/'summary.json').exists():print(c['id'],'retained complete');return
    start=time.perf_counter();P=make_tree(c['graph'],c['p'],c['gamma']);p=P.p;u=weights(c['graph'],p)
    sigma=exact_sigma(c['graph'],p,c['gamma']);matrices=[];details=[]
    for alpha in c['alphas']:
        sampler=AllocatedSampler(P,endpoint_allocations(P,u**alpha));draws=[]
        for batch,first in enumerate(range(0,c['covariance_rows'],c['covariance_chunk'])):
            size=min(c['covariance_chunk'],c['covariance_rows']-first)
            x=sample_tree(stream(c,batch,'cal_original'),P,size)
            field=sampler.fields(stream(c,batch,'cal_field'),x);r=np.linalg.norm(field,axis=2)
            values=(1-(i1e(r)/i0e(r))**2)/2
            if c['graph']=='star':values[:,1:]=values[:,1:].mean(axis=1)[:,None]
            else:values=(values+values[:,::-1])/2
            draws.append(values)
        values=np.concatenate(draws);r=values.mean(axis=0);se=values.std(axis=0,ddof=1)/np.sqrt(len(values))
        residual=np.diag(np.repeat(r,2));gamma=np.block([[sigma,sigma-residual],[sigma-residual,sigma]])
        minimum=float(np.linalg.eigvalsh(gamma).min())
        if minimum<=0:raise ArithmeticError('Estimated pair covariance is not positive definite')
        matrices.append(gamma);details.append(dict(alpha=alpha,r=r.tolist(),r_mcse=se.tolist(),minimum_eigenvalue=minimum))
        print(c['id'],'covariance',alpha,round(time.perf_counter()-start,2),flush=True)
    covariance_seconds=time.perf_counter()-start;roots=[np.linalg.cholesky(np.linalg.solve(g,np.eye(4*p))) for g in matrices]
    np.savez_compressed(out/'covariances.npz',gamma=np.array(matrices),sigma=sigma)
    phases={}
    for phase,total in [('training',c['training_draws']),('validation',c['validation_draws'])]:
        t0=time.perf_counter();fdps=[];powers=[];empties=[]
        for batch,first in enumerate(range(0,total,c['normal_chunk'])):
            size=min(c['normal_chunk'],total-first)
            b,active=signal_prior(stream(c,batch,phase+'_prior'),size,p,c['signals'],c['amplitude'])
            z=stream(c,batch,phase+'_normal').normal(size=(size,4*p));f=[];power=[];empty=[]
            for root in roots:
                g=c['noise_sd']*(z@root.T);w=np.linalg.norm(g[:,:2*p].reshape(size,p,2)+b,axis=2)-np.linalg.norm(g[:,2*p:].reshape(size,p,2),axis=2)
                a,d,e=batch_metrics(w,active,c['q']);f.append(a);power.append(d);empty.append(e)
            fdps.append(np.array(f).T);powers.append(np.array(power).T);empties.append(np.array(empty).T)
        f=np.concatenate(fdps);power=np.concatenate(powers);empty=np.concatenate(empties)
        np.savez_compressed(out/f'{phase}_metrics.npz',fdp=f,power=power,empty=empty)
        phases[phase]=dict(draws=total,seconds=time.perf_counter()-t0,fdp=f.mean(axis=0).tolist(),
            fdp_mcse=(f.std(axis=0,ddof=1)/np.sqrt(total)).tolist(),power=power.mean(axis=0).tolist(),
            power_mcse=(power.std(axis=0,ddof=1)/np.sqrt(total)).tolist(),empty=empty.mean(axis=0).tolist())
    chosen=min(range(len(c['alphas'])),key=lambda j:(-phases['training']['power'][j],abs(c['alphas'][j]),c['alphas'][j]))
    validation=np.load(out/'validation_metrics.npz');base=c['alphas'].index(0.)
    diff=validation['power'][:,chosen]-validation['power'][:,base]
    atomic_json(out/'summary.json',dict(config=c,covariance=details,covariance_seconds=covariance_seconds,
        phases=phases,chosen_index=chosen,chosen_alpha=c['alphas'][chosen],
        validation_power_difference_vs_symmetric=float(diff.mean()),paired_mcse=float(diff.std(ddof=1)/np.sqrt(len(diff))),
        conditional_mc_regret_bound_95=2*np.sqrt(np.log(2*len(c['alphas'])/.05)/(2*c['training_draws'])),
        seconds=time.perf_counter()-start))
    print(c['id'],'chosen',c['alphas'][chosen],'gain',diff.mean(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('configuration');args=parser.parse_args()
    calibrate(json.loads(Path(args.configuration).read_text()))
