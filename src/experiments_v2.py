"""Staged v2 experiments. Only saved, successful records enter numeric summaries.

Failures are separately counted and never replaced by favorable replications.
All X rows are exact iid draws. Model fits and tuning never inspect evaluated Y.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
import resource
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.linalg import solve
from toroidal import Model, make_tree, sample_tree, vec, select, swap_log_ratio, error_radii
from experimental_design import design, sample_design, rejection
from fit_model import fit, conditional_nll
from loading import AllocatedSampler, endpoint_allocations, perron_vector, joint_mvr_score
from metro_variants import PeriodicMetro
from metro_circle import elimination_order
from provenance import ROOT, prepare, atomic_json, stream


def get_model(config: dict) -> Model:
    p=config['p'];name=config['graph']
    if name in ('star','chain','binary'):
        return make_tree(name,p,config.get('gamma',1.))
    if name == 'cliques':
        P=design(name,p)
        return Model(P.h*config.get('field_multiplier',1.),P.edges,P.M*config.get('gamma',1.))
    if name == 'blocks3':
        if p%3:raise ValueError('blocks3 requires p divisible by three')
        rng=np.random.default_rng(77691)
        edges=np.array([(b,b+1) for b in range(0,p,3)]+[(b+1,b+2) for b in range(0,p,3)])
        return Model(rng.normal(0,.25,(p,2)),edges,rng.normal(0,.24,(len(edges),2,2)))
    raise ValueError('Unknown design')


def sample_original(rng,config,P,n):
    if config['graph'] in ('star','chain','binary'):
        x=sample_tree(rng,P,n)
    elif config['graph']=='cliques':
        x=sample_design(rng,'cliques',P,n)
    else:
        x=np.empty((n,P.p))
        for b in range(0,P.p,3):
            keep=np.flatnonzero((P.edges[:,0]>=b)&(P.edges[:,1]<b+3))
            sub=Model(P.h[b:b+3],P.edges[keep]-b,P.M[keep])
            x[:,b:b+3]=rejection(rng,sub,n)[0]
    harmonic=config.get('misspecified_harmonic',0.)
    if harmonic:
        # Exact tilt of the first-harmonic law by a second-harmonic node term.
        # Separate independent rejection proposals, never a time series.
        accepted=[];count=0;attempts=0
        plain={**config,'misspecified_harmonic':0.}
        while count<n:
            u=sample_original(rng,plain,P,max(100,3*(n-count)))
            a=np.log(rng.uniform(size=len(u))) < harmonic*np.cos(2*u[:,0])-abs(harmonic)
            accepted.append(u[a]);count+=int(a.sum());attempts+=len(u)
            if attempts>n*10000:raise RuntimeError('Tilt rejection resource limit')
        x=np.concatenate(accepted)[:n]
    return x


def block_score(x,xt,y,harmonics=1,penalty=5.):
    # All harmonics for an angle remain one atomic block.
    n,p=x.shape;k=2*harmonics
    def features(a):return np.concatenate([vec(r*a) for r in range(1,harmonics+1)],axis=2).reshape(n,-1)
    f=np.c_[features(x),features(xt)];f-=f.mean(0);yc=y-y.mean()
    coef=solve(f.T@f+penalty*np.eye(2*p*k),f.T@yc,assume_a='pos',check_finite=False)
    return np.linalg.norm(coef[:p*k].reshape(p,k),axis=1)-np.linalg.norm(coef[p*k:].reshape(p,k),axis=1)


def tune(config,P,out):
    path=out/'tuning.json'
    if path.exists():return json.loads(path.read_text())
    u,_=perron_vector(P);base,_=elimination_order(P.p,P.edges)
    orders={'minfill':base.tolist(),'reverse_minfill':base[::-1].tolist()}
    candidates=[]
    for alpha in [-1.,0.,.5,1.]:
        candidates.append(dict(kind='loading',alpha=alpha))
    for proposal,scale in [('uniform',1.),('vonmises',2.),('wrapped_normal',1.),('antipodal_mixture',2.)]:
        for name,order in orders.items():
            candidates.append(dict(kind='metro',proposal=proposal,scale=scale,order=name,indices=order))
    rows=[]
    for candidate in candidates:
        try:
            sam=(AllocatedSampler(P,endpoint_allocations(P,u**candidate['alpha'])) if candidate['kind']=='loading'
                 else PeriodicMetro(P,candidate['proposal'],candidate['scale'],np.array(candidate['indices']),max_frontier=12))
        except ValueError as exc:
            rows.append(dict(candidate=candidate,status='resource_excluded',reason=str(exc)))
            continue
        scores=[]
        for b in range(2):
            x=sample_original(stream(config,b,'tuning_original'),config,P,config.get('tune_n',1200))
            rng=stream(config,b,'tuning_'+json.dumps(candidate,sort_keys=True))
            xt=sam.sample(rng,x)
            if candidate['kind']=='metro':xt=xt[0]
            scores.append(joint_mvr_score(x,xt))
        rows.append(dict(candidate=candidate,status='ok',score=float(np.mean(scores)),replicate_scores=scores))
    chosen={kind:min([r for r in rows if r['status']=='ok' and r['candidate']['kind']==kind],key=lambda r:r['score'])['candidate'] for kind in ['loading','metro']}
    result=dict(candidates=rows,chosen=chosen,criterion='mean trace inverse of ridge-regularized full pair covariance',
                ridge=.001,independent_covariate_only=True,pilot_replicates=2)
    atomic_json(path,result);return result


def learned_models(config,P,pilot):
    # Fixed graph, fixed containing supergraph, and separately learned graph.
    result={};diagnostics={}
    for spec in config.get('fits',[]):
        name=spec['name'];m=spec['m'];data=pilot[:m];edges=P.edges.copy()
        if spec['graph']=='supergraph':
            edges=np.array([(b+j,b+k) for b in range(0,P.p,3) for j in range(3) for k in range(j+1,3)])
        if spec['graph']=='missing':
            active=set(config['active'])
            crossing=[i for i,(j,k) in enumerate(edges) if (j in active)!=(k in active)]
            edges=np.delete(edges,crossing[0] if crossing else 0,axis=0)
        if spec['graph']=='learned':
            superedges=np.array([(b+j,b+k) for b in range(0,P.p,3) for j in range(3) for k in range(j+1,3)])
            # Within-pilot split makes graph selection explicit, not oracle.
            screen,res=fit(data[:m//2],superedges,penalty=.1)
            edges=superedges[np.linalg.norm(screen.M,axis=(1,2))>.25]
            data=data[m//2:]
        t=time.perf_counter();Q,res=fit(data,edges,penalty=spec['penalty'])
        result[name]=Q
        diagnostics[name]=dict(fit_seconds=time.perf_counter()-t,iterations=int(res.nit),
                               max_gradient=float(np.max(abs(res.jac))),edges=edges.tolist(),m=m,
                               penalty=spec['penalty'],fields=Q.h.tolist(),matrices=Q.M.tolist())
    return result,diagnostics


def run(config,limit=None):
    out=prepare(config);P=get_model(config);t0=time.perf_counter()
    atomic_json(out/'true_model.json',dict(h=P.h.tolist(),edges=P.edges.tolist(),M=P.M.tolist()))
    tuning=tune(config,P,out) if config.get('tuning',False) else None
    u,rho=perron_vector(P)
    active=np.array(config['active'],int);null=np.setdiff1d(np.arange(P.p),active)
    samplers={'symmetric_v2':AllocatedSampler(P),
              'perron_v2':AllocatedSampler(P,endpoint_allocations(P,u)),
              'metro_uniform_v2':PeriodicMetro(P)}
    if tuning:
        samplers['tuned_loading_v2']=AllocatedSampler(P,endpoint_allocations(P,u**tuning['chosen']['loading']['alpha']))
        c=tuning['chosen']['metro']
        samplers['metro_tuned_v2']=PeriodicMetro(P,c['proposal'],c['scale'],np.array(c['indices']))
    target=min(config['repetitions'],limit) if limit else config['repetitions']
    for rep in range(target):
        path=out/f'records/{rep:06d}.json'
        if path.exists():continue
        try:
            start=time.perf_counter();x=sample_original(stream(config,rep,'original'),config,P,config['n'])
            original_time=time.perf_counter()-start
            rng_y=stream(config,rep,'response')
            eta=config.get('beta',1.5)*np.cos(config.get('response_harmonic',1)*x[:,active]).sum(1)
            y=(rng_y.binomial(1,expit(eta)) if config['response']=='binary'
               else eta+rng_y.normal(0,config.get('noise',4.),len(x)))
            models={'oracle':P};fitdiag={};current=dict(samplers)
            if config.get('fits'):
                m=max(s['m'] for s in config['fits'])
                start=time.perf_counter();pilot=sample_original(stream(config,rep,'pilot'),config,P,m)
                pilot_seconds=time.perf_counter()-start
                fitted,fitdiag=learned_models(config,P,pilot);models.update(fitted)
                for name,Q in fitted.items():current[name]=AllocatedSampler(Q)
            else:pilot_seconds=0.
            methods=[]
            for name,sam in current.items():
                rng=stream(config,rep,'knockoff_'+name);start=time.perf_counter()
                xt=sam.sample(rng,x)
                if isinstance(sam,PeriodicMetro):xt=xt[0]
                sample_time=time.perf_counter()-start;start=time.perf_counter()
                w=block_score(x,xt,y,config.get('score_harmonics',1))
                chosen=select(w,config.get('q',.1));true=int(np.isin(chosen,active).sum())
                metric=dict(method=name,fdp=(len(chosen)-true)/max(1,len(chosen)),
                    power=true/len(active) if len(active) else 0.,selected=chosen.tolist(),
                    scores=w.tolist(),empty=int(len(chosen)==0),similarity=float(np.cos(x-xt).mean()),
                    sampling_seconds=sample_time,selection_seconds=time.perf_counter()-start,
                    acceptance=sam.last_diagnostics['acceptance'] if isinstance(sam,PeriodicMetro) else None)
                if name in models and name!='oracle':
                    Q=models[name];err=P.fields(x)-Q.fields(x);ell=swap_log_ratio(P,Q,x,xt)
                    # For harmonic misspecification the identity needs the omitted term too.
                    if config.get('misspecified_harmonic',0.):
                        ell[:,0]+=config['misspecified_harmonic']*(np.cos(2*x[:,0])-np.cos(2*xt[:,0]))
                    metric.update(max_field_mse=float(np.max(np.mean(np.sum(err**2,axis=2),axis=0))),
                                  max_radius=float(error_radii(P,Q).max()),
                                  max_ell_second=float(np.max(np.mean(ell**2,axis=0))),
                                  max_null_column_logratio=float(np.max(ell[:,null].sum(0))) if len(null) else 0.,
                                  conditional_nll=float(conditional_nll(Q,x).mean()))
                methods.append(metric)
            record=dict(rep=rep,status='ok',original_seconds=original_time,pilot_seconds=pilot_seconds,
                        fits=fitdiag,methods=methods,max_rss_native=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        except Exception as exc:
            record=dict(rep=rep,status='failure',error=repr(exc),traceback=traceback.format_exc())
        atomic_json(path,record)
        if (rep+1)%100==0 or rep+1==target:
            print(config['id'],rep+1,'elapsed',round(time.perf_counter()-t0,2),flush=True)
    records=[json.loads(p.read_text()) for p in sorted((out/'records').glob('*.json'))]
    atomic_json(out/'execution.json',dict(target=config['repetitions'],completed=len(records),
                successful=sum(r['status']=='ok' for r in records),failures=sum(r['status']!='ok' for r in records),
                elapsed_this_invocation_seconds=time.perf_counter()-t0,
                rss_units='bytes on macOS; KiB on Linux',perron_radii=rho.tolist()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.limit)
