"""Frozen paired experiment comparing shared and separate regular pilot errors."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import resource
import time
import traceback
from pathlib import Path
import numpy as np
from toroidal import select
from v4.core import sample_pairs,fit_common_gamma
from v5.harmonic_score import features,Moments
from v6.core import fit_separate,witnesses
from v6.tracking import prepare,stream,atomic_json

SOURCES=['src/v6/run_experiment.py','src/v6/core.py','src/v6/tracking.py',
         'src/v4/core.py','src/v5/harmonic_score.py','src/toroidal.py','src/provenance.py']


def run(c:dict,first:int=0,stop:int|None=None) -> None:
    out=prepare(c,SOURCES);stop=c['repetitions'] if stop is None else min(stop,c['repetitions'])
    if not 0<=first<stop:raise ValueError('Invalid execution interval')
    started=time.perf_counter();utc=dt.datetime.now(dt.timezone.utc).isoformat();new=0
    k=c['pairs'];p=2*k;active=np.arange(0,p,2)
    for rep in range(first,stop):
        path=out/f'records/{rep:06d}.json'
        if path.exists():continue
        try:
            clock=time.perf_counter();pilot=sample_pairs(stream(c,rep,'pilot'),c['m'],k,c['gamma'],c['kappa'])
            pooled=fit_common_gamma(pilot);separate=fit_separate(pilot)
            work={'oracle':np.full(k,c['gamma']),'pooled':np.full(k,pooled),
                  'separate':separate,'shared_single':np.full(k,separate[0])}
            fit_seconds=time.perf_counter()-clock
            moments={name:Moments(4*p,1) for name in work}
            rng_x=stream(c,rep,'original');rng_y=stream(c,rep,'noise')
            rng_k={name:stream(c,rep,'knockoff_common') for name in work}
            original_seconds=0.;sampling={name:0. for name in work};gram={name:0. for name in work}
            for offset in range(0,c['n'],c['chunk']):
                count=min(c['chunk'],c['n']-offset);clock=time.perf_counter()
                x=sample_pairs(rng_x,count,k,c['gamma'],c['kappa'])
                y=c['amplitude']*np.cos(2*x[:,active]).sum(axis=1)+rng_y.normal(0,c['noise_sd'],count)
                original=features(x,1);original_seconds+=time.perf_counter()-clock
                for name,g in work.items():
                    clock=time.perf_counter();xt=witnesses(rng_k[name],x,g,c['kappa'])
                    witness=features(xt,1);sampling[name]+=time.perf_counter()-clock
                    clock=time.perf_counter();moments[name].update(np.c_[original,witness],y)
                    gram[name]+=time.perf_counter()-clock
            methods=[]
            for name,g in work.items():
                clock=time.perf_counter();w=moments[name].scores(p,1,available=1,penalty=c['penalty'])[:,0]
                solve_seconds=time.perf_counter()-clock
                chosen=select(w,c['q']);tp=int(np.isin(chosen,active).sum())
                methods.append(dict(method=name,gamma_work=g.tolist(),scores=w.tolist(),selected=chosen.tolist(),
                    fdp=(len(chosen)-tp)/max(1,len(chosen)),power=tp/k,empty=not len(chosen),
                    null_positive_fraction=float(np.mean(w[1::2]>0)),solve_seconds=solve_seconds))
            record=dict(rep=rep,status='ok',gamma_pooled=pooled,gamma_separate=separate.tolist(),methods=methods,
                original_seconds=original_seconds,fit_seconds=fit_seconds,sampling_seconds=sampling,gram_seconds=gram)
        except Exception as exc:
            record=dict(rep=rep,status='failure',error=str(exc),traceback=traceback.format_exc())
        atomic_json(path,record);new+=1
        if (rep+1-first)%50==0 or rep+1==stop:print(c['id'],first,rep+1,round(time.perf_counter()-started,2),flush=True)
    records=[json.loads((out/f'records/{i:06d}.json').read_text()) for i in range(first,stop)]
    execution=out/f'execution_{first:06d}_{stop:06d}.json'
    if new or not execution.exists():
        atomic_json(execution,dict(first=first,stop=stop,newly_executed=new,start_utc=utc,
            end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),seconds=time.perf_counter()-started,
            successful=sum(r['status']=='ok' for r in records),failures=sum(r['status']!='ok' for r in records),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('configuration');parser.add_argument('--first',type=int,default=0);parser.add_argument('--stop',type=int)
    a=parser.parse_args();run(json.loads(Path(a.configuration).read_text()),a.first,a.stop)
