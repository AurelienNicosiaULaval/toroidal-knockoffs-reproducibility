"""Paired exact-iid experiments: fixed harmonic scores and an independent regular pilot."""
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
from v4.core import sample_pairs,pair_knockoffs,fit_common_gamma
from v5.harmonic_score import features,Moments
from v5.tracking import ROOT,prepare,stream,atomic_json

SOURCES=['src/v5/run_experiment.py','src/v5/harmonic_score.py','src/v5/tracking.py',
         'src/v4/core.py','src/toroidal.py','src/provenance.py']


def run(c: dict, first: int=0, stop: int|None=None) -> None:
    out=prepare(c,SOURCES)
    stop=c['repetitions'] if stop is None else min(stop,c['repetitions'])
    if not 0<=first<stop:
        raise ValueError('Invalid repetition interval')
    start=time.perf_counter();start_utc=dt.datetime.now(dt.timezone.utc).isoformat()
    active=np.arange(0,2*c['pairs'],2);p=2*c['pairs'];new=0
    for rep in range(first,stop):
        path=out/f'records/{rep:06d}.json'
        if path.exists():
            continue
        try:
            clock=time.perf_counter()
            pilot=sample_pairs(stream(c,rep,'pilot'),c['m'],c['pairs'],c['gamma'],c['kappa'])
            estimated=fit_common_gamma(pilot);fit_seconds=time.perf_counter()-clock
            work={'oracle':c['gamma'],'mle':estimated}
            moments={name:Moments(8*p,len(c['responses'])) for name in work}
            rng_x=stream(c,rep,'original');rng_y=stream(c,rep,'noise')
            rng_k={name:stream(c,rep,'knockoff_common') for name in work}
            original_seconds=0.;sample_seconds={name:0. for name in work}
            gram_seconds={name:0. for name in work}
            for offset in range(0,c['n'],c['chunk']):
                size=min(c['chunk'],c['n']-offset)
                clock=time.perf_counter()
                x=sample_pairs(rng_x,size,c['pairs'],c['gamma'],c['kappa'])
                noise=rng_y.normal(0,c['noise_sd'],size)
                responses=[]
                for response in c['responses']:
                    amplitude=c['amplitude']/np.sqrt(c['n']) if response['local'] else c['amplitude']
                    responses.append(amplitude*np.cos(response['harmonic']*x[:,active]).sum(axis=1)+noise)
                y=np.stack(responses,axis=1);original=features(x)
                original_seconds+=time.perf_counter()-clock
                for label,value in work.items():
                    clock=time.perf_counter();xt=pair_knockoffs(rng_k[label],x,value,c['kappa'])
                    witness=features(xt);sample_seconds[label]+=time.perf_counter()-clock
                    clock=time.perf_counter();moments[label].update(np.c_[original,witness],y)
                    gram_seconds[label]+=time.perf_counter()-clock
            rows=[]
            for label in work:
                for harmonics in [1,2]:
                    clock=time.perf_counter();scores=moments[label].scores(p,harmonics,penalty=c['penalty'])
                    solve_seconds=time.perf_counter()-clock
                    for r,response in enumerate(c['responses']):
                        w=scores[:,r];chosen=select(w,c['q']);tp=int(np.isin(chosen,active).sum())
                        rows.append(dict(kernel=label,harmonics=harmonics,response=response['name'],
                            gamma_work=work[label],scores=w.tolist(),selected=chosen.tolist(),
                            fdp=(len(chosen)-tp)/max(1,len(chosen)),power=tp/len(active),empty=not len(chosen),
                            null_positive_fraction=float(np.mean(w[1::2]>0)),solve_seconds=solve_seconds))
            record=dict(rep=rep,status='ok',gamma_estimated=estimated,methods=rows,fit_seconds=fit_seconds,
                original_seconds=original_seconds,sampling_seconds=sample_seconds,gram_seconds=gram_seconds)
        except Exception as exc:
            record=dict(rep=rep,status='failure',error=str(exc),traceback=traceback.format_exc())
        atomic_json(path,record);new+=1
        if (rep+1-first)%50==0 or rep+1==stop:
            print(c['id'],first,rep+1,round(time.perf_counter()-start,2),flush=True)
    records=[json.loads((out/f'records/{rep:06d}.json').read_text()) for rep in range(first,stop)]
    execution=out/f'execution_{first:06d}_{stop:06d}.json'
    if new or not execution.exists():
        atomic_json(execution,dict(first=first,stop=stop,newly_executed=new,
            start_utc=start_utc,end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            seconds=time.perf_counter()-start,successful=sum(r['status']=='ok' for r in records),
            failures=sum(r['status']!='ok' for r in records),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('configuration')
    parser.add_argument('--first',type=int,default=0);parser.add_argument('--stop',type=int)
    a=parser.parse_args();run(json.loads(Path(a.configuration).read_text()),a.first,a.stop)
