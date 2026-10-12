"""Finite exact-iid checks of shared-pilot error; separate versioned mechanisms."""
from __future__ import annotations
import argparse,json,time,traceback,resource
from pathlib import Path
import numpy as np
from toroidal import select,vec
from v4.core import sample_pairs,pair_knockoffs,fit_common_gamma,ridge_scores
from v4.tracking import ROOT,prepare,atomic_json,stream

SOURCES=['src/v4/run_counterexample.py','src/v4/core.py','src/v4/tracking.py','src/toroidal.py','src/provenance.py']


def run(c:dict)->None:
    out=prepare(c,SOURCES);start=time.perf_counter();n=c['n'];K=c['pairs'];active=np.arange(0,2*K,2)
    for rep in range(c['repetitions']):
        file=out/f'records/{rep:06d}.json'
        if file.exists():continue
        try:
            t=time.perf_counter();x=sample_pairs(stream(c,rep,'original'),n,K,c['gamma'],c['kappa'])
            pilot=sample_pairs(stream(c,rep,'pilot'),n,K,c['gamma'],c['kappa']);original_seconds=time.perf_counter()-t
            y=np.cos(2*x[:,active]).sum(axis=1)+stream(c,rep,'noise').normal(0,c['noise_sd'],n)
            small=int(np.ceil(n**.25));t=time.perf_counter();full=fit_common_gamma(pilot);short=fit_common_gamma(pilot[:small]);fit_seconds=time.perf_counter()-t
            fits={'oracle':c['gamma'],'mle_equal':full,'mle_small':short,'bias_stress':max(0.,full-c['bias_constant']*n**(-.25))}
            rows=[]
            for label,g in fits.items():
                t=time.perf_counter();xt=pair_knockoffs(stream(c,rep,'knockoff_common'),x,g,c['kappa']);sampling_seconds=time.perf_counter()-t
                t=time.perf_counter();w=ridge_scores(x,xt,y);selected=select(w,c['q']);score_seconds=time.perf_counter()-t
                tp=int(np.isin(selected,active).sum());logratio=(c['gamma']-g)*np.sum((vec(x[:,1::2])-vec(xt[:,1::2]))*vec(x[:,0::2]),axis=(0,2))
                rows.append(dict(method=label,gamma_work=g,selected=selected.tolist(),scores=w.tolist(),
                    fdp=(len(selected)-tp)/max(1,len(selected)),power=tp/K,empty=not len(selected),
                    null_positive_fraction=float(np.mean(w[1::2]>0)),max_null_logratio=float(logratio.max()),
                    sampling_seconds=sampling_seconds,score_seconds=score_seconds))
            record=dict(rep=rep,status='ok',m_small=small,gamma_mle_equal=full,gamma_mle_small=short,
                original_and_pilot_seconds=original_seconds,fit_seconds=fit_seconds,methods=rows)
        except Exception as exc:record=dict(rep=rep,status='failure',error=str(exc),traceback=traceback.format_exc())
        atomic_json(file,record)
        if (rep+1)%100==0 or rep+1==c['repetitions']:print(c['id'],rep+1,round(time.perf_counter()-start,2),flush=True)
    records=[json.loads(f.read_text()) for f in (out/'records').glob('*.json')]
    atomic_json(out/'execution.json',dict(completed=len(records),successful=sum(r['status']=='ok' for r in records),
        failures=sum(r['status']!='ok' for r in records),seconds_this_invocation=time.perf_counter()-start,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('configuration');args=parser.parse_args();run(json.loads(Path(args.configuration).read_text()))
