"""Finite-sample local-alternative comparisons, with exact known covariate laws."""
from __future__ import annotations
import argparse,json,time,traceback,resource
from pathlib import Path
import numpy as np
from toroidal import make_tree,sample_tree,vec,select
from loading import AllocatedSampler,endpoint_allocations
from v4.core import signal_prior,ridge_scores
from v4.calibrate_coupling import weights
from v4.tracking import ROOT,prepare,atomic_json,stream

SOURCES=['src/v4/run_local.py','src/v4/calibrate_coupling.py','src/v4/core.py','src/v4/tracking.py',
         'src/toroidal.py','src/loading.py','src/provenance.py']


def run(c:dict)->None:
    calibration=ROOT/'results/v4'/c['calibration_id']/'summary.json'
    saved=json.loads(calibration.read_text());chosen=saved['chosen_alpha']
    out=prepare(c,SOURCES+[str(calibration.relative_to(ROOT))]);start=time.perf_counter()
    P=make_tree(c['graph'],c['p'],c['gamma']);u=weights(c['graph'],c['p'])
    samplers=[AllocatedSampler(P,endpoint_allocations(P,u**alpha)) for alpha in c['alphas']]
    for rep in range(c['repetitions']):
        file=out/f'records/{rep:06d}.json'
        if file.exists():continue
        try:
            t=time.perf_counter();x=sample_tree(stream(c,rep,'original'),P,c['n']);original_seconds=time.perf_counter()-t
            b,truth=signal_prior(stream(c,rep,'prior'),1,P.p,c['signals'],c['amplitude']);b=b[0];active=np.flatnonzero(truth[0])
            y=np.sum(vec(x)*b,axis=(1,2))/np.sqrt(c['n'])+stream(c,rep,'noise').normal(0,c['noise_sd'],c['n'])
            rows=[]
            for alpha,sampler in zip(c['alphas'],samplers):
                t=time.perf_counter();xt=sampler.sample(stream(c,rep,'knockoff_common'),x);sampling_seconds=time.perf_counter()-t
                t=time.perf_counter();w=ridge_scores(x,xt,y);selected=select(w,c['q']);score_seconds=time.perf_counter()-t
                tp=int(np.isin(selected,active).sum())
                rows.append(dict(alpha=alpha,tuned=alpha==chosen,selected=selected.tolist(),scores=w.tolist(),
                    fdp=(len(selected)-tp)/max(1,len(selected)),power=tp/len(active),empty=not len(selected),
                    sampling_seconds=sampling_seconds,score_seconds=score_seconds))
            record=dict(rep=rep,status='ok',active=active.tolist(),local_coefficients=b.tolist(),original_seconds=original_seconds,methods=rows)
        except Exception as exc:record=dict(rep=rep,status='failure',error=str(exc),traceback=traceback.format_exc())
        atomic_json(file,record)
        if (rep+1)%100==0 or rep+1==c['repetitions']:print(c['id'],rep+1,round(time.perf_counter()-start,2),flush=True)
    records=[json.loads(f.read_text()) for f in (out/'records').glob('*.json')]
    atomic_json(out/'execution.json',dict(completed=len(records),successful=sum(r['status']=='ok' for r in records),
        failures=sum(r['status']!='ok' for r in records),chosen_alpha=chosen,
        seconds_this_invocation=time.perf_counter()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('configuration');args=parser.parse_args();run(json.loads(Path(args.configuration).read_text()))
