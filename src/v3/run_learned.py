"""Fixed-p, learned-law study v3.0; 2 paired responses per exact-iid X dataset."""
from __future__ import annotations
import argparse,hashlib,json,platform,time,traceback
from pathlib import Path
import numpy as np
import scipy
from toroidal import select,swap_log_ratio
from experiments_v2 import get_model,sample_original,block_score
from loading import AllocatedSampler
from provenance import ROOT,atomic_json,digest,stream
from v3.fit_verified import fit_verified

SOURCES=['src/v3/run_learned.py','src/v3/fit_verified.py','src/toroidal.py','src/loading.py',
         'src/experiments_v2.py','src/experimental_design.py','src/fit_model.py','src/provenance.py']


def run(c:dict,limit:int|None=None):
    out=ROOT/'results/v3'/c['id'];out.mkdir(parents=True,exist_ok=True)
    manifest=dict(config=c,code={f:digest(ROOT/f) for f in SOURCES},python=platform.python_version(),
                  numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform())
    signature=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())['signature']!=signature:raise RuntimeError('Version signature changed')
    else:
        atomic_json(out/'manifest.json',dict(**manifest,signature=signature))
        for f in SOURCES:
            path=out/'source_snapshot'/f;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/f).read_bytes())
    P=get_model(c);n=c['n'];p=P.p;active=np.array(c['active']);null=np.setdiff1d(np.arange(p),active)
    superedges=np.array([(b+j,b+k) for b in range(0,p,3) for j in range(3) for k in range(j+1,3)])
    truth=set(map(tuple,P.edges));start=time.perf_counter()
    target=min(c['repetitions'],limit) if limit else c['repetitions']
    for rep in range(target):
        file=out/f'records/{rep:06d}.json'
        if file.exists():continue
        try:
            x=sample_original(stream(c,rep,'original'),c,P,n)
            pilot=sample_original(stream(c,rep,'pilot'),c,P,n)
            noise=stream(c,rep,'response').normal(0,4.,n)
            linear=1.5*np.cos(x[:,active]).sum(1)+noise
            responses={'linear':linear,'nonlinear_omitted':linear+2*np.cos(2*x[:,active]).sum(1)}
            models={'oracle':P};fits={}
            for label,m in [('known_sqrt',int(np.ceil(np.sqrt(n)))),('known_equal',n)]:
                Q,diagnostic=fit_verified(pilot[:m],P.edges);models[label]=Q
                fits[label]=dict(**diagnostic,h=Q.h.tolist(),M=Q.M.tolist(),edges=Q.edges.tolist())
            t=time.perf_counter();half=n//2;screen,ds=fit_verified(pilot[:half],superedges)
            threshold=half**(-.25);edges=superedges[np.linalg.norm(screen.M,axis=(1,2))>threshold]
            Q,df=fit_verified(pilot[half:],edges);models['learned_equal']=Q
            chosen=set(map(tuple,edges));fits['learned_equal']=dict(screen=ds,refit=df,
                total_fit_seconds=time.perf_counter()-t,threshold=threshold,missing_edges=len(truth-chosen),
                extra_edges=len(chosen-truth),h=Q.h.tolist(),M=Q.M.tolist(),edges=Q.edges.tolist())
            methods=[]
            for label,Q in models.items():
                t=time.perf_counter();xt=AllocatedSampler(Q).sample(stream(c,rep,'knockoff_'+label),x);sampletime=time.perf_counter()-t
                logratio=swap_log_ratio(P,Q,x,xt).sum(0)
                error=P.fields(x)-Q.fields(x)
                for response,y in responses.items():
                    t=time.perf_counter();w=block_score(x,xt,y);chosen=select(w,c['q']);scoretime=time.perf_counter()-t
                    true=int(np.isin(chosen,active).sum())
                    methods.append(dict(method=label,response=response,selected=chosen.tolist(),scores=w.tolist(),
                        fdp=(len(chosen)-true)/max(1,len(chosen)),power=true/len(active),empty=not len(chosen),
                        max_null_logratio=float(logratio[null].max()),
                        max_field_mse=float(np.mean(np.sum(error*error,axis=2),axis=0).max()),
                        sampling_seconds=sampletime,score_seconds=scoretime))
            record=dict(rep=rep,status='ok',fits=fits,methods=methods)
        except Exception as exc:record=dict(rep=rep,status='failure',error=str(exc),traceback=traceback.format_exc())
        atomic_json(file,record)
        if (rep+1)%100==0 or rep+1==target:print(c['id'],rep+1,round(time.perf_counter()-start,2),flush=True)
    records=[json.loads(f.read_text()) for f in sorted((out/'records').glob('*.json'))]
    atomic_json(out/'execution.json',dict(completed=len(records),successful=sum(r['status']=='ok' for r in records),
                failures=sum(r['status']!='ok' for r in records),seconds_this_invocation=time.perf_counter()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('--limit',type=int);args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()),args.limit)
