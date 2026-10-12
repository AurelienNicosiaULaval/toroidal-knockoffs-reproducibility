"""Separate oracle validation of the analytic rare-signal limit; exact-iid X."""
from __future__ import annotations
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from toroidal import select
from loading import AllocatedSampler
from experiments_v2 import get_model,sample_original,block_score
from provenance import ROOT,atomic_json,digest,stream
from v3.sign_limit import sign_limit


def main():
    c=json.loads((ROOT/'configs/v3/sign_validation.json').read_text());out=ROOT/'results/v3'/c['id'];out.mkdir(exist_ok=True)
    files=['src/v3/validate_sign_limit.py','src/v3/sign_limit.py','src/toroidal.py','src/loading.py',
           'src/experiments_v2.py','src/experimental_design.py','src/provenance.py']
    source={f:digest(ROOT/f) for f in files};signature=hashlib.sha256(json.dumps(dict(config=c,code=source),sort_keys=True).encode()).hexdigest()
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())['signature']!=signature:raise RuntimeError('Version changed')
    else:
        atomic_json(out/'manifest.json',dict(config=c,code=source,signature=signature))
        for f in files:
            path=out/'source_snapshot'/f;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/f).read_bytes())
    P=get_model(c);sampler=AllocatedSampler(P);start=time.perf_counter()
    for rep in range(c['repetitions']):
        file=out/f'records/{rep:06d}.json'
        if file.exists():continue
        x=sample_original(stream(c,rep,'original'),c,P,c['n'])
        xt=sampler.sample(stream(c,rep,'knockoff'),x);noise=stream(c,rep,'noise').normal(0,c['noise_sd'],c['n'])
        rows=[]
        for s in c['signal_counts']:
            active=np.array(c['active_when_three'] if s else [],int)
            y=c['signal_coefficient']*np.cos(x[:,active]).sum(1)+noise
            w=block_score(x,xt,y);chosen=select(w,c['q']);tp=int(np.isin(chosen,active).sum())
            null=np.setdiff1d(np.arange(P.p),active)
            separated=bool(not s or np.min(w[active])>np.max(np.abs(w[null])))
            rows.append(dict(signals=s,selected=chosen.tolist(),scores=w.tolist(),fdp=(len(chosen)-tp)/max(1,len(chosen)),
                power=tp/s if s else None,nonempty=bool(len(chosen)),active_above_null=separated))
        atomic_json(file,dict(rep=rep,status='ok',results=rows))
    records=[json.loads(f.read_text()) for f in sorted((out/'records').glob('*.json'))];summaries=[]
    for s in c['signal_counts']:
        rows=[r for entry in records for r in entry['results'] if r['signals']==s]
        selected=sum(r['nonempty'] for r in rows);ci=binomtest(selected,len(rows)).proportion_ci()
        fdps=np.array([r['fdp'] for r in rows]);power=np.array([r['power'] for r in rows]) if s else None
        summaries.append(dict(signals=s,repetitions=len(rows),nonempty_count=selected,nonempty_rate=selected/len(rows),
            nonempty_clopper_pearson_95=[ci.low,ci.high],fdr=float(fdps.mean()),fdr_mcse=float(fdps.std(ddof=1)/np.sqrt(len(rows))),
            power=float(power.mean()) if s else None,power_mcse=float(power.std(ddof=1)/np.sqrt(len(rows))) if s else None,
            separation_rate=float(np.mean([r['active_above_null'] for r in rows])),asymptotic_prediction=sign_limit(c['p'],s)))
    atomic_json(out/'summary.json',dict(completed=len(records),method_response_evaluations=2*len(records),
        failures=0,seconds_this_invocation=time.perf_counter()-start,summary=summaries))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
