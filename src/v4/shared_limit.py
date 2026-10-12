"""Monte Carlo evaluation of the proved Gaussian common-pilot limit."""
from __future__ import annotations
import json,time
import numpy as np
from v4.tracking import ROOT,prepare,atomic_json,stream


def main():
    c=dict(id='v4_0_shared_normal_limit',seed=930411,draws=500000,chunk=10000,
           scales=[0.,.25,.5,1.,2.,4.,8.],nulls=10,signals=10,q=.1)
    out=prepare(c,['src/v4/shared_limit.py','src/v4/tracking.py','src/provenance.py'])
    if (out/'summary.json').exists():return
    start=time.perf_counter();runs=[]
    for batch,first in enumerate(range(0,c['draws'],c['chunk'])):
        size=min(c['chunk'],c['draws']-first);rng=stream(c,batch,'normal');z=rng.normal(size=(size,10));common=rng.normal(size=(size,1));columns=[]
        for t in c['scales']:
            w=z+t*common;order=np.argsort(-abs(w),axis=1);signs=np.take_along_axis(w,order,axis=1)>0
            columns.append(np.cumprod(signs,axis=1).sum(axis=1))
        runs.append(np.array(columns,dtype=np.int8).T)
    runs=np.concatenate(runs);np.savez_compressed(out/'individual_runs.npz',positive_run=runs)
    fdp=runs/(10+runs);records=[]
    for j,t in enumerate(c['scales']):
        d=fdp[:,j]-fdp[:,0]
        records.append(dict(scale=t,fdr=float(fdp[:,j].mean()),mcse=float(fdp[:,j].std(ddof=1)/np.sqrt(len(fdp))),
            difference_vs_zero=float(d.mean()),paired_mcse=float(d.std(ddof=1)/np.sqrt(len(d)))))
    atomic_json(out/'summary.json',dict(config=c,records=records,analytic_infinite_scale_fdr=.25,seconds=time.perf_counter()-start))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
