"""Strict v6 aggregation, including covariance diagnostics across independent data sets."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from v6.tracking import ROOT,atomic_json,digest


def covariance_summary(x:np.ndarray) -> tuple[dict,np.ndarray]:
    rows,k=x.shape;s=x.sum(axis=0);cross=x.T@x
    cov=(cross-np.outer(s,s)/rows)/(rows-1)
    leave_sum=s[None,:]-x
    leave=(cross[None,:,:]-x[:,:,None]*x[:,None,:]-leave_sum[:,:,None]*leave_sum[:,None,:]/(rows-1))/(rows-2)
    def metrics(c):
        d=np.diagonal(c,axis1=-2,axis2=-1);var=d.mean(axis=-1)
        off=(c.sum(axis=(-2,-1))-d.sum(axis=-1))/(k*(k-1))
        scale=np.sqrt(d[..., :,None]*d[...,None,:])
        corr=np.divide(c,scale,out=np.zeros_like(c),where=scale>0)
        mean_corr=(corr.sum(axis=(-2,-1))-np.diagonal(corr,axis1=-2,axis2=-1).sum(axis=-1))/(k*(k-1))
        return np.stack([var,off,mean_corr],axis=-1)
    point=metrics(cov);jack=metrics(leave)
    se=np.sqrt((rows-1)/rows*((jack-jack.mean(axis=0))**2).sum(axis=0))
    result={'covariance':cov.tolist(),'mean':x.mean(axis=0).tolist()}
    for j,name in enumerate(['variance','offdiag_covariance','offdiag_correlation']):
        result[name]=float(point[j]);result[name+'_jackknife_se']=float(se[j])
    return result,jack


def aggregate() -> pd.DataFrame:
    output=ROOT/'results/v6/aggregate';output.mkdir(exist_ok=True)
    summaries=[];paired=[];mechanisms=[];inventory=[];individual=[]
    for label in ['small','large','comparable']:
        c=json.loads((ROOT/f'configs/v6/{label}.json').read_text());directory=ROOT/'results/v6'/c['id']
        files=sorted((directory/'records').glob('*.json'))
        if len(files)!=c['repetitions']:raise RuntimeError(f'Incomplete {label}: {len(files)}')
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['config']!=c:raise RuntimeError('Configuration changed')
        for name,sha in manifest['code'].items():
            if digest(directory/'source_snapshot'/name)!=sha:raise RuntimeError('Snapshot changed: '+name)
        records=[json.loads(p.read_text()) for p in files]
        if [r['rep'] for r in records]!=list(range(c['repetitions'])):raise RuntimeError('Replicate IDs changed')
        if any(r['status']!='ok' for r in records):raise RuntimeError('Failures require inspection')
        methods=['oracle','pooled','separate','shared_single']
        if any([a['method'] for a in r['methods']]!=methods for r in records):raise RuntimeError('Method order changed')
        meta=dict(id=c['id'],n=c['n'],m=c['m'],repetitions=len(records));jacks={};covs={}
        for j,method in enumerate(methods):
            rows=[r['methods'][j] for r in records]
            scores=np.array([r['scores'][1::2] for r in rows])*np.sqrt(c['n'])
            g=np.array([r['gamma_work'] for r in rows]);error=g-c['gamma']
            covariance,jack=covariance_summary(scores);jacks[method]=jack;covs[method]=covariance
            result=dict(**meta,method=method)
            values={key:np.array([r[key] for r in rows],float) for key in ['fdp','power','empty','null_positive_fraction']}
            signs=np.take_along_axis(scores,np.argsort(-np.abs(scores),axis=1),axis=1)>0
            values.update(pilot_bias=error.mean(axis=1),pilot_mse=(error**2).mean(axis=1),pilot_boundary=(g==0).mean(axis=1),null_mean=scores.mean(axis=1),largest_null_positive=signs[:,0].astype(float),null_run=np.cumprod(signs,axis=1).sum(axis=1))
            for key,v in values.items():
                result[key]=float(v.mean());result[key+'_mcse']=float(v.std(ddof=1)/np.sqrt(len(v)))
            result.update({key:value for key,value in covariance.items() if key not in ['mean','covariance']})
            summaries.append(result)
            mechanisms.append(dict(**meta,method=method,null_score=covariance,pilot_covariance=np.cov(g,rowvar=False).tolist()))
            for r,record in zip(rows,records):
                individual.append(dict(**meta,rep=record['rep'],method=method,**{key:r[key] for key in ['fdp','power','empty','null_positive_fraction']}))
        for left,right in [('separate','pooled'),('separate','shared_single'),('pooled','oracle'),('separate','oracle'),('shared_single','oracle')]:
            jl=methods.index(left);jr=methods.index(right);item=dict(**meta,left=left,right=right)
            for key in ['fdp','power','empty','null_positive_fraction']:
                delta=np.array([float(r['methods'][jl][key])-float(r['methods'][jr][key]) for r in records])
                item[key+'_difference']=float(delta.mean());item[key+'_paired_mcse']=float(delta.std(ddof=1)/np.sqrt(len(delta)))
            for column,key in enumerate(['variance','offdiag_covariance','offdiag_correlation']):
                jack=jacks[left][:,column]-jacks[right][:,column]
                item[key+'_difference']=covs[left][key]-covs[right][key]
                item[key+'_paired_jackknife_se']=float(np.sqrt((len(jack)-1)/len(jack)*((jack-jack.mean())**2).sum()))
            paired.append(item)
        jobs=[json.loads(p.read_text()) for p in sorted(directory.glob('execution_*.json'))]
        covered=sorted(i for job in jobs for i in range(job['first'],job['stop']))
        if covered!=list(range(c['repetitions'])):raise RuntimeError('Execution ranges incomplete/overlapping')
        inventory.append(dict(**meta,method_evaluations=4*len(records),failures=0,jobs=jobs,
            original_seconds=sum(r['original_seconds'] for r in records),fit_seconds=sum(r['fit_seconds'] for r in records),
            sampling_seconds={m:sum(r['sampling_seconds'][m] for r in records) for m in methods},
            gram_seconds={m:sum(r['gram_seconds'][m] for r in records) for m in methods},
            record_hashes={p.name:digest(p) for p in files}))
    summary=pd.DataFrame(summaries);summary.to_csv(output/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired.csv',index=False);pd.DataFrame(individual).to_csv(output/'individual_metrics.csv',index=False)
    atomic_json(output/'mechanisms.json',mechanisms);atomic_json(output/'inventory.json',inventory)
    atomic_json(output/'counts.json',dict(datasets=sum(r['repetitions'] for r in inventory),method_evaluations=sum(r['method_evaluations'] for r in inventory),failures=0))
    return summary


def figure(summary:pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    names=['oracle','pooled','separate','shared_single'];labels=['Known law','Pooled','Separate','Shared single']
    for i,(method,label) in enumerate(zip(names,labels)):
        rows=summary[summary.method==method].set_index('id').loc[['v6_0_small','v6_0_large','v6_0_comparable']]
        x=np.arange(3)+(i-1.5)*.15
        axes[0].errorbar(x,rows.fdp,yerr=1.96*rows.fdp_mcse,fmt='o',capsize=3,label=label)
        axes[1].errorbar(x,rows.offdiag_correlation,yerr=1.96*rows.offdiag_correlation_jackknife_se,fmt='o',capsize=3,label=label)
    axes[0].axhline(.1,color='black',ls='--',lw=1);axes[1].axhline(0,color='black',ls='--',lw=1)
    for ax in axes:ax.set_xticks(range(3),labels=['80,000 / 5','320,000 / 20','80,000 / 80,000']);ax.set_xlabel('Selection size / pilot size')
    axes[0].set(ylabel='Mean FDP',title='FDR at nominal 0.10');axes[0].legend(fontsize=8)
    axes[1].set(ylabel='Average off-diagonal correlation',title='Dependence among null scores')
    fig.text(.5,.01,'1,000 paired repetitions per configuration. Bars: ±1.96 SE; covariance diagnostics use a replicate jackknife.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1])
    for suffix in ['png','svg']:fig.savefig(ROOT/f'results/v6/aggregate/results.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    data=aggregate();figure(data);print(data.to_string(index=False))
