"""Recompute manuscript metrics from retained records without rerunning data.

All thresholds and FDPs are recomputed independently from the saved scores.
Study versions remain distinct. No fitted-pilot MSE uncertainty is reported.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection(scores, q):
    w=np.asarray(scores,float)
    if w.ndim!=1 or not np.isfinite(w).all():
        raise ValueError('Finite score vector required')
    for threshold in sorted(set(np.abs(w[w!=0]))):
        if (1+int(np.sum(w<=-threshold)))/max(1,int(np.sum(w>=threshold)))<=q:
            return np.flatnonzero(w>=threshold)
    return np.array([],dtype=int)


def reaggregate():
    start=time.perf_counter()
    configs=sorted((ROOT/'configs/v5').glob('regular_n*.json'))+sorted((ROOT/'configs/v5').glob('harmonic_cost_n*.json'))
    configs+= [ROOT/f'configs/v6/{name}.json' for name in ['small','large','comparable']]
    individual=[];ledger=[];inputs=[]
    for config_path in configs:
        c=json.loads(config_path.read_text())
        version='v5' if 'v5' in c['id'] else 'v6'
        folder=ROOT/'results'/version/c['id']
        manifest=json.loads((folder/'manifest.json').read_text())
        if manifest['config']!=c:raise RuntimeError('Configuration/manifest mismatch')
        for name,sha in manifest['code'].items():
            if digest(folder/'source_snapshot'/name)!=sha:raise RuntimeError('Changed retained source snapshot')
        files=sorted((folder/'records').glob('*.json'))
        if len(files)!=c['repetitions']:raise RuntimeError('Incomplete study')
        for rep,path in enumerate(files):
            record=json.loads(path.read_text())
            if record['rep']!=rep or record['status']!='ok':raise RuntimeError('Missing ID or failed study requires inspection')
            ledger.append(dict(study=c['id'],rep=rep,path=str(path.relative_to(ROOT)),sha256=digest(path)))
            for method in record['methods']:
                selected=selection(method['scores'],c['q'])
                active=np.arange(0,2*c['pairs'],2)
                tp=int(np.isin(selected,active).sum())
                values=dict(fdp=(len(selected)-tp)/max(1,len(selected)),power=tp/len(active),empty=float(len(selected)==0))
                if selected.tolist()!=method['selected']:raise AssertionError('Saved threshold decisions disagree')
                for metric,value in values.items():
                    if abs(value-method[metric])>1e-14:raise AssertionError('Saved metric disagrees: '+metric)
                label=(method['kernel']+'_H'+str(method['harmonics'])) if version=='v5' else method['method']
                individual.append(dict(study=c['id'],version=version,rep=rep,n=c['n'],m=c['m'],
                                       method=label,response=method.get('response','second_strong'),**values))
        jobs=[json.loads(p.read_text()) for p in folder.glob('execution_*.json')]
        covered=sorted(i for j in jobs for i in range(j['first'],j['stop']))
        if covered!=list(range(c['repetitions'])):raise RuntimeError('Execution ledger does not cover unique repeats')
        inputs.append(dict(config=str(config_path.relative_to(ROOT)),config_sha256=digest(config_path),
                           manifest_sha256=digest(folder/'manifest.json'),study=c['id'],datasets=len(files),
                           archived_execution_seconds=sum(j['seconds'] for j in jobs),
                           archived_failures=sum(j['failures'] for j in jobs)))
    frame=pd.DataFrame(individual)
    keys=['study','version','n','m','response','method']
    summary=[];paired=[]
    for key,group in frame.groupby(keys,sort=True):
        row=dict(zip(keys,key));row['repetitions']=len(group)
        for metric in ['fdp','power','empty']:
            value=group[metric].to_numpy(float)
            row[metric]=float(value.mean())
            row[metric+'_mcse']=float(value.std(ddof=1)/np.sqrt(len(value)))
        summary.append(row)
    for (study,response),group in frame.groupby(['study','response'],sort=True):
        methods=sorted(group.method.unique())
        for left,right in itertools.combinations(methods,2):
            l=group[group.method==left].set_index('rep');r=group[group.method==right].set_index('rep')
            if not l.index.equals(r.index):raise RuntimeError('Unpaired record IDs')
            row=dict(study=study,response=response,left=left,right=right,repetitions=len(l))
            for metric in ['fdp','power','empty']:
                value=(l[metric]-r[metric]).to_numpy(float)
                row[metric+'_difference']=float(value.mean())
                row[metric+'_paired_mcse']=float(value.std(ddof=1)/np.sqrt(len(value)))
            paired.append(row)
    summaries=pd.DataFrame(summary);contrasts=pd.DataFrame(paired)
    # Compare only bounded inferential metrics, leaving heavy-tail pilot diagnostics aside.
    for version in ['v5','v6']:
        retained=pd.read_csv(ROOT/f'results/{version}/aggregate/summary.csv')
        for row in summaries[summaries.version==version].itertuples():
            candidates=retained[retained.id==row.study]
            if version=='v5':
                kernel,h=row.method.split('_H')
                candidates=candidates[(candidates.kernel==kernel)&(candidates.harmonics==int(h))&(candidates.response==row.response)]
            else:candidates=candidates[candidates.method==row.method]
            if len(candidates)!=1:raise RuntimeError('Ambiguous comparison')
            for metric in ['fdp','power','empty']:
                for suffix in ['', '_mcse']:
                    if abs(getattr(row,metric+suffix)-float(candidates.iloc[0][metric+suffix]))>5e-15:
                        raise AssertionError('Retained aggregate disagrees')
    out=ROOT/'results/v10'
    summaries.to_csv(out/'summary.csv',index=False)
    contrasts.to_csv(out/'paired.csv',index=False)
    frame.to_csv(out/'individual_metrics.csv',index=False)
    pd.DataFrame(ledger).to_csv(out/'record_ledger.csv',index=False)
    result=dict(inputs=inputs,retained_datasets=len(ledger),method_evaluations=len(frame),
                all_decisions_recomputed=True,aggregate_max_tolerance=5e-15,
                failures=sum(s['archived_failures'] for s in inputs),seconds=time.perf_counter()-start,
                scope='reaggregation of saved data, not new simulation of these datasets')
    (out/'aggregation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))
    return summaries,contrasts


if __name__=='__main__':reaggregate()
