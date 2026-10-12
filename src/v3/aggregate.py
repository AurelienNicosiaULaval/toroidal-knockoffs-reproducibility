"""Aggregate only the explicitly registered v3.0 experiments, preserving failures."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from provenance import ROOT, atomic_json, digest


def summarize(values):
    a=np.asarray(values,float)
    return float(a.mean()),float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else float('nan')


def main():
    rows=[];fits=[];audits=[];failures=[];sources={}
    specifications=json.loads((ROOT/'configs/v3/study_manifest.json').read_text())['experiments']
    for specification in specifications:
        c=json.loads((ROOT/specification).read_text());out=ROOT/'results/v3'/c['id']
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['config']!=c:raise RuntimeError('Configuration differs from executed manifest')
        for source,expected in manifest['code'].items():
            if digest(out/'source_snapshot'/source)!=expected:raise RuntimeError('Corrupt source snapshot')
        files=sorted((out/'records').glob('*.json'))
        if len(files)!=c['repetitions']:raise RuntimeError(f"Incomplete experiment: {c['id']}")
        for file in files:
            sources[str(file.relative_to(ROOT))]=digest(file);r=json.loads(file.read_text())
            if r['status']!='ok':
                failures.append(dict(experiment=c['id'],rep=r['rep'],error=r['error']));continue
            for entry in r['methods']:rows.append(dict(n=c['n'],rep=r['rep'],**entry))
            for label,fit in r['fits'].items():
                parts=[fit['screen'],fit['refit']] if label=='learned_equal' else [fit]
                fits.append(dict(n=c['n'],rep=r['rep'],method=label,
                    fit_seconds=sum(f['seconds'] for f in parts),
                    max_gradient_l2=max(f['gradient_l2'] for f in parts),
                    max_objective_gap_upper=max(f['global_objective_gap_upper'] for f in parts),
                    missing_edges=fit.get('missing_edges',0),extra_edges=fit.get('extra_edges',0)))
        execution=json.loads((out/'execution.json').read_text())
        audits.append(dict(experiment=c['id'],n=c['n'],**execution,signature=manifest['signature']))
    table=pd.DataFrame(rows);summary=[];paired=[]
    for (n,response,method),group in table.groupby(['n','response','method']):
        item=dict(n=n,response=response,method=method,repetitions=len(group))
        for column in ['fdp','power','empty','max_field_mse','sampling_seconds','score_seconds']:
            item[column],item[column+'_mcse']=summarize(group[column])
        for probability in [.5,.9,.99]:item[f'null_logratio_q{round(100*probability)}']=float(group.max_null_logratio.quantile(probability))
        summary.append(item)
    for (n,response),group in table.groupby(['n','response']):
        for left,right in [('known_sqrt','oracle'),('known_equal','oracle'),('learned_equal','oracle'),('known_sqrt','known_equal')]:
            a=group[group.method==left].set_index('rep');b=group[group.method==right].set_index('rep')
            if not a.index.equals(b.index):raise RuntimeError('Unpaired replication indices')
            record=dict(n=n,response=response,left=left,right=right,repetitions=len(a))
            for metric in ['fdp','power','empty']:record[metric],record[metric+'_mcse']=summarize(a[metric].astype(float)-b[metric].astype(float))
            paired.append(record)
    output=ROOT/'results/v3/aggregate';output.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired.csv',index=False)
    pd.DataFrame(fits).to_csv(output/'fit_diagnostics.csv',index=False)
    table.to_json(output/'method_records.jsonl',orient='records',lines=True)
    atomic_json(output/'audit.json',dict(experiments=audits,failures=failures,
        successful_datasets=sum(a['successful'] for a in audits),method_response_evaluations=len(rows),
        individual_record_sha256=sources))
    print(pd.DataFrame(summary)[['n','response','method','fdp','fdp_mcse','power','empty']].to_string(index=False))


if __name__=='__main__':main()
