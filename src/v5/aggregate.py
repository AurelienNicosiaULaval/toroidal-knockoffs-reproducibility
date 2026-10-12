"""Aggregate the four prespecified finite v5 studies, keeping all mechanisms distinct."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from v5.tracking import ROOT,atomic_json,digest


def aggregate() -> tuple[pd.DataFrame,pd.DataFrame]:
    paths=sorted((ROOT/'configs/v5').glob('regular_n*.json'))+sorted((ROOT/'configs/v5').glob('harmonic_cost_n*.json'))
    summaries=[];comparisons=[];individual=[];inventory=[]
    for path in paths:
        config=json.loads(path.read_text());directory=ROOT/'results/v5'/config['id']
        files=sorted((directory/'records').glob('*.json'))
        if len(files)!=config['repetitions']:
            raise RuntimeError(f"Incomplete {config['id']}: {len(files)}/{config['repetitions']}")
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['config']!=config:raise RuntimeError('Manifest/configuration mismatch')
        for name,sha in manifest['code'].items():
            if digest(directory/'source_snapshot'/name)!=sha:raise RuntimeError(f'Snapshot changed: {name}')
        records=[json.loads(f.read_text()) for f in files]
        if [r['rep'] for r in records]!=list(range(config['repetitions'])):
            raise RuntimeError('Missing, duplicated or mislabelled replicate identifiers')
        failures=sum(r['status']!='ok' for r in records)
        if failures:raise RuntimeError(f'{failures} failures require inspection')
        method_keys=lambda r:[(v['kernel'],v['harmonics'],v['response']) for v in r['methods']]
        if any(method_keys(r)!=method_keys(records[0]) for r in records):
            raise RuntimeError('Method ordering changed within an experiment')
        meta=dict(id=config['id'],n=config['n'],m=config['m'],repetitions=len(records))
        for j,first in enumerate(records[0]['methods']):
            result=dict(**meta,kernel=first['kernel'],harmonics=first['harmonics'],response=first['response'])
            for metric in ['fdp','power','empty','null_positive_fraction']:
                values=np.array([r['methods'][j][metric] for r in records],float)
                result[metric]=float(values.mean());result[metric+'_mcse']=float(values.std(ddof=1)/np.sqrt(len(values)))
            summaries.append(result)
            for r in records:
                individual.append(dict(**meta,rep=r['rep'],kernel=first['kernel'],harmonics=first['harmonics'],
                    response=first['response'],gamma_estimated=r['gamma_estimated'],
                    **{k:r['methods'][j][k] for k in ['fdp','power','empty']}))
        for response in [r['name'] for r in config['responses']]:
            index={(r['kernel'],r['harmonics']):j for j,r in enumerate(records[0]['methods']) if r['response']==response}
            contrasts=[('harmonic_gain_oracle',('oracle',2),('oracle',1)),
                       ('harmonic_gain_mle',('mle',2),('mle',1)),
                       ('pilot_effect_h1',('mle',1),('oracle',1)),
                       ('pilot_effect_h2',('mle',2),('oracle',2))]
            for name,left,right in contrasts:
                result=dict(**meta,response=response,contrast=name)
                for metric in ['fdp','power','empty']:
                    delta=np.array([float(r['methods'][index[left]][metric])-float(r['methods'][index[right]][metric]) for r in records])
                    result[metric+'_difference']=float(delta.mean());result[metric+'_paired_mcse']=float(delta.std(ddof=1)/np.sqrt(len(delta)))
                comparisons.append(result)
        jobs=[json.loads(p.read_text()) for p in sorted(directory.glob('execution_*.json'))]
        covered=sorted(i for j in jobs for i in range(j['first'],j['stop']))
        if covered!=list(range(config['repetitions'])):
            raise RuntimeError('Missing or overlapping execution accounting')
        inventory.append(dict(**meta,method_evaluations=len(records)*len(records[0]['methods']),failures=failures,jobs=jobs,
            original_seconds=sum(r['original_seconds'] for r in records),fit_seconds=sum(r['fit_seconds'] for r in records),
            sampling_seconds={k:sum(r['sampling_seconds'][k] for r in records) for k in ['oracle','mle']},
            gram_seconds={k:sum(r['gram_seconds'][k] for r in records) for k in ['oracle','mle']},
            record_hashes={p.name:digest(p) for p in files}))
    out=ROOT/'results/v5/aggregate';out.mkdir(exist_ok=True)
    summary=pd.DataFrame(summaries);paired=pd.DataFrame(comparisons)
    summary.to_csv(out/'summary.csv',index=False);paired.to_csv(out/'paired.csv',index=False)
    pd.DataFrame(individual).to_csv(out/'individual_metrics.csv',index=False)
    atomic_json(out/'inventory.json',inventory)
    atomic_json(out/'counts.json',dict(datasets=sum(r['repetitions'] for r in inventory),
        method_evaluations=sum(r['method_evaluations'] for r in inventory),failures=sum(r['failures'] for r in inventory)))
    return summary,paired


def figure(summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.1))
    large=summary[summary['response']=='second_strong']
    for (kernel,h),group in large.groupby(['kernel','harmonics']):
        group=group.sort_values('n')
        axes[0].errorbar(group['n'],group['fdp'],yerr=1.96*group['fdp_mcse'],marker='o',capsize=3,label=f'{kernel}, H={h}')
    limits=json.loads((ROOT/'results/v5/v5_0_normal_integral/summary.json').read_text())['records']
    prediction=[r['fdr'] for r in limits if r['nodes']==256][-1]
    axes[0].axhline(.1,color='black',ls='--',lw=1,label='Nominal 0.10')
    axes[0].axhline(prediction,color='grey',ls=':',lw=1,label='Predicted H=1 limit')
    axes[0].set(xlabel='Selection sample size',ylabel='Mean FDP',title='Regular pilot: n/m = 16,000')
    axes[0].set_xticks([80000,320000],labels=['80,000','320,000']);axes[0].legend(fontsize=7)
    small=summary[(summary['response']=='first_local')&(summary['kernel']=='oracle')]
    for h,group in small.groupby('harmonics'):
        group=group.sort_values('n')
        axes[1].errorbar(group['n'],group['power'],yerr=1.96*group['power_mcse'],marker='o',capsize=3,label=f'H={h}')
    axes[1].set(xlabel='Selection sample size',ylabel='Unconditional power',title='Cost when the first harmonic suffices')
    axes[1].set_xticks([400,1600]);axes[1].legend(fontsize=8)
    fig.text(.5,.012,'Bars: ±1.96 Monte Carlo SE; 1,000 paired repetitions per configuration.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1])
    for suffix in ['png','svg']:fig.savefig(ROOT/f'results/v5/aggregate/results.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    data,_=aggregate();figure(data);print(data.to_string(index=False))
