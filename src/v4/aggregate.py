"""Rebuild curated v4 tables from individual records, without pooling versions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v4.tracking import ROOT, atomic_json, digest


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    configurations = sorted((ROOT / 'configs/v4').glob('local_*.json'))
    configurations += sorted((ROOT / 'configs/v4').glob('counter_n*.json'))
    rows, contrasts, inventory, individuals = [], [], [], []
    for path in configurations:
        config = json.loads(path.read_text())
        directory = ROOT / 'results/v4' / config['id']
        files = sorted((directory / 'records').glob('*.json'))
        if len(files) != config['repetitions']:
            raise RuntimeError(f"Incomplete study: {config['id']} ({len(files)} records)")
        execution = json.loads((directory / 'execution.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['config'] != config:
            raise RuntimeError(f'Configuration mismatch: {path}')
        for source, sha in manifest['code'].items():
            if digest(directory / 'source_snapshot' / source) != sha:
                raise RuntimeError(f'Source snapshot changed: {source}')
        records = [json.loads(f.read_text()) for f in files]
        failures = [r for r in records if r['status'] != 'ok']
        if failures:
            raise RuntimeError(f"{len(failures)} failures in {config['id']}; inspect before aggregation")
        kind = 'local' if 'local_' in config['id'] else 'counter'
        metadata = dict(id=config['id'], kind=kind, n=config['n'], repetitions=len(records),
                        graph=config.get('graph', 'independent_pairs'), gamma=config['gamma'])
        base_index = config['alphas'].index(0.) if kind == 'local' else 0
        for j, first in enumerate(records[0]['methods']):
            label = str(first['alpha']) if kind == 'local' else first['method']
            result = dict(**metadata, method=label, tuned=first.get('tuned', False))
            for metric in ('fdp', 'power', 'empty', 'sampling_seconds', 'score_seconds'):
                values = np.array([r['methods'][j][metric] for r in records], float)
                result[metric] = float(values.mean())
                result[metric + '_mcse'] = float(values.std(ddof=1) / np.sqrt(len(values)))
            if kind == 'counter':
                for metric in ('gamma_work', 'null_positive_fraction', 'max_null_logratio'):
                    result[metric] = float(np.mean([r['methods'][j][metric] for r in records]))
            rows.append(result)
            for r in records:
                individuals.append(dict(**metadata, rep=r['rep'], method=label,
                                        **{k:r['methods'][j][k] for k in ('fdp','power','empty')}))
            if j != base_index:
                comparison = dict(**metadata, method=label,
                                  reference='0.0' if kind == 'local' else 'oracle')
                for metric in ('fdp', 'power', 'empty'):
                    difference = np.array([float(r['methods'][j][metric]) -
                                           float(r['methods'][base_index][metric]) for r in records])
                    comparison[metric + '_difference'] = float(difference.mean())
                    comparison[metric + '_paired_mcse'] = float(difference.std(ddof=1)/np.sqrt(len(records)))
                contrasts.append(comparison)
        inventory.append(dict(**metadata, **execution,
                              method_evaluations=len(records)*len(records[0]['methods']),
                              manifest_sha256=digest(directory/'manifest.json'),
                              records_sha256={f.name:digest(f) for f in files}))
    output = ROOT / 'results/v4/aggregate'
    output.mkdir(exist_ok=True)
    summary, paired = pd.DataFrame(rows), pd.DataFrame(contrasts)
    summary.to_csv(output / 'summary.csv', index=False)
    paired.to_csv(output / 'paired.csv', index=False)
    pd.DataFrame(individuals).to_csv(output / 'individual_metrics.csv', index=False)
    atomic_json(output / 'inventory.json', inventory)
    atomic_json(output / 'counts.json', dict(datasets=sum(x['repetitions'] for x in inventory),
        method_evaluations=sum(x['method_evaluations'] for x in inventory),
        failures=sum(x['failures'] for x in inventory), summary_rows=len(rows)))
    return summary, paired


def figures(summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    counter = summary[summary['kind'] == 'counter']
    labels = {'oracle':'Known law', 'mle_equal':'MLE, m=n',
              'mle_small':'MLE, small pilot', 'bias_stress':'Deliberately biased estimate'}
    for label, group in counter.groupby('method', sort=False):
        group = group.sort_values('n')
        axes[0].errorbar(group['n'], group['fdp'], yerr=1.96*group['fdp_mcse'],
                         marker='o', label=labels[label], capsize=3)
    axes[0].axhline(.1, color='black', ls='--', lw=1)
    axes[0].set(xscale='log', xlabel='Selection sample size n', ylabel='Mean FDP',
                title='Finite toroidal experiments: 1,000 repetitions')
    axes[0].legend(fontsize=8)
    axes[0].set_xticks([2000,8000,32000], labels=['2,000','8,000','32,000'])
    for tag, label in [('star_g0p5','Star, g=0.5'),('star_g1p0','Star, g=1'),('chain_g1p0','Chain, g=1')]:
        data = json.loads((ROOT/f'results/v4/v4_0_cal_{tag}/summary.json').read_text())
        phase = data['phases']['validation']
        axes[1].errorbar(data['config']['alphas'], phase['power'],
            yerr=1.96*np.array(phase['power_mcse']), marker='o', capsize=3, label=label)
    axes[1].set(xlabel='Endpoint allocation exponent', ylabel='Unconditional power',
                title='Gaussian local limit: 200,000 validation draws')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for extension in ['png','svg']:
        fig.savefig(ROOT/f'results/v4/aggregate/scientific_results.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    summary, paired = aggregate()
    figures(summary)
    print(summary.to_string(index=False))
