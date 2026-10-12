#!/usr/bin/env python3
"""Plot only independently audited saved results, with provenance."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(audit: Path) -> None:
    receipt = json.loads((audit / 'receipt.json').read_text())
    if receipt['status'] != 'complete':
        raise RuntimeError('An incomplete audit cannot support this figure')
    source = audit / 'summary_recomputed.csv'
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'results/statistical_papers' / ('evaluation_figure_' + stamp)
    output.mkdir()
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
    retained = []
    null_names = [(f'h2_{hc}_wald_{kind}_by', f'{kind.capitalize()}, {label}')
                  for hc, label in [('hc0', 'HC0'), ('hc3style', 'HC3-style')]
                  for kind in ['pilot', 'naive', 'oracle']]
    fair_names = [('h2_hc3style_wald_pilot_by', 'Wald + pilot variance'),
                  ('h2_hc3style_wald_naive_by', 'Wald without pilot variance'),
                  ('h2_hc3style_wald_oracle_by', 'Known-law Wald'),
                  ('h2_knockoff_plus_pilot', 'Fitted-law knockoff+'),
                  ('h2_knockoff_plus_oracle', 'Known-law knockoff+')]
    panels = [('sp06_3', 'null_n400_m400', null_names, 'fdp', 'a  Global-null FDR\nn=400, m=400, p=12, H=2', .25),
              ('sp06_4_hc', 'binary_n1600_m400_p20', fair_names, 'fdp', 'b  Binary-response FDR\nn=1600, m=400, p=20, 10 signals', .11),
              ('sp06_4_hc', 'binary_n1600_m400_p20', fair_names, 'power', 'c  Unconditional power\nSame binary datasets and methods', 1.)]
    for panel, (axis, (version, case, names, metric, title, limit)) in enumerate(zip(axes, panels)):
        chosen = [next(r for r in rows if r['version'] == version and r['configuration'] == case and r['method'] == name)
                  for name, _ in names]
        mean = np.array([float(r[metric]) for r in chosen])
        mcse = np.array([float(r[metric + '_mcse']) for r in chosen])
        y = np.arange(len(chosen))
        axis.errorbar(mean, y, xerr=2 * mcse, fmt='o', color='black', capsize=3, markersize=4)
        axis.set_yticks(y, [label for _, label in names])
        axis.invert_yaxis()
        axis.set_xlim(0, limit)
        axis.set_title(title, loc='left', fontsize=10)
        axis.set_xlabel('FDR' if metric == 'fdp' else 'Power')
        axis.grid(axis='x', alpha=.2)
        if metric == 'fdp':
            axis.axvline(.1, color='grey', linestyle='--', linewidth=1)
        for row in chosen:
            retained.append({'panel': panel + 1, 'version': version, 'configuration': case,
                             'method': row['method'], 'metric': metric, 'mean': row[metric],
                             'mcse': row[metric + '_mcse'], 'repetitions': row['repetitions']})
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('Fig1.' + suffix), dpi=180)
    plt.close(fig)
    with (output / 'points.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(retained[0]))
        writer.writeheader()
        writer.writerows(retained)
    note = {'status': 'complete', 'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
            'audit_receipt_sha256': sha(audit / 'receipt.json'), 'script_sha256': sha(Path(__file__)),
            'outputs_sha256': {p.name: sha(p) for p in output.iterdir()},
            'caption': 'Dots are means over 1000 repetitions; bars are plus/minus two Monte Carlo standard errors, not simultaneous confidence intervals. Dashed lines denote q=0.1. HC3-style is a leverage sensitivity for fixed positive ridge, not exact ridge jackknife HC3. No datasets are added by plotting.'}
    (output / 'receipt.json').write_text(json.dumps(note, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    main(parser.parse_args().audit.resolve())
