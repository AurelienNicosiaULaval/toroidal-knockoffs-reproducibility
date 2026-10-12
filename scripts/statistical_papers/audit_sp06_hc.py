#!/usr/bin/env python3
"""Independent read-only audit of the paired SP06.3 and SP06.4-HC analyses.

No project fitting, p-value or selection routine is imported. The chi-square
survival functions at two and four degrees of freedom have closed forms.
This reconstructs saved results; it does not produce new simulated datasets.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def by_mask(values: np.ndarray, level: float) -> np.ndarray:
    p = len(values)
    thresholds = level * np.arange(1, p + 1) / (p * sum(1 / k for k in range(1, p + 1)))
    eligible = np.flatnonzero(np.sort(values) <= thresholds)
    return values <= thresholds[eligible[-1]] if len(eligible) else np.zeros(p, bool)


def knockoff_mask(scores: np.ndarray) -> np.ndarray:
    for threshold in sorted(set(abs(scores)) - {0.}):
        positive = np.count_nonzero(scores >= threshold)
        negative = np.count_nonzero(scores <= -threshold)
        if 10 * (1 + negative) <= max(1, positive):
            return scores >= threshold
    return np.zeros(len(scores), bool)


def pvalue(q: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    if np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise ArithmeticError("Non-positive contrast covariance")
    if q.shape[1] == 2:
        a, b, c = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
        statistic = (c * q[:, 0] ** 2 - 2 * b * q[:, 0] * q[:, 1] + a * q[:, 1] ** 2) / (a * c - b ** 2)
        return np.exp(-statistic / 2)
    if q.shape[1] == 4:
        # Deliberately uses an inverse, unlike the generator's Cholesky solve.
        statistic = np.einsum('pi,pij,pj->p', q, np.linalg.inv(covariance), q)
        return np.exp(-statistic / 2) * (1 + statistic / 2)
    raise ValueError("This audit covers only the frozen H=1 and H=2 analyses")


def moments(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def audit() -> None:
    started = time.perf_counter()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'results/statistical_papers' / ('hc_audit_' + stamp)
    output.mkdir()
    rows, paired, cases, provenance = [], [], [], {}
    maxima = {'variance': 0., 'pvalue': 0., 'summary': 0., 'paired': 0.}
    minimum_hc_increment = float('inf')
    for version in ('sp06_3', 'sp06_4_hc'):
        for summary_path in sorted((ROOT / 'results/statistical_papers' / version).glob('*/summary.json')):
            folder = summary_path.parent
            saved = json.loads(summary_path.read_text())
            manifest_path = folder / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            cfg = manifest['config']
            original = cfg['source_config']
            if original['level'] != .1:
                raise ValueError("Knockoff audit uses integer counts for q=0.1")
            source = ROOT / 'results/statistical_papers' / cfg['source_version'] / cfg['id']
            source_manifest = json.loads((source / 'manifest.json').read_text())
            for context_folder, context_manifest in ((folder, manifest), (source, source_manifest)):
                for relative, expected in context_manifest['code_sha256'].items():
                    if sha(ROOT / relative) != expected or sha(context_folder / 'source_snapshot' / relative) != expected:
                        raise RuntimeError(f"Frozen source mismatch: {relative}")
            for field, expected in (('manifest.json', manifest['source_manifest_sha256']),
                                    ('summary.json', manifest['source_summary_sha256'])):
                if sha(source / field) != expected:
                    raise RuntimeError(f"Source {field} changed")
            if cfg.get('protocol'):
                expected_protocol = cfg.get('protocol_sha256', manifest['code_sha256'].get(cfg['protocol']))
                if not expected_protocol or sha(ROOT / cfg['protocol']) != expected_protocol:
                    raise RuntimeError("Frozen follow-up protocol changed")
            provenance[str(summary_path.relative_to(ROOT))] = sha(summary_path)
            provenance[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
            names = [row['method'] for row in saved['summary']]
            values = {name: {k: [] for k in ('fdp', 'power', 'empty', 'marginal_null_rejection')} for name in names}
            for index in range(cfg['repetitions']):
                path = folder / f'rep_{index:04d}.json'
                record = json.loads(path.read_text())
                original_path = source / path.name
                original_record = json.loads(original_path.read_text())
                if record['errors'] or any(r['failed'] for r in record['methods']):
                    raise RuntimeError("Failures require a separately specified metric audit")
                if record['replicate'] != index or record['method_order'] != names:
                    raise RuntimeError("Record identity/order mismatch")
                if sha(path.with_suffix('.npz')) != record['arrays_sha256']:
                    raise RuntimeError("Analysis array hash mismatch")
                if sha(original_path) != record['source_json_sha256']:
                    raise RuntimeError("Source record hash mismatch")
                if sha(original_path.with_suffix('.npz')) != record['source_arrays_sha256']:
                    raise RuntimeError("Source array hash mismatch")
                with np.load(original_path.with_suffix('.npz'), allow_pickle=False) as old, np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
                    k = len(old['pilot_parameters'])
                    v = np.zeros((k, k))
                    v[np.triu_indices(k)] = old['pilot_covariance_upper']
                    v += np.triu(v, 1).T
                    pv, masks = {}, {}
                    for h in (1, 2):
                        q = arrays[f'h{h}_contrasts']
                        components = arrays[f'h{h}_oracle_components']
                        b = arrays[f'h{h}_pilot_derivative']
                        correction = np.einsum('pik,kl,pjl->pij', b, v, b, optimize=True) / original['m']
                        for hc_index, hc in enumerate(('hc0', 'hc3style')):
                            total = components[hc_index, 0] + correction
                            retained = arrays[f'h{h}_total_variance'][hc_index]
                            maxima['variance'] = max(maxima['variance'], float(abs(total - retained).max()))
                            np.testing.assert_allclose(total, retained, rtol=3e-10, atol=2e-12)
                            for kind, qi, covariance in (('pilot', q[0], total), ('naive', q[0], components[hc_index, 0]),
                                                        ('oracle', q[1], components[hc_index, 1])):
                                name = f'h{h}_{hc}_wald_{kind}_by'
                                pv[name] = pvalue(qi, covariance)
                                masks[name] = by_mask(pv[name], .1)
                        for kind_index, kind in enumerate(('pilot', 'oracle')):
                            masks[f'h{h}_knockoff_plus_{kind}'] = knockoff_mask(arrays[f'h{h}_knockoff_scores'][kind_index])
                        increment = components[1] - components[0]
                        smallest = float(np.linalg.eigvalsh(increment).min())
                        minimum_hc_increment = min(minimum_hc_increment, smallest)
                        if smallest < -1e-10 * max(1., float(abs(components).max())):
                            raise ArithmeticError("HC3-style covariance increment is not positive semidefinite")
                    observed_pv = np.array([pv[name] for name in record['pvalue_order']])
                    maxima['pvalue'] = max(maxima['pvalue'], float(abs(observed_pv - arrays['pvalues']).max()))
                    np.testing.assert_allclose(observed_pv, arrays['pvalues'], rtol=3e-9, atol=5e-12)
                    np.testing.assert_array_equal([masks[name] for name in names], arrays['selected_mask'])
                    np.testing.assert_array_equal(arrays['pvalues'][:3], old['pvalues'])
                    np.testing.assert_array_equal(arrays['h1_knockoff_scores'], old['knockoff_scores'])
                active = np.array(record['active'], int)
                null = np.setdiff1d(np.arange(original['p']), active)
                for method in record['methods']:
                    name = method['method']
                    mask = masks[name]
                    np.testing.assert_array_equal(np.flatnonzero(mask), method['selected'])
                    metric = {'fdp': float(mask[null].sum() / max(1, mask.sum())),
                              'power': float(mask[active].sum() / max(1, len(active))), 'empty': float(not mask.any())}
                    if name in pv:
                        metric['marginal_null_rejection'] = float((pv[name][null] <= .05).mean())
                    for key, value in metric.items():
                        np.testing.assert_allclose(value, method[key], rtol=0, atol=1e-14)
                        values[name][key].append(value)
            for name in names:
                row = {'version': version, 'configuration': cfg['id'], 'method': name,
                       'n': original['n'], 'm': original['m'], 'p': original['p'], 'repetitions': cfg['repetitions']}
                retained = next(r for r in saved['summary'] if r['method'] == name)
                for metric, series in values[name].items():
                    if series:
                        mean, mcse = moments(np.array(series))
                        row.update({metric: mean, metric + '_mcse': mcse})
                        maxima['summary'] = max(maxima['summary'], abs(mean - retained[metric]), abs(mcse - retained[metric + '_mcse']))
                        np.testing.assert_allclose([mean, mcse], [retained[metric], retained[metric + '_mcse']], rtol=0, atol=5e-15)
                rows.append(row)
            for left, right in itertools.combinations(names, 2):
                for metric in ('fdp', 'power', 'empty'):
                    mean, mcse = moments(np.array(values[left][metric]) - np.array(values[right][metric]))
                    retained = next(r for r in saved['paired'] if r['left'] == left and r['right'] == right and r['metric'] == metric)
                    maxima['paired'] = max(maxima['paired'], abs(mean - retained['difference_left_minus_right']), abs(mcse - retained['mcse']))
                    np.testing.assert_allclose([mean, mcse], [retained['difference_left_minus_right'], retained['mcse']], rtol=0, atol=5e-15)
                    paired.append({'version': version, 'configuration': cfg['id'], 'left': left, 'right': right,
                                   'metric': metric, 'difference_left_minus_right': mean, 'mcse': mcse})
            cases.append({'version': version, 'configuration': cfg['id'], 'records': cfg['repetitions'],
                          'decisions': cfg['repetitions'] * len(names), 'source_version': cfg['source_version']})
            print(version, cfg['id'], 'verified', flush=True)
    for name, table in (('summary_recomputed.csv', rows), ('paired_recomputed.csv', paired)):
        columns = list(dict.fromkeys(k for row in table for k in row))
        with (output / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table)
    receipt = {'status': 'complete', 'cases': cases, 'max_absolute_errors': maxima,
               'minimum_HC3style_covariance_increment_eigenvalue': minimum_hc_increment,
               'source_sha256': provenance, 'script_sha256': sha(Path(__file__)),
               'elapsed_seconds': time.perf_counter() - started,
               'scope': 'Independent recalculation of saved SP06.3 and SP06.4-HC results; 8000 paired reanalyses, 128000 decisions; zero new datasets and no seed replays'}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_sha256'}, indent=2))
    print(output)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        audit()
