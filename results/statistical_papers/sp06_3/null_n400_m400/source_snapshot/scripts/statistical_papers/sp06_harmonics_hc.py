#!/usr/bin/env python3
"""Resumable, paired SP06.3 reanalysis of frozen selection datasets."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import scipy
from scipy.special import expit
from threadpoolctl import threadpool_limits

from fit_model import unpack
from provenance import atomic_json, digest, stream
from sp06.harmonics_hc import block_norm_scores, harmonic_features, hc_pair
from sp06.star import canonical_statistics, exact_star, true_star
from sp06.wald import by_select
from sp06_selection import SOURCES as ORIGINAL_SOURCES, atomic_arrays, moments, read_record, serial
from toroidal import EdgeSampler, knockstat, select

SOURCES = ORIGINAL_SOURCES + ['src/sp06/harmonics_hc.py', 'tests/test_sp06_harmonics_hc.py',
    'scripts/statistical_papers/sp06_harmonics_hc.py', 'configs/statistical_papers/sp06_3_harmonics_hc.json',
    'docs/statistical_papers/SP06_3_PROTOCOLE_FR_20261010.md']


def method_names(config: dict) -> list[str]:
    names = []
    for h in config['orders']:
        for hc in config['covariances']:
            names += [f'h{h}_{hc}_wald_{kind}_by' for kind in ('pilot', 'naive', 'oracle')]
        names += [f'h{h}_knockoff_plus_{kind}' for kind in ('pilot', 'oracle')]
    return names


def prepare(base: dict, case: str) -> tuple[dict, Path, Path]:
    source = ROOT / 'results/statistical_papers' / base['source_version'] / case
    source_manifest = json.loads((source / 'manifest.json').read_text())
    for relative, expected in source_manifest['code_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen source changed: {relative}')
    config = dict(base, id=case, source_config=source_manifest['config'])
    manifest = {'config': config, 'code_sha256': {p: digest(ROOT / p) for p in SOURCES},
                'source_manifest_sha256': digest(source / 'manifest.json'),
                'source_summary_sha256': digest(source / 'summary.json'),
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'scipy': scipy.__version__, 'platform': platform.platform(), 'blas_threads': 1}}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output = ROOT / 'results/statistical_papers' / base['version'] / case
    path = output / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text())['signature'] != signature:
            raise RuntimeError('Source/configuration changed; use a new analysis version')
    else:
        manifest.update(signature=signature, created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(path, manifest)
        for relative in SOURCES:
            target = output / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
    return config, source, output


def repetition(config: dict, source: Path, index: int) -> tuple[dict, dict]:
    started = time.perf_counter()
    original_config = config['source_config']
    path = source / f'rep_{index:04d}.json'
    previous, retained = read_record(path)
    p, n, m = (original_config[key] for key in ('p', 'n', 'm'))
    truth = true_star(original_config)
    active = [] if original_config['response'] == 'null' else original_config['active']
    null = np.setdiff1d(np.arange(p), active)
    names = method_names(config)
    masks = {name: np.zeros(p, bool) for name in names}
    pvalues = {name: np.ones(p) for name in names if '_wald_' in name}
    failed = {name: True for name in names}
    arrays = {'pvalues': np.ones((len(pvalues), p)), 'selected_mask': np.zeros((len(names), p), bool)}
    errors, times = [], {}
    reconstruction = time.perf_counter()
    if previous['errors'] or any(method['failed'] for method in previous['methods']):
        raise RuntimeError('Source failures need an explicit reconstruction policy')
    x, sampling = exact_star(stream(original_config, index, 'original'), truth, n)
    if sampling != previous['original_sampler']:
        raise RuntimeError('Original sampling replay mismatch')
    rng = stream(original_config, index, 'response')
    if original_config['response'] == 'null':
        y = rng.normal(scale=original_config['noise_sd'], size=n)
    else:
        nonlinear = np.cos(2 * (x[:, active] - original_config['field_phase'])).sum(axis=1) / np.sqrt(len(active))
        y = (rng.binomial(1, expit(original_config['binary_scale'] * nonlinear + original_config['binary_intercept']))
             if original_config['response'] == 'binary' else
             original_config['continuous_scale'] * nonlinear + rng.normal(scale=original_config['noise_sd'], size=n))
    fitted = unpack(retained['pilot_parameters'], p, truth.edges)
    k = len(retained['pilot_parameters'])
    covariance = np.zeros((k, k))
    covariance[np.triu_indices(k)] = retained['pilot_covariance_upper']
    covariance += np.triu(covariance, 1).T
    t = canonical_statistics(x, truth.edges)
    oracle = EdgeSampler(truth).sample(stream(original_config, index, 'oracle_witness'), x)
    witness = EdgeSampler(fitted).sample(stream(original_config, index, 'estimated_witness'), x)
    times['reconstruct_original_response_witnesses'] = time.perf_counter() - reconstruction
    for h in config['orders']:
        first = harmonic_features(x, h)
        arrays[f'h{h}_contrasts'] = np.full((2, p, 2 * h), np.nan)
        arrays[f'h{h}_oracle_components'] = np.full((2, 2, p, 2 * h, 2 * h), np.nan)
        arrays[f'h{h}_total_variance'] = np.full((2, p, 2 * h, 2 * h), np.nan)
        arrays[f'h{h}_pilot_derivative'] = np.full((p, 2 * h, k), np.nan)
        arrays[f'h{h}_leverage_min_mean_max'] = np.full((2, 3), np.nan)
        arrays[f'h{h}_knockoff_scores'] = np.full((2, p), np.nan)
        for kind_index, (kind, angles, v) in enumerate((('pilot', witness, covariance), ('oracle', oracle, np.zeros_like(covariance)))):
            second = harmonic_features(angles, h)
            phase = time.perf_counter()
            try:
                zero, hc3, leverage = hc_pair(first, second, y, t, v, m, 2 * h,
                                            original_config['level'], original_config['ridge_penalty'])
                arrays[f'h{h}_contrasts'][kind_index] = zero.contrasts
                arrays[f'h{h}_leverage_min_mean_max'][kind_index] = leverage.min(), leverage.mean(), leverage.max()
                for hc_index, (hc, result) in enumerate(zip(config['covariances'], (zero, hc3))):
                    name = f'h{h}_{hc}_wald_{kind}_by'
                    arrays[f'h{h}_oracle_components'][hc_index, kind_index] = result.oracle_variance_component
                    chosen = result.selected
                    masks[name][chosen] = True
                    pvalues[name] = result.pvalues
                    failed[name] = False
                    if kind == 'pilot':
                        name_naive = f'h{h}_{hc}_wald_naive_by'
                        masks[name_naive][result.naive_selected] = True
                        pvalues[name_naive] = result.naive_pvalues
                        failed[name_naive] = False
                        arrays[f'h{h}_total_variance'][hc_index] = result.total_variance
                        arrays[f'h{h}_pilot_derivative'] = zero.pilot_derivative
            except Exception as error:
                errors.append({'stage': f'h{h}_{kind}_wald', 'error': repr(error)})
            times[f'h{h}_{kind}_wald'] = time.perf_counter() - phase
            phase = time.perf_counter()
            try:
                score = (knockstat(x, angles, y, penalty=original_config['ridge_penalty']) if h == 1 else
                         block_norm_scores(first, second, y, 2 * h, original_config['ridge_penalty']))
                name = f'h{h}_knockoff_plus_{kind}'
                masks[name][select(score, original_config['level'])] = True
                arrays[f'h{h}_knockoff_scores'][kind_index] = score
                failed[name] = False
            except Exception as error:
                errors.append({'stage': f'h{h}_{kind}_knockoff', 'error': repr(error)})
            times[f'h{h}_{kind}_knockoff'] = time.perf_counter() - phase
    # Exact reference agreement is required, and not converted to abstention.
    for previous_index, kind in enumerate(('pilot', 'naive', 'oracle')):
        name = f'h1_hc0_wald_{kind}_by'
        if failed[name]:
            raise RuntimeError('Cannot validate H1 against the source study')
        np.testing.assert_array_equal(pvalues[name], retained['pvalues'][previous_index])
    np.testing.assert_array_equal(arrays['h1_knockoff_scores'], retained['knockoff_scores'])
    entries = []
    for method_index, name in enumerate(names):
        mask = masks[name]
        arrays['selected_mask'][method_index] = mask
        item = {'method': name, 'selected': np.flatnonzero(mask).tolist(), 'failed': failed[name],
                'fdp': float(mask[null].sum() / max(1, mask.sum())),
                'power': float(mask[active].sum() / max(1, len(active))), 'empty': not mask.any()}
        if name in pvalues:
            item['marginal_null_rejection'] = float((pvalues[name][null] <= original_config['marginal_level']).mean())
        entries.append(item)
    arrays['pvalues'] = np.array(list(pvalues.values()))
    record = {'replicate': index, 'config_id': config['id'], 'method_order': names,
              'pvalue_order': list(pvalues), 'active': active, 'methods': entries, 'errors': errors,
              'source_json_sha256': digest(path), 'source_arrays_sha256': previous['arrays_sha256'],
              'source_version': config['source_version'], 'new_independent_datasets': 0,
              'reference_h1_exact_match': True, 'times_seconds': times,
              'elapsed_seconds': time.perf_counter() - started,
              'max_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return serial(record), arrays


def aggregate(config: dict, output: Path) -> dict:
    records = [read_record(output / f'rep_{i:04d}.json') for i in range(config['repetitions'])]
    names = method_names(config)
    metrics, rows, contrasts = {}, [], []
    for i, (record, arrays) in enumerate(records):
        if record['replicate'] != i or not record['reference_h1_exact_match']:
            raise RuntimeError('Record/reference verification mismatch')
        for j, method in enumerate(record['methods']):
            if method['failed']:
                continue
            if '_wald_' in method['method']:
                chosen = by_select(arrays['pvalues'][record['pvalue_order'].index(method['method'])], config['source_config']['level'])
            else:
                h = int(method['method'][1])
                kind = 0 if method['method'].endswith('pilot') else 1
                chosen = select(arrays[f'h{h}_knockoff_scores'][kind], config['source_config']['level'])
            if chosen.tolist() != method['selected']:
                raise RuntimeError('Selection recomputation mismatch')
            np.testing.assert_array_equal(arrays['selected_mask'][j], np.isin(np.arange(config['source_config']['p']), chosen))
    for j, name in enumerate(names):
        failed = np.array([r[0]['methods'][j]['failed'] for r in records])
        row = {'configuration': config['id'], 'method': name, 'repetitions': len(records),
               'n': config['source_config']['n'], 'm': config['source_config']['m'],
               'failures': int(failed.sum()), 'failure_rate': float(failed.mean())}
        for key in ['fdp', 'power', 'empty'] + (['marginal_null_rejection'] if '_wald_' in name else []):
            values = np.array([r[0]['methods'][j][key] for r in records], float)
            row[key], row[key + '_mcse'] = moments(values)
            row[key + '_failure_lower'] = float((values * ~failed).sum() / len(records))
            row[key + '_failure_upper'] = float(((values * ~failed).sum() + failed.sum()) / len(records))
            metrics[name, key] = values
        if name == 'h2_hc3style_wald_pilot_by':
            original = config['source_config']
            row['fdr_diagnostic'] = row['fdp'] <= original['level'] + original['gate_mcse_multiplier'] * row['fdp_mcse']
            row['marginal_diagnostic'] = row['marginal_null_rejection'] <= original['marginal_level'] + original['gate_mcse_multiplier'] * row['marginal_null_rejection_mcse']
            row['calibration_diagnostic_complete'] = not failed.any()
        rows.append(row)
    for left, right in itertools.combinations(names, 2):
        for key in ('fdp', 'power', 'empty'):
            mean, mcse = moments(metrics[left, key] - metrics[right, key])
            contrasts.append({'configuration': config['id'], 'left': left, 'right': right, 'metric': key,
                              'difference_left_minus_right': mean, 'mcse': mcse})
    timing = {name: {'total_seconds': sum(r[0]['times_seconds'].get(name, 0.) for r in records)}
              for name in sorted(set().union(*(r[0]['times_seconds'] for r in records)))}
    result = serial({'config': config, 'summary': rows, 'paired': contrasts, 'timing': timing,
                     'failed_repetitions': [r[0]['replicate'] for r in records if r[0]['errors']],
                     'independent_datasets_added': 0, 'reanalysed_datasets': len(records),
                     'max_rss_bytes_macos': max(r[0]['max_rss_bytes_macos'] for r in records),
                     'scope': config['scope']})
    path = output / 'summary.json'
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise RuntimeError('Existing summary differs; preserved')
    else:
        atomic_json(path, result)
    return result


def run(config_path: Path, aggregate_only: bool, replay: bool) -> None:
    base = json.loads(config_path.read_text())
    started = time.perf_counter()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    log = ROOT / 'results/statistical_papers' / base['version'] / f'invocation_{stamp}.json'
    completed = []
    try:
        with threadpool_limits(limits=1):
            for case in base['cases']:
                config, source, output = prepare(base, case)
                case_started, new = time.perf_counter(), 0
                for index in range(base['repetitions']):
                    path = output / f'rep_{index:04d}.json'
                    if path.exists():
                        continue
                    if aggregate_only:
                        raise FileNotFoundError(path)
                    if path.with_suffix('.npz').exists():
                        raise RuntimeError(f'Orphaned NPZ, inspect before resuming: {path}')
                    record, arrays = repetition(config, source, index)
                    atomic_arrays(path.with_suffix('.npz'), arrays)
                    record['arrays_sha256'] = digest(path.with_suffix('.npz'))
                    atomic_json(path, record)
                    new += 1
                    if (index + 1) % 100 == 0:
                        print(case, index + 1, 'saved', flush=True)
                        atomic_json(log, {'status': 'running', 'completed': completed, 'case': case,
                                         'saved_index': index, 'elapsed_seconds': time.perf_counter() - started})
                summary = aggregate(config, output)
                replays = []
                if replay:
                    for index in base['replay_indices']:
                        stored, retained = read_record(output / f'rep_{index:04d}.json')
                        generated, values = repetition(config, source, index)
                        excluded = {'times_seconds', 'elapsed_seconds', 'arrays_sha256', 'max_rss_bytes_macos'}
                        if {k: v for k, v in stored.items() if k not in excluded} != {k: v for k, v in generated.items() if k not in excluded}:
                            raise RuntimeError(f'Replay record mismatch at {case}:{index}')
                        for key in values:
                            np.testing.assert_array_equal(values[key], retained[key])
                        replays.append(index)
                item = {'case': case, 'new_analyses': new, 'reanalysed_datasets': base['repetitions'],
                        'new_independent_datasets': 0, 'replayed_indices': replays,
                        'elapsed_seconds_including_aggregation_and_replay': time.perf_counter() - case_started,
                        'summary_sha256': digest(output / 'summary.json'),
                        'failed_repetitions': summary['failed_repetitions']}
                completed.append(item)
                print(json.dumps(item), flush=True)
        atomic_json(log, {'status': 'complete', 'completed': completed, 'elapsed_seconds': time.perf_counter() - started,
                         'scope': base['scope']})
    except BaseException as error:
        atomic_json(log, {'status': 'incomplete', 'completed': completed, 'elapsed_seconds': time.perf_counter() - started,
                         'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    print(log, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/statistical_papers/sp06_3_harmonics_hc.json')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    run(args.config.resolve(), args.aggregate_only, args.replay)
