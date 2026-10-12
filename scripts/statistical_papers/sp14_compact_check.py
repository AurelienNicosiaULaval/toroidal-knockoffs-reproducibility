"""Independent numerical checks of SP14 outputs, without rerunning simulations.

The producer's von Mises moments are tested against direct angular integration.
Ridge is independently solved by augmented QR with reversed feature ordering.
Selectors and Gaussian KL are independently recomputed for all nine datasets.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import uuid

import numpy as np
from scipy.linalg import lstsq, solve_triangular
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def independent_filter(w: np.ndarray, q: float) -> np.ndarray:
    candidates = sorted(set(abs(float(value)) for value in w if value != 0))
    for t in candidates:
        negative = sum(float(value) <= -t for value in w)
        positive = sum(float(value) >= t for value in w)
        if (1+negative)/max(positive, 1) <= q:
            return np.asarray([float(value) >= t for value in w], dtype=bool)
    return np.asarray([False]*len(w))


def direct_vm(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(4096)*(2*np.pi/4096)
    feature = np.column_stack([np.cos(theta), np.sin(theta), np.cos(2*theta), np.sin(2*theta)])
    log_weight = field[0]*np.cos(theta)+field[1]*np.sin(theta)
    weight = np.exp(log_weight-log_weight.max())
    weight /= weight.sum()
    return weight@feature, (feature.T*weight)@feature


def gaussian_kl(mean: np.ndarray, covariance: np.ndarray, reference: np.ndarray) -> float:
    chol = np.linalg.cholesky(reference)
    left = solve_triangular(chol, covariance, lower=True)
    whitened = solve_triangular(chol, left.T, lower=True).T
    whitened = (whitened+whitened.T)/2
    values = np.linalg.eigvalsh(whitened)
    if not np.all(values > 0):
        raise ArithmeticError('Nonpositive whitened Gaussian covariance')
    wm = solve_triangular(chol, mean, lower=True)
    return .5*float(wm@wm + np.sum(values-1-np.log(values)))


def main(result: Path, cache: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    out = ROOT/'results/statistical_papers'/('sp14_independent_'+stamp)
    out.mkdir(parents=True)
    receipt = out/'receipt.json'
    record = {'status': 'running', 'producer_folder': str(result), 'cache': str(cache),
              'new_independent_datasets': 0, 'checked_datasets': 0, 'QR_checked_datasets': 0,
              'scope': 'Compact package: nine saved coefficient/selector checks and three raw-input QR checks; six large raw designs omitted'}
    receipt.write_text(json.dumps(record, indent=2)+'\n')
    print(str(out), flush=True)
    started = time.perf_counter()
    try:
        data = Path(__file__).read_bytes()
        (out/'executed_source.py').write_bytes(data)
        record['source_sha256'] = sha(data)
        producer_receipt = (result/'receipt.json').read_bytes()
        producer = json.loads(producer_receipt)
        assert producer['status'] == 'complete'
        (out/'producer_receipt.json').write_bytes(producer_receipt)
        record['producer_receipt_sha256'] = sha(producer_receipt)
        for name, expected in producer['files_sha256'].items():
            assert sha((result/name).read_bytes()) == expected, name
        # Only the analytic moment function is imported, to test it against a
        # separate angular quadrature; regression and selection are not imported.
        spec = importlib.util.spec_from_file_location('frozen_sp14_producer', result/'executed_source.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fields = np.array([[0., 0.], [1., 0.], [-.3, .7], [8., -4.], [1e-10, -1e-10]])
        analytic_mean, analytic_second = module.conditional_moments(fields)
        moment_errors = []
        for i, field in enumerate(fields):
            direct_mean, direct_second = direct_vm(field)
            error = max(float(np.max(np.abs(direct_mean-analytic_mean[i]))),
                        float(np.max(np.abs(direct_second-analytic_second[i]))))
            assert error < 1e-10
            moment_errors.append(error)
        record['direct_VM_maximum_error'] = max(moment_errors)
        rows = list(csv.DictReader((result/'individual_diagnostics.csv').open()))
        assert len(rows) == 9
        checks = []
        for row in rows:
            k, n, m = int(row['K']), int(row['n']), int(row['m'])
            name = f'K{k}_n{n}_m{m}.npz'
            if n == 2000:
                assert sha((cache/name).read_bytes()) == producer['inputs_sha256'][name]
            with np.load(result/name) as saved:
                chosen = independent_filter(saved['scores'], .1)
                assert np.array_equal(chosen, saved['selected'])
                total, false = int(chosen.sum()), int(sum(chosen[1::2]))
                assert total == int(row['selected']) and false == int(row['false_selected'])
                kl = gaussian_kl(saved['conditional_mean'], saved['conditional_covariance'], saved['oracle_covariance'])
                kl_error = abs(kl-float(row['Gaussian_KL_to_oracle']))
                assert kl_error < 1e-10
                check = {'K': k, 'n': n, 'm': m, 'selector_identical': True,
                         'Gaussian_KL_difference': kl_error, 'QR_checked': n == 2000}
                if n == 2000:
                    with np.load(cache/name) as raw:
                        # Reversed angles AND harmonic order, constructed separately.
                        columns = []
                        for j in range(k):
                            for angle in [raw['witness_second'][:, j], raw['witness_first'][:, j],
                                          raw['root'][:, j], raw['original_first'][:, j]]:
                                columns.extend([np.sin(2*angle), np.cos(2*angle), np.sin(angle), np.cos(angle)])
                        design = np.column_stack(columns)
                        dc = design-design.mean(axis=0)
                        yc = raw['y']-raw['y'].mean()
                        fit, _, rank, _ = lstsq(np.vstack([dc, np.eye(16*k)]),
                                              np.concatenate([yc, np.zeros(16*k)]),
                                              lapack_driver='gelsy')
                        assert rank == 16*k
                        permutation = np.concatenate([16*j+np.arange(15, -1, -1) for j in range(k)])
                        canonical = np.empty_like(fit)
                        canonical[permutation] = fit
                        beta_error = float(np.max(np.abs(canonical-saved['beta'])))
                        assert beta_error < 1e-10
                        # A wrongly normalized penalty must be detected by the
                        # correct identity and by the independent QR coefficients.
                        wrong = np.linalg.solve(saved['gram']+np.eye(16*k),
                            (saved['gram']+np.eye(16*k)/n)@saved['beta'])
                        wrong_error = float(np.max(np.abs(wrong-canonical)))
                        assert wrong_error > 1e-4
                        original1 = raw['original_first']
                        angles_h1 = np.stack([original1, raw['root'], raw['witness_first'], raw['witness_second']], axis=2)
                        h1 = np.stack([np.cos(angles_h1), np.sin(angles_h1)], axis=3).reshape(n, 8*k)
                        h1 -= h1.mean(axis=0)
                        noiseless = np.cos(2*original1).sum(axis=1)
                        noiseless -= noiseless.mean()
                        fitted_h1, _, _, _ = lstsq(h1, noiseless, lapack_driver='gelsy')
                        nonrepresentation = float(np.linalg.norm(noiseless-h1@fitted_h1)/np.sqrt(n))
                        assert nonrepresentation > .1
                        check.update({'QR_maximum_coefficient_error': beta_error,
                                      'wrong_penalty_control_error': wrong_error,
                                      'H1_unrepresented_mean_RMS': nonrepresentation})
                        record['QR_checked_datasets'] += 1
                checks.append(check)
                record['checked_datasets'] += 1
                receipt.write_text(json.dumps(record, indent=2)+'\n')
                print(json.dumps(check), flush=True)
        assert record['checked_datasets'] == 9 and record['QR_checked_datasets'] == 3
        fields = ['K', 'n', 'm', 'selector_identical', 'Gaussian_KL_difference', 'QR_checked',
                  'QR_maximum_coefficient_error', 'wrong_penalty_control_error', 'H1_unrepresented_mean_RMS']
        with (out/'independent_checks.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(checks)
        record['negative_controls'] = {'wrong_penalty_detected': 3, 'H1_nonrepresentation_detected': 3}
        record['status'] = 'complete'
    except Exception as error:
        record['status'] = 'incomplete'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        record['files_sha256'] = {path.name: sha(path.read_bytes()) for path in out.iterdir()
                                if path.is_file() and path.name != 'receipt.json'}
        receipt.write_text(json.dumps(record, indent=2)+'\n')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.result, args.cache)
