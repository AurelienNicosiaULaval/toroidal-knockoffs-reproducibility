"""SP14: H=2 reanalysis of nine archived exact-iid SP12 datasets.

This is a bounded implementation diagnostic, not a Monte Carlo FDR study.
It creates no original, pilot, witness or response draws. No project regression,
selector or population-moment routine is imported.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import resource
import time
import uuid

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import i0e, ive
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/statistical_papers/SP14_PROTOCOLE_FR_20261010.md'
G, KAPPA, RIDGE, LEVEL = .2, .04, 1., .1


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def harmonics(angle: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(angle), np.sin(angle),
                     np.cos(2 * angle), np.sin(2 * angle)], axis=-1)


def conditional_moments(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and raw second moments of [cos,sin,cos2,sin2] under VM(field)."""
    radius = np.linalg.norm(field, axis=1)
    phase = np.arctan2(field[:, 1], field[:, 0])
    orders = np.arange(5)
    ratios = ive(orders[None, :], radius[:, None]) / i0e(radius[:, None])
    cosine = ratios * np.cos(phase[:, None] * orders)
    sine = ratios * np.sin(phase[:, None] * orders)
    mean = np.column_stack([cosine[:, 1], sine[:, 1], cosine[:, 2], sine[:, 2]])
    second = np.empty((len(field), 4, 4))
    indices = [(1, 'cos'), (1, 'sin'), (2, 'cos'), (2, 'sin')]
    for a, (l, kind_a) in enumerate(indices):
        for b, (k, kind_b) in enumerate(indices):
            if kind_a == kind_b == 'cos':
                second[:, a, b] = .5 * (cosine[:, abs(l-k)] + cosine[:, l+k])
            elif kind_a == kind_b == 'sin':
                second[:, a, b] = .5 * (cosine[:, abs(l-k)] - cosine[:, l+k])
            elif kind_a == 'cos':
                second[:, a, b] = .5 * (sine[:, k+l] + np.sign(k-l)*sine[:, abs(k-l)])
            else:
                second[:, a, b] = .5 * (sine[:, l+k] + np.sign(l-k)*sine[:, abs(l-k)])
    return mean, second


def population(t: float, angular: int, gaussian: int) -> dict[str, np.ndarray | float]:
    """Integrate originals, Gaussian fields and conditional witness moments."""
    nodes = np.arange(angular) * (2*np.pi/angular)
    root, increment = np.meshgrid(nodes, nodes, indexing='ij')
    root, increment = root.ravel(), increment.ravel()
    first = root + increment
    original = np.column_stack([harmonics(first), harmonics(root)])
    response = np.cos(2 * first)
    weight = np.exp(KAPPA*np.cos(root) + G*np.cos(increment))
    weight /= angular**2 * np.exp(KAPPA+G) * i0e(KAPPA) * i0e(G)
    raw_mass = float(weight.sum())
    weight /= raw_mass
    z, wz = hermgauss(gaussian)
    z, wz = np.sqrt(2)*z, wz/np.sqrt(np.pi)
    mean, raw, dz = np.zeros(16), np.zeros((16, 16)), np.zeros(16)
    natural = t * (original[:, :2] + original[:, 4:6])
    for i, zi in enumerate(z):
        for j, zj in enumerate(z):
            common = natural + np.sqrt(t)*np.array([zi, zj])
            m1, s1 = conditional_moments(common)
            shifted = common.copy()
            shifted[:, 0] += KAPPA
            m2, s2 = conditional_moments(shifted)
            dm = np.column_stack([original, m1, m2])
            w = weight*wz[i]*wz[j]
            mean += w @ dm
            # Integrate the response cross moment independently of raw covariance.
            dz += (w*response) @ dm
            raw += (dm.T*w) @ dm
            for start, m, s in [(8, m1, s1), (12, m2, s2)]:
                correction = s - m[:, :, None]*m[:, None, :]
                raw[start:start+4, start:start+4] += np.einsum('n,nab->ab', w, correction)
    gamma = raw - np.outer(mean, mean)
    response_mean = float(weight @ response)
    cross = dz - response_mean*mean
    beta = np.linalg.solve(gamma, cross)
    return {'mean': mean, 'gamma': gamma, 'cross': cross, 'beta': beta,
            'raw_mass': raw_mass, 'angular': angular, 'gaussian': gaussian}


def scores(beta: np.ndarray) -> np.ndarray:
    b = beta.reshape(-1, 16)
    return np.column_stack([
        np.linalg.norm(b[:, :4], axis=1)-np.linalg.norm(b[:, 8:12], axis=1),
        np.linalg.norm(b[:, 4:8], axis=1)-np.linalg.norm(b[:, 12:16], axis=1)]).ravel()


def select(w: np.ndarray) -> np.ndarray:
    for threshold in np.unique(np.abs(w[w != 0])):
        if (1+np.count_nonzero(w <= -threshold))/max(1, np.count_nonzero(w >= threshold)) <= LEVEL:
            return w >= threshold
    return np.zeros(len(w), dtype=bool)


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(cache: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    out = ROOT/'results/statistical_papers'/('sp14_represented_'+stamp)
    out.mkdir(parents=True)
    record = {'status': 'running', 'version': 'SP14',
              'Gaussian_KL_implementation': 'Cholesky log determinants, diagnostic revision 2',
              'new_independent_datasets': 0,
              'reanalysed_archived_datasets': 0, 'input_cache': str(cache),
              'scope': 'Nine matched diagnostics, no Monte Carlo FDR estimate',
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'configuration': {'g': G, 'known_kappa': KAPPA, 'ridge': RIDGE,
                                'level': LEVEL, 'harmonics': 2, 'noise_variance': 1.}}
    receipt = out/'receipt.json'
    receipt.write_text(json.dumps(record, indent=2)+'\n')
    print(str(out), flush=True)
    started = time.perf_counter()
    try:
        # Buffer copies: never fcopyfile on an iCloud/File Provider source.
        for source, name in [(Path(__file__), 'executed_source.py'), (PROTOCOL, 'protocol.md')]:
            data = source.read_bytes()
            (out/name).write_bytes(data)
            record[name+'_sha256'] = digest(data)
        old_data = (cache/'original_receipt.json').read_bytes()
        (out/'input_receipt.json').write_bytes(old_data)
        old_receipt = json.loads(old_data)
        assert old_receipt['status'] == 'complete'
        record['input_receipt_sha256'] = digest(old_data)
        inputs = {name: sha for name, sha in old_receipt['files_sha256'].items() if name.endswith('.npz')}
        assert len(inputs) == 9
        record['inputs_sha256'] = inputs
        receipt.write_text(json.dumps(record, indent=2)+'\n')
        quadrature_rows, arrays = [], {}
        beta_pair = np.zeros(16)
        beta_pair[2] = 1.
        swap = np.arange(16)
        swap[4:8], swap[12:16] = np.arange(12, 16), np.arange(4, 8)
        population_started = time.perf_counter()
        for t in [.18, .20, .22]:
            coarse, fine = population(t, 24, 16), population(t, 32, 24)
            difference = max(float(np.max(np.abs(coarse[key]-fine[key])))
                             for key in ['mean', 'gamma', 'cross', 'beta'])
            if difference > 1e-8:
                refined = population(t, 48, 32)
                difference = max(float(np.max(np.abs(fine[key]-refined[key])))
                                 for key in ['mean', 'gamma', 'cross', 'beta'])
                fine = refined
            if difference > 1e-8:
                raise ArithmeticError(f'Unresolved population quadrature at t={t}: {difference}')
            target_error = float(np.max(np.abs(fine['beta']-beta_pair)))
            symmetry_error = float(np.max(np.abs(fine['gamma']-fine['gamma'][np.ix_(swap, swap)])))
            assert target_error < 1e-8
            assert float(np.linalg.eigvalsh(fine['gamma'])[0]) > 0
            if t == G:
                assert symmetry_error < 1e-8
                oracle_gamma = fine['gamma'].copy()
            else:
                assert symmetry_error > 1e-5
            key = f't_{t:.2f}'.replace('.', '_')
            for name in ['mean', 'gamma', 'cross', 'beta']:
                arrays[key+'_'+name] = fine[name]
            row = {'working_g': t, 'maximum_refinement_difference': difference,
                   'angular': fine['angular'], 'gaussian': fine['gaussian'],
                   'represented_target_error': target_error, 'null_swap_error': symmetry_error,
                   'minimum_eigenvalue': float(np.linalg.eigvalsh(fine['gamma'])[0]),
                   'raw_mass_error': abs(float(fine['raw_mass'])-1)}
            quadrature_rows.append(row)
            print('population', json.dumps(row), flush=True)
        np.savez_compressed(out/'population.npz', **arrays)
        save_csv(out/'population_checks.csv', quadrature_rows)
        record['population_seconds'] = time.perf_counter()-population_started
        rows = []
        for name in sorted(inputs):
            case_started = time.perf_counter()
            data = (cache/name).read_bytes()
            assert digest(data) == inputs[name], f'Input hash mismatch: {name}'
            with np.load(cache/name) as stored:
                n, k = stored['root'].shape
                m = stored['pilot_increment'].shape[0]
                angles = np.stack([stored['original_first'], stored['root'],
                                   stored['witness_first'], stored['witness_second']], axis=2)
                design = harmonics(angles).reshape(n, 16*k)
                dc = design - design.mean(axis=0)
                y, noise = stored['y'], stored['noise']
                gram = dc.T@dc/n
                penalized = gram + RIDGE/n*np.eye(16*k)
                beta = np.linalg.solve(penalized, dc.T@(y-y.mean())/n)
                target = np.tile(beta_pair, k)
                identity_rhs = np.linalg.solve(penalized, dc.T@noise/n-RIDGE/n*target)
                identity_error = float(np.max(np.abs(beta-target-identity_rhs)))
                assert identity_error < 1e-10
                mean = -RIDGE/np.sqrt(n)*np.linalg.solve(penalized, target)
                inverse = np.linalg.inv(penalized)
                covariance = inverse@gram@inverse  # sigma^2=1, conditional sqrt(n) error
                gamma0 = np.kron(np.eye(k), oracle_gamma)
                covariance0 = np.kron(np.eye(k), np.linalg.inv(oracle_gamma))
                # Positive definite Gaussian covariances: Cholesky avoids the
                # platform slogdet warnings observed in diagnostic revision 1.
                logdet = 2*float(np.log(np.diag(np.linalg.cholesky(covariance))).sum())
                logdet0 = 2*float(np.log(np.diag(np.linalg.cholesky(covariance0))).sum())
                kl = .5*(float(mean@gamma0@mean)+float(np.trace(gamma0@covariance))-16*k+logdet0-logdet)
                assert kl >= -1e-10
                kl = max(0., float(kl))
                w, chosen = scores(beta), select(scores(beta))
                total, false = int(chosen.sum()), int(chosen[1::2].sum())
                h1 = stored['selected']
                response_error = float(np.max(np.abs(y-np.cos(2*stored['original_first']).sum(axis=1)-noise)))
                assert response_error < 1e-10
                row = {'K': k, 'n': n, 'm': m, 'working_g': float(stored['working_g']),
                       'K_squared_over_n': k*k/n, 'm_over_n': m/n,
                       'ridge_identity_error': identity_error, 'response_identity_error': response_error,
                       'minimum_empirical_eigenvalue': float(np.linalg.eigvalsh(gram)[0]),
                       'covariance_frobenius_error': float(np.linalg.norm(covariance-covariance0, 'fro')),
                       'Gaussian_KL_to_oracle': kl, 'conditional_TV_upper_bound': min(1., np.sqrt(kl/2)),
                       'selected': total, 'false_selected': false, 'FDP': false/max(1, total),
                       'power': float(chosen[::2].mean()), 'empty': total == 0,
                       'minimum_active_score': float(w[::2].min()), 'maximum_null_magnitude': float(np.abs(w[1::2]).max()),
                       'H1_selected': int(h1.sum()), 'H1_power': float(h1[::2].mean()),
                       'seconds': time.perf_counter()-case_started}
                np.savez_compressed(out/name, beta=beta, target=target, scores=w, selected=chosen,
                    gram=gram, conditional_mean=mean, conditional_covariance=covariance,
                    oracle_covariance=covariance0, oracle_gamma_pair=oracle_gamma)
                rows.append(row)
            save_csv(out/'individual_diagnostics.csv', rows)
            record['reanalysed_archived_datasets'] = len(rows)
            receipt.write_text(json.dumps(record, indent=2)+'\n')
            print('reanalysis', json.dumps(row), flush=True)
        assert len(rows) == 9
        record['status'] = 'complete'
        record['peak_process_bytes_macos'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception as error:
        record['status'] = 'incomplete'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['elapsed_seconds'] = time.perf_counter()-started
        record['files_sha256'] = {path.name: digest(path.read_bytes()) for path in out.iterdir()
                                if path.is_file() and path.name != 'receipt.json'}
        receipt.write_text(json.dumps(record, indent=2)+'\n')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.cache)
