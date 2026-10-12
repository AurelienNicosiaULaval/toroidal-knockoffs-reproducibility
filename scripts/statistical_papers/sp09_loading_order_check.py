#!/usr/bin/env python3
"""Independent known-law loading checks; no project sampler or response data."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from scipy.integrate import quad
from scipy.special import ive
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def vectors(angles: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def root_psd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, basis = np.linalg.eigh(matrix)
    if values.min() < -1e-11 * max(1.0, np.linalg.norm(matrix, 2)):
        raise ValueError('The specified loading is not positive semidefinite')
    return (basis * np.sqrt(np.maximum(values, 0))) @ basis.T, values


def posterior_moments(field: np.ndarray, harmonic: int) -> np.ndarray:
    radius = np.linalg.norm(field, axis=-1)
    angle = np.arctan2(field[..., 1], field[..., 0])
    amplitude = ive(harmonic, radius) / ive(0, radius)
    return amplitude[..., None] * np.stack(
        [np.cos(harmonic * angle), np.sin(harmonic * angle)], axis=-1)


def rice_correlation(loading: float, harmonic: int) -> tuple[float, float, float]:
    """Radial integration in units sqrt(loading), stabilized by scaled Bessel."""
    def radial(y: float) -> float:
        radius = np.sqrt(loading) * y
        return y * np.exp(-0.5 * (y - np.sqrt(loading)) ** 2) * ive(0, radius)

    normalization, norm_error = quad(radial, 0, np.inf, epsabs=1e-11, epsrel=1e-11)
    def integrand(y: float) -> float:
        radius = np.sqrt(loading) * y
        ratio = ive(harmonic, radius) / ive(0, radius)
        return radial(y) * ratio * ratio

    value, error = quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-11)
    if abs(normalization - 1) > 1e-9:
        raise ArithmeticError('Rice density did not integrate to one')
    return value, error + norm_error, normalization


def average_and_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def iid_rejection(rng: np.random.Generator, subjects: int,
                  fields: np.ndarray, matrices: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Global rejection from uniform angles, with a deterministic safe envelope."""
    envelope = np.linalg.norm(fields, axis=1).sum() + sum(np.linalg.norm(a, 2) for a in matrices)
    chunks: list[np.ndarray] = []
    kept = proposals = accepted = 0
    while kept < subjects:
        batch = max(20000, min(500000, 6 * (subjects - kept)))
        angles = rng.uniform(-np.pi, np.pi, (batch, 3))
        v = vectors(angles)
        log_density = np.einsum('njk,jk->n', v, fields)
        for leaf, matrix in enumerate(matrices, start=1):
            log_density += np.einsum('ni,ij,nj->n', v[:, 0], matrix, v[:, leaf])
        if np.max(log_density) > envelope + 1e-12:
            raise ArithmeticError('Analytic rejection envelope was exceeded')
        good = angles[np.log(rng.uniform(size=batch)) <= log_density - envelope]
        chunks.append(good)
        kept += len(good); proposals += batch; accepted += len(good)
    return np.concatenate(chunks)[:subjects], {
        'subjects_used': subjects, 'proposals': proposals, 'accepted_total': accepted,
        'acceptance': accepted / proposals, 'log_envelope': float(envelope),
        'exact_iid': True, 'MCMC': False}


def statistic(v: np.ndarray, second: np.ndarray) -> np.ndarray:
    edge_parts = [np.einsum('ni,nj->nij', v[:, 0], v[:, j]).reshape(len(v), 4)
                  for j in [1, 2]]
    complex_mean = v[..., 0] + 1j * v[..., 1]
    triple = complex_mean.prod(axis=1)
    return np.column_stack([v.reshape(len(v), 6), *edge_parts,
                            second.reshape(len(v), 6), triple.real, triple.imag])


def conditional_statistic(field: np.ndarray) -> np.ndarray:
    return statistic(posterior_moments(field, 1), posterior_moments(field, 2))


def matrix_identity(first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray | float]:
    """Check E[M1(M2-M1)^T+(M2-M1)M1^T]=0, without storing n*k*k."""
    size = first.shape[1]
    sums = np.zeros((size, size)); squares = np.zeros_like(sums)
    for start in range(0, len(first), 2000):
        a = first[start:start + 2000]; change = second[start:start + 2000] - a
        row = np.einsum('ni,nj->nij', a, change)
        row = row + row.transpose(0, 2, 1)
        sums += row.sum(axis=0); squares += np.square(row).sum(axis=0)
    n = len(first); mean = sums / n
    variance = np.maximum((squares - n * mean * mean) / (n - 1), 0)
    se = np.sqrt(variance / n)
    z = np.abs(mean) / np.maximum(se, 1e-14)
    change = second - first
    gram = change.T @ change / n
    covariance_change = (second.T @ second - first.T @ first) / n
    return {'identity_mean': mean, 'identity_mcse': se, 'identity_max_z': float(z.max()),
            'positive_gram': gram, 'direct_second_moment_change': covariance_change,
            'gram_min_eigenvalue': float(np.linalg.eigvalsh(gram).min())}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, default=100000)
    args = parser.parse_args()
    if args.subjects < 1000:
        raise ValueError('At least 1000 subjects are needed for these Monte Carlo diagnostics')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    out = ROOT / 'results/statistical_papers' / ('sp09_loading_' + stamp)
    out.mkdir(parents=True)
    protocol = ROOT / 'docs/statistical_papers/SP09_PROTOCOLE_FR_20261010.md'
    shutil.copyfile(__file__, out / 'executed_source.py')
    shutil.copyfile(protocol, out / 'protocol.md')
    receipt = {'status': 'running', 'source_sha256': sha(Path(__file__)),
               'protocol_sha256': sha(protocol), 'subjects_per_case': args.subjects,
               'new_selection_datasets': 0, 'response_used': False,
               'project_sampler_imported': False, 'seed': 2026101009,
               'scope': 'known-law quadrature and iid coupling diagnostics, not FDR simulations'}
    started = time.perf_counter()
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    try:
        seeds = np.random.SeedSequence(2026101009).spawn(12)
        rows: list[dict] = []
        with threadpool_limits(limits=1):
            for index, loading in enumerate([.125, .5, 1., 4., 16.]):
                rng = np.random.default_rng(seeds[index])
                x = rng.uniform(-np.pi, np.pi, args.subjects)
                field = loading * vectors(x) + np.sqrt(loading) * rng.normal(size=(args.subjects, 2))
                w = rng.vonmises(np.arctan2(field[:, 1], field[:, 0]), np.linalg.norm(field, axis=1))
                np.savez_compressed(out / f'univariate_{index}.npz', original=x, field=field, witness=w)
                for harmonic in [1, 2, 3]:
                    target, error, norm = rice_correlation(loading, harmonic)
                    estimate, se = average_and_se(np.cos(harmonic * (x - w)))
                    rows.append({'loading': loading, 'harmonic': harmonic,
                                 'quadrature_correlation': target, 'quadrature_error': error,
                                 'normalization': norm, 'mc_correlation': float(estimate),
                                 'mcse': float(se), 'difference_z': float((estimate - target) / se)})
            write_csv(out / 'univariate.csv', rows)
            for harmonic in [1, 2, 3]:
                values = [row['quadrature_correlation'] for row in rows if row['harmonic'] == harmonic]
                if np.min(np.diff(values)) <= 0:
                    raise ArithmeticError('Quadrature did not exhibit strict loading order')
            if max(abs(row['difference_z']) for row in rows) > 5:
                raise ArithmeticError('Univariate diagnostic exceeds five Monte Carlo standard errors')
            fields = np.array([.4 * np.array([np.cos(.3), np.sin(.3)]),
                               .2 * np.array([np.cos(-.5), np.sin(-.5)]),
                               .25 * np.array([np.cos(1.1), np.sin(1.1)])])
            matrices = [rotation(.2) @ np.diag([.55, .25]) @ rotation(-.3).T,
                        rotation(.2) @ np.diag([.55, .35]) @ rotation(.6).T]
            j = np.zeros((6, 6))
            for leaf, matrix in enumerate(matrices, start=1):
                j[:2, 2*leaf:2*leaf+2] = matrix
                j[2*leaf:2*leaf+2, :2] = matrix.T
            loads = np.array([1.1, .55, .55]); balanced = np.repeat(.55 * np.sqrt(2), 3)
            a1 = j + np.diag(np.repeat(loads, 2)); a2 = a1 + .4 * np.eye(6)
            ap = j + np.diag(np.repeat(balanced, 2))
            root1, eigen1 = root_psd(a1); rootp, eigenp = root_psd(ap)
            if np.count_nonzero(eigen1 > 1e-10) != 5:
                raise ArithmeticError('Singular baseline was not the intended rank-five case')
            x, rejection = iid_rejection(np.random.default_rng(seeds[5]), args.subjects, fields, matrices)
            v = vectors(x); s = v.reshape(args.subjects, 6)
            rng = np.random.default_rng(seeds[6])
            g1 = s @ a1 + rng.normal(size=s.shape) @ root1
            gr = .4 * s + np.sqrt(.4) * rng.normal(size=s.shape)
            g2 = g1 + gr
            gp = s @ ap + np.random.default_rng(seeds[7]).normal(size=s.shape) @ rootp
            f1 = (g1 + fields.reshape(6)).reshape(args.subjects, 3, 2)
            f2 = (g2 + fields.reshape(6)).reshape(args.subjects, 3, 2)
            fp = (gp + fields.reshape(6)).reshape(args.subjects, 3, 2)
            first = conditional_statistic(f1); second = conditional_statistic(f2)
            identity = matrix_identity(first, second)
            if identity['identity_max_z'] > 5:
                raise ArithmeticError('Multivariate identity exceeds five Monte Carlo standard errors')
            inverse2 = np.linalg.pinv(a2, rcond=1e-12)
            bridge = a1 - a1 @ inverse2 @ a1
            _, bridge_values = root_psd(bridge)
            range_defect = float(np.linalg.norm(a1 - a1 @ inverse2 @ a2, 2))
            if range_defect > 1e-10:
                raise ArithmeticError('Gaussian bridge range identity failed')
            innovation1 = 2 * (1 - np.square(posterior_moments(f1, 1)).sum(axis=-1))
            innovation2 = 2 * (1 - np.square(posterior_moments(f2, 1)).sum(axis=-1))
            innovationp = 2 * (1 - np.square(posterior_moments(fp, 1)).sum(axis=-1))
            mean_order, se_order = average_and_se(innovation2 - innovation1)
            mean_tradeoff, se_tradeoff = average_and_se(innovationp - innovation1)
            if not np.all(mean_order + 5 * se_order < 0):
                raise ArithmeticError('Increased loading failed its innovation diagnostic')
            if not (mean_tradeoff[0] - 5 * se_tradeoff[0] > 0 and
                    np.all(mean_tradeoff[1:] + 5 * se_tradeoff[1:] < 0)):
                raise ArithmeticError('Noncomparable allocation did not show the expected tradeoff')
            star_rows = []
            for coordinate in range(3):
                star_rows.append({'coordinate_zero_based': coordinate,
                                  'symmetric_loading': loads[coordinate],
                                  'balanced_loading': balanced[coordinate],
                                  'innovation_symmetric': innovation1[:, coordinate].mean(),
                                  'innovation_increased': innovation2[:, coordinate].mean(),
                                  'increased_minus_symmetric': mean_order[coordinate],
                                  'paired_mcse_increased': se_order[coordinate],
                                  'innovation_balanced': innovationp[:, coordinate].mean(),
                                  'balanced_minus_symmetric': mean_tradeoff[coordinate],
                                  'paired_mcse_balanced': se_tradeoff[coordinate]})
            write_csv(out / 'star.csv', star_rows)
            np.savez_compressed(out / 'star_inputs.npz', original=x, field_symmetric=f1,
                                field_increased=f2, field_balanced=fp, h=fields, J=j,
                                A_symmetric=a1, A_increased=a2, A_balanced=ap)
            np.savez_compressed(out / 'matrix_diagnostics.npz',
                                **{key: value for key, value in identity.items() if isinstance(value, np.ndarray)},
                                bridge=bridge, bridge_eigenvalues=bridge_values,
                                A1_eigenvalues=eigen1, Ap_eigenvalues=eigenp)
        receipt.update(status='complete', rejection=rejection,
                       quadrature_rows=len(rows), star_rows=len(star_rows),
                       univariate_max_abs_z=max(abs(row['difference_z']) for row in rows),
                       matrix_identity_max_z=identity['identity_max_z'],
                       covariance_gain_gram_min_eigenvalue=identity['gram_min_eigenvalue'],
                       bridge_range_defect=range_defect,
                       bridge_min_eigenvalue=float(bridge_values.min()),
                       A1_rank=5, positive_control='balanced centre and leaves move in opposite directions',
                       elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        receipt.update(status='incomplete', error=repr(error), elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        receipt['file_sha256'] = {f.name: sha(f) for f in sorted(out.iterdir())
                                  if f.is_file() and f.name != 'receipt.json'}
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2)); print(out)


if __name__ == '__main__':
    main()
