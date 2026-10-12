#!/usr/bin/env python3
"""Recheck retained SP09 pairs and fields without importing their generator."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
from scipy.special import i0e, i1e, ive, roots_hermitenorm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/statistical_papers/sp09_loading_20261010T155304_010893a9'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_mean(field: np.ndarray) -> np.ndarray:
    radius = np.sqrt(np.sum(field * field, axis=-1))
    ratio = i1e(radius) / i0e(radius)
    scale = np.divide(ratio, radius, out=np.zeros_like(radius), where=radius != 0)
    return field * scale[..., None]


def gaussian_quadrature(loading: float, harmonic: int, size: int) -> float:
    points, weights = roots_hermitenorm(size)
    radius = np.sqrt((loading + np.sqrt(loading) * points[:, None]) ** 2
                     + loading * points[None, :] ** 2)
    ratio = ive(harmonic, radius) / i0e(radius)
    return float(weights @ (ratio * ratio) @ weights / (2 * np.pi))


def main() -> None:
    started = time.perf_counter()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    out = ROOT / 'results/statistical_papers' / ('sp09_recheck_' + stamp)
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out / 'executed_source.py')
    record = {'status': 'running', 'source_sha256': sha(Path(__file__)),
              'new_selection_datasets': 0, 'new_covariate_samples': 0,
              'generator_imported': False, 'data_receipt_sha256': sha(DATA / 'receipt.json')}
    try:
        frozen = json.loads((DATA / 'receipt.json').read_text())
        for name, expected in frozen['file_sha256'].items():
            if sha(DATA / name) != expected:
                raise AssertionError('Frozen input differs: ' + name)
        with (DATA / 'univariate.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        checks = []
        for index, loading in enumerate([.125, .5, 1., 4., 16.]):
            data = np.load(DATA / f'univariate_{index}.npz')
            for harmonic in [1, 2, 3]:
                row = next(x for x in rows if float(x['loading']) == loading
                           and int(x['harmonic']) == harmonic)
                values = np.cos(harmonic * (data['original'] - data['witness']))
                mean = values.mean(); se = values.std(ddof=1) / np.sqrt(len(values))
                np.testing.assert_allclose([mean, se], [float(row['mc_correlation']), float(row['mcse'])],
                                           atol=1e-14, rtol=1e-13)
                low = gaussian_quadrature(loading, harmonic, 160)
                high = gaussian_quadrature(loading, harmonic, 320)
                rice = float(row['quadrature_correlation'])
                if max(abs(high - rice), abs(high - low)) > 1e-8:
                    raise ArithmeticError('Two-dimensional Gaussian quadrature did not agree')
                checks.append({'loading': loading, 'harmonic': harmonic,
                               'gaussian_160': low, 'gaussian_320': high,
                               'rice': rice, 'gaussian_minus_rice': high-rice})
        data = np.load(DATA / 'star_inputs.npz')
        moments = [first_mean(data[k]) for k in ['field_symmetric','field_increased','field_balanced']]
        innovations = [2 * (1 - np.sum(m*m, axis=-1)) for m in moments]
        with (DATA / 'star.csv').open() as handle:
            star = list(csv.DictReader(handle))
        maximum = 0.0
        for j, row in enumerate(star):
            for k, key, difference, error in [
                    (1,'innovation_increased','increased_minus_symmetric','paired_mcse_increased'),
                    (2,'innovation_balanced','balanced_minus_symmetric','paired_mcse_balanced')]:
                delta = innovations[k][:,j]-innovations[0][:,j]
                actual = [innovations[k][:,j].mean(),delta.mean(),delta.std(ddof=1)/np.sqrt(len(delta))]
                expected = [float(row[key]),float(row[difference]),float(row[error])]
                np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-12)
                maximum=max(maximum,float(np.max(np.abs(np.array(actual)-expected))))
        a1=data['A_symmetric']; a2=data['A_increased']
        bridge=a1-a1@np.linalg.pinv(a2)@a1
        if np.linalg.eigvalsh(bridge).min() < -1e-12:
            raise AssertionError('Bridge covariance is not positive semidefinite')
        with (out/'quadrature_crosscheck.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
        record.update(status='complete',raw_univariate_pairs_rechecked=500000,
                      retained_star_subjects_rechecked=100000,
                      independent_quadrature_rows=len(checks),
                      quadrature_max_abs_difference=max(abs(x['gaussian_minus_rice']) for x in checks),
                      star_max_abs_difference=maximum,
                      bridge_min_eigenvalue=float(np.linalg.eigvalsh(bridge).min()))
    except BaseException as error:
        record.update(status='incomplete',error=repr(error))
        raise
    finally:
        record['elapsed_seconds']=time.perf_counter()-started
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2));print(out)


if __name__=='__main__':
    main()
