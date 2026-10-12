#!/usr/bin/env python3
"""Independent Gaussian-limit checks under the frozen SP11 protocol.

No toroidal generator, model estimator or project selector is imported.
Rows are reused across configurations for paired differences.
"""
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
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammainc, gammaln

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/statistical_papers/SP11_PROTOCOLE_FR_20261010.md'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gaussian_tv(rank: int, inflation: float) -> float:
    if rank < 0 or inflation < 0 or not np.isfinite(inflation):
        raise ValueError('Rank and finite inflation must be nonnegative')
    if rank == 0 or inflation == 0:
        return 0.0
    radius = rank * (1 + inflation) * np.log1p(inflation) / inflation
    return float(gammainc(rank / 2, radius / 2) -
                 gammainc(rank / 2, radius / (2 * (1 + inflation))))


def radial_tv(rank: int, inflation: float) -> float:
    if inflation == 0:
        return 0.0
    radius = np.sqrt(rank * (1 + inflation) * np.log1p(inflation) / inflation)
    log_constant = (1 - rank / 2) * np.log(2.) - gammaln(rank / 2)

    def integrand(x: float) -> float:
        base = np.exp(log_constant + (rank - 1) * np.log(x) - x*x/2) if x else (np.exp(log_constant) if rank == 1 else 0.)
        inflated = base * np.exp(-rank/2*np.log1p(inflation) + x*x/2*inflation/(1+inflation))
        return base - inflated

    value, error = quad(integrand, 0., radius, epsabs=2e-12, epsrel=2e-12, limit=200)
    if error > 1e-10:
        raise ArithmeticError('Radial integral exceeds fixed error tolerance')
    return float(value)


def threshold_batch(nulls: np.ndarray, signals: int, level: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= level < 1 or signals < 0:
        raise ValueError('Invalid level or signal count')
    order = np.argsort(-np.abs(nulls), axis=1)
    positives = np.take_along_axis(nulls, order, axis=1) > 0
    positive_count = np.cumsum(positives, axis=1)
    negative_count = np.cumsum(~positives, axis=1)
    passing = (1 + negative_count) <= level * np.maximum(signals + positive_count, 1)
    indices = np.arange(1, nulls.shape[1] + 1)
    last = np.max(np.where(passing, indices, 0), axis=1)
    false = np.where(last > 0, positive_count[np.arange(len(nulls)), np.maximum(last-1, 0)], 0)
    fdp = false / np.maximum(signals + false, 1)
    active_only = signals > 0 and 1/signals <= level
    selected = (last > 0) | active_only
    return fdp, selected


def threshold_scalar(nulls: np.ndarray, signals: int, level: float) -> tuple[float, bool]:
    # Enumerate ascending finite candidate thresholds; separated alternatives
    # enter every finite threshold in this limiting experiment.
    for t in sorted(set(abs(float(w)) for w in nulls if w != 0)):
        negative = sum(w <= -t for w in nulls)
        positive = signals + sum(w >= t for w in nulls)
        if (1 + negative) <= level * max(positive, 1):
            false = positive - signals
            return false/max(positive, 1), True
    return 0., signals > 0 and 1/signals <= level


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    start = time.perf_counter()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    out = ROOT / 'results/statistical_papers' / ('sp11_gaussian_risk_' + stamp)
    out.mkdir()
    shutil.copyfile(__file__, out/'executed_source.py')
    shutil.copyfile(PROTOCOL, out/'protocol.md')
    record = {'status': 'running', 'protocol_sha256': sha(PROTOCOL),
              'executed_source_sha256': sha(Path(__file__)),
              'scope': 'Synthetic Gaussian limiting scores; not toroidal data or finite-sample fitted-law guarantees',
              'new_toroidal_selection_datasets': 0, 'new_toroidal_covariate_samples': 0}
    try:
        quadratures = []
        for rank in [1, 2, 3, 5, 10, 20, 40]:
            for inflation in [0., .05, .1, .5, 2., 5.]:
                exact = gaussian_tv(rank, inflation)
                independent = radial_tv(rank, inflation)
                if abs(exact-independent) > 1e-10:
                    raise ArithmeticError('Independent Gaussian TV calculations disagree')
                quadratures.append({'rank': rank, 'inflation': inflation, 'cdf_tv': exact,
                                    'radial_tv': independent, 'difference': exact-independent})
        write_csv(out/'quadratures.csv', quadratures)
        floors = []
        for rank in [1, 3, 10, 20]:
            critical = brentq(lambda t: gaussian_tv(rank, t)-(.1-1/20), 1e-9, 10., xtol=1e-13)
            floors.append({'p': 20, 'target_q': .1, 'rank': rank, 'pilot_factor': 1.,
                           'critical_inflation': critical, 'critical_m_over_n': 1/(2*critical),
                           'interpretation': 'Below this m/n bound, the dimension-based input is <1/p'})
        write_csv(out/'discovery_floor.csv', floors)

        subject_count = 200000
        streams = np.random.SeedSequence(2026101101).spawn(2)
        original_noise = np.random.default_rng(streams[0]).normal(size=(subject_count, 10))
        pilot_noise = np.random.default_rng(streams[1]).normal(size=(subject_count, 3))
        angle = 2*np.pi*np.arange(10)/10
        common = np.ones((10, 1))/np.sqrt(10)
        opposing = np.r_[np.ones(5),-np.ones(5)][:,None]/np.sqrt(10)
        diffuse = np.c_[common[:,0], np.cos(angle)/np.sqrt(5), np.sin(angle)/np.sqrt(5)]
        designs = {'common_positive': common, 'opposing': opposing, 'rank_three': diffuse}
        summaries, paired, outputs, transforms = [], [], {}, []
        for name, basis in designs.items():
            rank = basis.shape[1]
            np.testing.assert_allclose(basis.T@basis, np.eye(rank), atol=1e-14)
            for inflation in [.1, .5, 2.]:
                matrix = inflation*basis@basis.T
                if np.linalg.eigvalsh(matrix).max() > inflation+1e-12:
                    raise ArithmeticError('Covariance exceeds the claimed information budget')
                e = np.linspace(inflation/rank, inflation, rank)
                scale = np.diag(np.sqrt(e/inflation))
                noise = np.diag(1-e/inflation)
                np.testing.assert_allclose(scale@scale.T+noise, np.eye(rank), atol=1e-14)
                np.testing.assert_allclose((1+inflation)*scale@scale.T+noise, np.diag(1+e), atol=1e-14)
                transforms.append({'configuration': name, 'inflation': inflation, 'rank': rank})
                nulls = original_noise + np.sqrt(inflation)*pilot_noise[:,:rank]@basis.T
                delta = gaussian_tv(rank, inflation)
                nominal, calibrated = .1, max(0., .1-delta)
                by_method = {}
                for method, level in [('unadjusted', nominal), ('budget_level', calibrated)]:
                    fdp, selected = threshold_batch(nulls, 10, level)
                    for i in range(1000):
                        scalar_fdp, scalar_selected = threshold_scalar(nulls[i], 10, level)
                        if scalar_fdp != fdp[i] or scalar_selected != selected[i]:
                            raise ArithmeticError('Batch and scalar thresholds disagree')
                    key = name + '_' + str(inflation) + '_' + method
                    outputs[key+'_fdp'] = fdp
                    outputs[key+'_selected'] = selected
                    by_method[method] = (fdp, selected)
                    # The additional N/p factor is a known-support benchmark
                    # in this limiting diagnostic, not part of the chosen input.
                    upper = min(1., level + .5*delta)
                    summaries.append({'configuration': name, 'inflation': inflation, 'rank': rank,
                                      'method': method, 'input_q': level, 'replications': subject_count,
                                      'gaussian_tv_envelope': delta, 'fdr_upper_bound': upper,
                                      'fdr': float(fdp.mean()), 'fdr_mcse': float(fdp.std(ddof=1)/np.sqrt(subject_count)),
                                      'power': float(selected.mean()), 'power_mcse': float(selected.std(ddof=1)/np.sqrt(subject_count)),
                                      'empty': float((~selected).mean()), 'below_discovery_floor': level < 1/20})
                for metric, index in [('fdp', 0), ('power', 1)]:
                    difference = by_method['budget_level'][index].astype(float)-by_method['unadjusted'][index].astype(float)
                    paired.append({'configuration': name, 'inflation': inflation, 'metric': metric,
                                   'budget_minus_unadjusted': float(difference.mean()),
                                   'mcse': float(difference.std(ddof=1)/np.sqrt(subject_count))})
        # A deliberately excessive matrix must fail the budget check.
        wrong = 1.01*.5*common@common.T
        if not np.linalg.eigvalsh(wrong).max() > .5+1e-12:
            raise ArithmeticError('Outside-budget positive control was not detected')
        write_csv(out/'summary.csv', summaries)
        write_csv(out/'paired.csv', paired)
        np.savez_compressed(out/'raw_noise.npz', original_noise=original_noise, pilot_noise=pilot_noise,
                            common_basis=common, opposing_basis=opposing, rank_three_basis=diffuse)
        np.savez_compressed(out/'individual_results.npz', **outputs)
        record.update(status='complete', independent_gaussian_rows=subject_count,
                      matched_covariance_configurations=9, methods_per_configuration=2,
                      gaussian_threshold_applications=subject_count*9*2,
                      scalar_threshold_crosschecks=1000*9*2,
                      quadrature_rows=len(quadratures),
                      quadrature_max_difference=max(abs(r['difference']) for r in quadratures),
                      outside_budget_control_detected=True, covariance_channels_checked=len(transforms))
        record['file_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    except BaseException as error:
        record.update(status='incomplete', error=repr(error))
        raise
    finally:
        record['elapsed_seconds']=time.perf_counter()-start
        (out/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
        print(out)


if __name__=='__main__':
    main()
