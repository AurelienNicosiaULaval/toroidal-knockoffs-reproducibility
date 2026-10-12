"""Exact-iid, bounded diagnostics of the SP12 uniform-regression calculation.

Nine datasets are implementation diagnostics, not a Monte Carlo FDR study.
No project generator, population integrator or selector is imported.
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
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq
from scipy.special import i0e, i1e, ive
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/statistical_papers/SP12_PROTOCOLE_FR_20261010.md'
G, KAPPA, RIDGE = .2, .04, 1.


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_vm(radius: np.ndarray) -> np.ndarray:
    return i1e(radius) / i0e(radius)


def fit_g(increment: np.ndarray) -> tuple[float, bool]:
    mean = float(np.cos(increment).mean())
    if mean <= 0:
        return 0., True
    upper = 1.
    while float(mean_vm(np.asarray(upper))) <= mean:
        upper *= 2
    return brentq(lambda x: float(mean_vm(np.asarray(x))) - mean,
                  0., upper, xtol=1e-14), False


def vm_moments(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.linalg.norm(field, axis=1)
    unit = np.divide(field, r[:, None], out=np.zeros_like(field), where=r[:, None] > 0)
    first = mean_vm(r)[:, None] * unit
    second_ratio = ive(2, r) / i0e(r)
    cos_two = unit[:, 0] ** 2 - unit[:, 1] ** 2
    sin_two = 2 * unit[:, 0] * unit[:, 1]
    second = np.empty((len(r), 2, 2))
    second[:, 0, 0] = (1 + second_ratio * cos_two) / 2
    second[:, 1, 1] = (1 - second_ratio * cos_two) / 2
    second[:, 0, 1] = second[:, 1, 0] = second_ratio * sin_two / 2
    return first, second


def population(t: float, angular: int, gaussian: int) -> dict:
    """Periodic integration plus conditional witness moments; no response draws."""
    nodes = np.arange(angular) * (2 * np.pi / angular)
    root, increment = np.meshgrid(nodes, nodes, indexing='ij')
    root, increment = root.ravel(), increment.ravel()
    first = root + increment
    original = np.column_stack([np.cos(first), np.sin(first), np.cos(root), np.sin(root)])
    response = np.cos(2 * first)
    weight = np.exp(KAPPA * np.cos(root) + G * np.cos(increment))
    weight /= angular ** 2 * np.exp(KAPPA + G) * i0e(KAPPA) * i0e(G)
    raw_mass = float(weight.sum())
    weight /= raw_mass
    z, wz = hermgauss(gaussian)
    z, wz = np.sqrt(2) * z, wz / np.sqrt(np.pi)
    mu = np.zeros(8); raw = np.zeros((8, 8)); dz = np.zeros(8)
    natural = t * (original[:, :2] + original[:, 2:4])
    for i, zi in enumerate(z):
        for j, zj in enumerate(z):
            common = natural + np.sqrt(t) * np.array([zi, zj])
            m1, s1 = vm_moments(common)
            shifted = common.copy(); shifted[:, 0] += KAPPA
            m2, s2 = vm_moments(shifted)
            dmean = np.column_stack([original, m1, m2])
            w = weight * wz[i] * wz[j]
            mu += w @ dmean
            dz += (w * response) @ dmean
            raw += (dmean.T * w) @ dmean
            for start, moment, second in [(4, m1, s1), (6, m2, s2)]:
                correction = second - moment[:, :, None] * moment[:, None, :]
                raw[start:start+2, start:start+2] += np.einsum('n,nab->ab', w, correction)
    gamma = raw - np.outer(mu, mu)
    response_mean = float(weight @ response)
    response_variance = float(weight @ (response - response_mean) ** 2)
    covariance = dz - mu * response_mean
    beta = np.linalg.solve(gamma, covariance)
    residual_variance = response_variance - float(covariance @ beta)
    gradient = np.array([0., 0., -1., 0., 0., 0., 1., 0.])
    chi_variance = float(gradient @ np.linalg.solve(gamma, gradient))
    return {'mu': mu, 'gamma': gamma, 'beta': beta,
            'response_mean': response_mean, 'response_variance': response_variance,
            'residual_variance': residual_variance, 'chi_variance': chi_variance,
            'alpha': residual_variance * chi_variance, 'raw_angular_mass': raw_mass,
            'angular': angular, 'gaussian': gaussian}


def scores(beta: np.ndarray) -> np.ndarray:
    b = beta.reshape(-1, 8)
    return np.column_stack([np.linalg.norm(b[:, :2], axis=1) - np.linalg.norm(b[:, 4:6], axis=1),
                            np.linalg.norm(b[:, 2:4], axis=1) - np.linalg.norm(b[:, 6:8], axis=1)]).ravel()


def selection(w: np.ndarray) -> np.ndarray:
    for threshold in np.unique(np.abs(w[w != 0])):
        positives = np.count_nonzero(w >= threshold)
        if (1 + np.count_nonzero(w <= -threshold)) / max(1, positives) <= .1:
            return w >= threshold
    return np.zeros(len(w), dtype=bool)


def main() -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    out = ROOT / 'results/statistical_papers' / ('sp12_growing_' + stamp)
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__), out / 'executed_source.py')
    shutil.copyfile(PROTOCOL, out / 'protocol.md')
    started = time.perf_counter()
    record = {'status': 'running', 'source_sha256': sha(Path(__file__)),
              'protocol_sha256': sha(PROTOCOL), 'new_selection_datasets': 0,
              'selection_subjects': 0, 'pilot_subjects': 0,
              'scope': 'Nine diagnostics, not an FDR calibration or numerical proof of the limit',
              'iid_originals': 'Independent von Mises root and increment, no MCMC'}
    rows, quadratures, errors = [], [], []
    try:
        config = [{'K': k, 'n': n, 'm': 40} for k in [5, 10, 20] for n in [2000, 8000, 32000]]
        (out / 'configuration.json').write_text(json.dumps({'g': G, 'kappa': KAPPA,
            'ridge': RIDGE, 'level': .1, 'seed_root': 2026101201, 'cases': config}, indent=2)+'\n')
        seeds = np.random.SeedSequence(2026101201).spawn(len(config))
        # The two deterministic threshold controls retain every negative count.
        for k in [5, 10, 20, 100]:
            positive = np.column_stack([np.ones(k), np.full(k, .001)]).ravel()
            negative = np.column_stack([np.ones(k), np.full(k, -.001)]).ravel()
            assert selection(positive).all()
            chosen = selection(negative)
            assert not chosen[1::2].any()
            assert chosen[::2].all() == (k >= 10)
        for case, seed in zip(config, seeds):
            k, n, m = case['K'], case['n'], case['m']
            rng = [np.random.default_rng(s) for s in seed.spawn(8)]
            root = rng[0].vonmises(0., KAPPA, (n, k))
            inc = rng[1].vonmises(0., G, (n, k))
            x1 = (root + inc + np.pi) % (2*np.pi) - np.pi
            pilot_root = rng[2].vonmises(0., KAPPA, (m, k))
            pilot_inc = rng[3].vonmises(0., G, (m, k))
            t, clipped = fit_g(pilot_inc)
            latent = rng[4].standard_normal((n, k, 2))
            v1 = np.stack([np.cos(x1), np.sin(x1)], axis=2)
            v2 = np.stack([np.cos(root), np.sin(root)], axis=2)
            field = t*(v1+v2) + np.sqrt(t)*latent
            shifted = field.copy(); shifted[:, :, 0] += KAPPA
            witness1 = rng[5].vonmises(np.arctan2(field[:, :, 1], field[:, :, 0]), np.linalg.norm(field, axis=2))
            witness2 = rng[6].vonmises(np.arctan2(shifted[:, :, 1], shifted[:, :, 0]), np.linalg.norm(shifted, axis=2))
            noise = rng[7].standard_normal(n)
            y = np.cos(2*x1).sum(axis=1) + noise
            angles = np.stack([x1, root, witness1, witness2], axis=2)
            d = np.stack([np.cos(angles), np.sin(angles)], axis=3).reshape(n, 8*k)
            dc, yc = d-d.mean(axis=0), y-y.mean()
            gram = dc.T @ dc/n
            beta = np.linalg.solve(gram+RIDGE/n*np.eye(8*k), dc.T @ yc/n)
            coarse = population(t, 24, 16); fine = population(t, 32, 24)
            difference = max(float(np.max(np.abs(coarse[key]-fine[key]))) for key in ['mu','gamma','beta'])
            if difference > 1e-9:
                refined = population(t, 48, 32)
                difference = max(float(np.max(np.abs(fine[key]-refined[key]))) for key in ['mu','gamma','beta'])
                fine = refined
            if difference > 1e-9:
                raise ArithmeticError('Population quadrature insufficient: '+str(case))
            quadratures.append({**case,'working_g':t,'maximum_difference':difference,
                                'angular':fine['angular'],'gaussian':fine['gaussian'],
                                'raw_mass_error':abs(fine['raw_angular_mass']-1)})
            target = np.tile(fine['beta'], k)
            error = float(np.linalg.norm(beta-target))
            w = scores(beta); wp = scores(target)
            max_error = float(np.max(np.abs(w-wp)))
            if max_error > np.sqrt(2)*error + 1e-12:
                raise AssertionError('Deterministic norm-score bound failed')
            chosen = selection(w)
            total = int(chosen.sum()); false = int(chosen[1::2].sum())
            delta = t-G
            kl = k*((G-t)*float(mean_vm(np.asarray(G)))
                    + np.log(i0e(t))+abs(t)-np.log(i0e(G))-abs(G))
            if kl < -1e-12:
                raise ArithmeticError('Negative analytic KL')
            row = {**case,'working_g':t,'pilot_clipped':clipped,'sqrt_Km_pilot_error':np.sqrt(k*m)*delta,
                   'uniform_conditional_field_error':abs(delta),'matrix_frobenius_error':np.sqrt(2*k)*abs(delta),
                   'per_subject_KL':max(0.,kl),'m_times_KL':m*max(0.,kl),
                   'n_over_K3m':n/(k**3*m),'regression_error':error,
                   'regression_error_over_K_sqrt_n':error*np.sqrt(n)/k,
                   'maximum_score_error':max_error,'null_population_score':float(wp[1]),
                   'minimum_empirical_eigenvalue':float(np.linalg.eigvalsh(gram)[0]),
                   'minimum_population_eigenvalue':float(np.linalg.eigvalsh(fine['gamma'])[0]),
                   'selected':total,'false_selected':false,'FDP':false/max(1,total),
                   'power':float(chosen[::2].mean()),'empty':total==0}
            rows.append(row)
            name=f'K{k}_n{n}_m{m}'
            np.savez_compressed(out/(name+'.npz'), root=root, increment=inc, original_first=x1,
                pilot_root=pilot_root,pilot_increment=pilot_inc,latent=latent,
                witness_first=witness1,witness_second=witness2,noise=noise,y=y,
                beta=beta,working_g=t,mu=fine['mu'],gamma=fine['gamma'],population_beta=fine['beta'],
                scores=w,selected=chosen)
            record['new_selection_datasets'] += 1
            record['selection_subjects'] += n; record['pilot_subjects'] += m
            print(name+' recorded',flush=True)
        oracle_coarse, oracle_fine = population(G,24,16),population(G,32,24)
        odiff=max(float(np.max(np.abs(oracle_coarse[key]-oracle_fine[key]))) for key in ['mu','gamma','beta'])
        if odiff > 1e-9:
            raise ArithmeticError('Oracle quadrature insufficient')
        # A finite difference checks the existing negative derivative without
        # importing the historical population integrator.
        h=1e-4
        slopes=[]
        for step in [h,h/2]:
            plus,minus=population(G+step,32,24),population(G-step,32,24)
            slopes.append(float((scores(plus['beta'])[1]-scores(minus['beta'])[1])/(2*step)))
        if abs(slopes[1]-slopes[0])>1e-8 or slopes[1]>=0:
            raise ArithmeticError('Negative derivative check failed')
        oracle_serial={key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in oracle_fine.items()}
        oracle_serial.update(negative_slope=slopes[1],slope_resolution_difference=abs(slopes[1]-slopes[0]),
                             quadrature_difference=odiff,leading_slope=-KAPPA*G/8)
        (out/'oracle_population.json').write_text(json.dumps(oracle_serial,indent=2)+'\n')
        for name,values in [('diagnostics.csv',rows),('quadratures.csv',quadratures)]:
            with (out/name).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        record.update(status='complete',root_seed=2026101201,root_spawn_keys=[list(s.spawn_key) for s in seeds],
                      maximum_quadrature_difference=max(r['maximum_difference'] for r in quadratures),
                      oracle_alpha=oracle_fine['alpha'],oracle_null_slope=slopes[1],
                      files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
                      threshold_controls_passed=8,fit_errors=errors)
    except BaseException as error:
        record.update(status='incomplete',error=repr(error))
        raise
    finally:
        record['elapsed_seconds']=time.perf_counter()-started
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
