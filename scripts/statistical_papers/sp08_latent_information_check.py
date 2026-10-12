#!/usr/bin/env python3
"""Exact finite-state checks of the half-information redraw bound.

Independent of all project knockoff generators, fitters and selectors.
The negative-correlation control remains a valid knockoff kernel but violates
the stronger redraw bound. No FDR simulation is performed.
"""
from __future__ import annotations

import datetime as dt
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANGLES = np.arange(4) * np.pi / 2
V = np.c_[np.cos(ANGLES), np.sin(ANGLES)]
V[abs(V) < 1e-14] = 0.
Z = np.cos(2 * ANGLES).astype(int)
STATES = np.array(list(itertools.product(range(4), repeat=2)))


def law(theta: float) -> np.ndarray:
    t = V[STATES[:, 0], 0] * Z[STATES[:, 1]]
    weight = np.exp(theta * t)
    return weight / weight.sum()


def kernel(theta: float, gamma: float) -> np.ndarray:
    answer = np.zeros((len(STATES), len(STATES)))
    for i, (x1, x2) in enumerate(STATES):
        z = Z[x2]
        q = np.exp(theta * z * V[:, 0])
        q /= q.sum()
        mean = q @ V[:, 0]
        variance = q @ ((V[:, 0] - mean) ** 2)
        f = (V[:, 0] - mean) / np.sqrt(variance)
        transition = q * (1 - gamma * f[x1] * f)
        if transition.min() < -1e-12:
            raise ArithmeticError('Control kernel outside its nonnegative neighborhood')
        for j, (w1, w2) in enumerate(STATES):
            if Z[w2] == z:
                answer[i, j] = transition[w1] / 2
    np.testing.assert_allclose(answer.sum(axis=1), 1., atol=2e-15)
    return answer


def design() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = np.repeat(STATES, len(STATES), axis=0)
    witness = np.tile(STATES, (len(STATES), 1))
    d = np.c_[V[original[:, 0]], V[original[:, 1]], V[witness[:, 0]], V[witness[:, 1]]]
    y = Z[original[:, 1]].astype(float)
    t = V[original[:, 0], 0] * y
    return d, y, t


def regression(theta: float, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d, y, _ = design()
    weight = (law(0.)[:, None] * kernel(theta, gamma)).ravel()
    dc = d - weight @ d
    yc = y - weight @ y
    covariance = dc.T @ (weight[:, None] * dc)
    beta = np.linalg.solve(covariance, dc.T @ (weight * yc))
    return beta, covariance, weight, dc


def swap_error(theta: float, gamma: float) -> float:
    joint = law(theta)[:, None] * kernel(theta, gamma)
    mapping = {tuple(x): i for i, x in enumerate(STATES)}
    error = 0.
    for subset in itertools.product((False, True), repeat=2):
        for i, x in enumerate(STATES):
            for j, w in enumerate(STATES):
                xx, ww = x.copy(), w.copy()
                for coordinate, swap in enumerate(subset):
                    if swap:
                        xx[coordinate], ww[coordinate] = ww[coordinate], xx[coordinate]
                error = max(error, abs(joint[i, j] - joint[mapping[tuple(xx)], mapping[tuple(ww)]]))
    return error


def check(gamma: float) -> dict:
    beta, covariance, weight, dc = regression(0., gamma)
    d, y, t = design()
    residual = y - weight @ y - dc @ beta
    contrast = np.zeros((2, 8))
    contrast[:, :2] = np.eye(2)
    contrast[:, 4:6] = -np.eye(2)
    influence = (contrast @ np.linalg.solve(covariance, dc.T)).T * residual[:, None]
    omega = influence.T @ (weight[:, None] * influence)
    b = -influence.T @ (weight * (t - weight @ t))
    information = float(weight @ ((t - weight @ t) ** 2))
    step = 1e-5
    derivative = contrast @ (regression(step, gamma)[0] - regression(-step, gamma)[0]) / (2 * step)
    np.testing.assert_allclose(b, [-1., 0.], atol=2e-14)
    np.testing.assert_allclose(derivative, b, atol=3e-10)
    np.testing.assert_allclose(beta, 0., atol=2e-14)
    np.testing.assert_allclose(omega, np.diag([4 / (1 + gamma), 4]), atol=2e-14)
    ratio = float(b @ np.linalg.solve(omega, b) / information)
    np.testing.assert_allclose(ratio, (1 + gamma) / 2, atol=2e-14)
    errors = [swap_error(theta, gamma) for theta in (-.1, 0., .1)]
    assert max(errors) < 2e-15
    if gamma == 0:
        np.testing.assert_allclose(ratio, .5, atol=2e-14)
    else:
        assert ratio > .5 and ratio < 1
    return {'gamma': gamma, 'max_swap_error': max(errors), 'minimum_design_eigenvalue': float(np.linalg.eigvalsh(covariance).min()),
            'information': information, 'null_derivative': b.tolist(), 'derivative_finite_difference_error': float(abs(b - derivative).max()),
            'null_influence_covariance': omega.tolist(), 'relative_information': ratio,
            'half_bound_valid': ratio <= .5 + 2e-14, 'general_bound_valid': ratio <= 1 + 2e-14}


def main() -> None:
    started = time.perf_counter()
    redraw, control = check(0.), check(.25)
    f0 = Fraction(467686735, 5959489536)
    assert f0 + Fraction(1, 48) < Fraction(1, 10)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'results/statistical_papers' / ('sp08_latent_information_' + stamp)
    output.mkdir()
    result = {'status': 'complete', 'redraw': redraw, 'valid_nonredraw_control': control,
              'pilot_ratio_three_bound_exact': str(f0 + Fraction(1, 48)), 'bound_decimal': float(f0 + Fraction(1, 48)),
              'states_per_joint_table': 256, 'new_selection_datasets': 0, 'elapsed_seconds': time.perf_counter() - started,
              'scope': 'Exact finite-angle sums and independent regression finite differences; not continuous VM or FDR simulation',
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (Path(__file__), ROOT / 'docs/statistical_papers/SP08_LATENT_INFORMATION_EN_20261010.md')}}
    (output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(output / 'tables.npz', states=STATES, angles=ANGLES, redraw_joint=law(0.)[:, None]*kernel(0., 0.),
                        control_joint=law(0.)[:, None]*kernel(0., .25), design=design()[0], response=design()[1], sufficient_statistic=design()[2])
    print(json.dumps(result, indent=2))
    print(output)


if __name__ == '__main__':
    main()
