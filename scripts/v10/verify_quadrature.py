"""Independent population integration in root/increment coordinates.

Uses conditional circular moments and common Gaussian noise, not the retained
Cartesian-angle quadrature implementation. Grid comparisons are diagnostics,
not certified integration bounds or proofs of asymptotic remainders.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import ive
from numpy.polynomial.hermite import hermgauss

ROOT = Path(__file__).resolve().parents[2]


def population(g: float, t: float, kappa: float, angular: int, gaussian: int):
    if min(g, t) < 0 or min(angular, gaussian) < 2:
        raise ValueError('Nonnegative interactions and positive quadrature orders required')
    angle = np.arange(angular) * 2 * np.pi / angular
    root, increment = np.meshgrid(angle, angle, indexing='ij')
    root, increment = root.ravel(), increment.ravel()
    probability = np.exp(kappa * np.cos(root) + g * np.cos(increment))
    probability /= probability.sum()
    x1, x2 = root + increment, root
    original = np.c_[np.cos(x1), np.cos(x2)]
    response = np.cos(2 * x1)
    w = np.c_[np.cos(x1) + np.cos(x2), np.sin(x1) + np.sin(x2)]
    nodes, weights = hermgauss(gaussian)
    zx, zy = np.meshgrid(nodes, nodes, indexing='ij')
    z = np.c_[zx.ravel(), zy.ravel()] * np.sqrt(2)
    normal_weight = np.outer(weights, weights).ravel() / np.pi
    mean, cross, second = np.zeros(4), np.zeros(4), np.zeros((4, 4))
    for start in range(0, len(root), 64):
        base = t * w[start:start+64, None, :] + np.sqrt(t) * z[None, :, :]
        field = np.stack([base, base + [kappa, 0]], axis=2)
        radius = np.linalg.norm(field, axis=3)
        direction_x = np.divide(field[..., 0], radius, out=np.zeros_like(radius), where=radius != 0)
        direction_y = np.divide(field[..., 1], radius, out=np.zeros_like(radius), where=radius != 0)
        circular_mean = ive(1, radius) / ive(0, radius) * direction_x
        cosine_second = (1 + ive(2, radius) / ive(0, radius)
                         * (direction_x**2 - direction_y**2)) / 2
        witness_mean = np.einsum('a,naj->nj', normal_weight, circular_mean)
        pp = probability[start:start+64]
        oo = original[start:start+64]
        yy = response[start:start+64]
        design_mean = np.c_[oo, witness_mean]
        mean += pp @ design_mean
        cross += (pp * yy) @ design_mean
        second[:2, :2] += (pp[:, None] * oo).T @ oo
        ow = (pp[:, None] * oo).T @ witness_mean
        second[:2, 2:] += ow
        second[2:, :2] += ow.T
        # Both witnesses use the same Gaussian field, independently conditional on it.
        second[2:, 2:] += np.einsum('n,a,naj,nak->jk', pp, normal_weight, circular_mean, circular_mean)
        conditional_variance = cosine_second - circular_mean**2
        second[2:, 2:] += np.diag(np.einsum('n,a,naj->j', pp, normal_weight, conditional_variance))
    covariance = second - np.outer(mean, mean)
    coefficient = np.linalg.solve(covariance, cross - mean * (probability @ response))
    score = np.abs(coefficient[:2]) - np.abs(coefficient[2:])
    return coefficient, score, covariance


def verify():
    start = time.perf_counter()
    records = []
    from v5.population import population as retained
    for g in [.2, .1, .05, .025]:
        kappa = g**2
        delta = min(1e-4, g / 100)
        b, score, covariance = population(g, g, kappa, 32, 16)
        b2, score2, _ = population(g, g, kappa, 48, 24)
        slope = (population(g, g + delta, kappa, 48, 24)[1][1]
                 - population(g, g - delta, kappa, 48, 24)[1][1]) / (2 * delta)
        reference = retained(g, g, kappa, 48, 24)
        expected = np.array([g/4, -g**2/16, 0, -g**2/16]) * kappa
        assert b2[1] < 0 and b2[3] < 0 and score2[0] > 0 and slope < 0
        np.testing.assert_allclose(b2, np.asarray(reference['coefficient'])[[0, 2, 4, 6]], atol=2e-13)
        assert abs(score2[1]) < 2e-13
        records.append(dict(g=g, kappa=kappa, coefficient=b2.tolist(),
                            leading_coefficient=expected.tolist(),
                            scaled_remainder_max=float(np.max(np.abs(b2-expected))/(kappa*(2*g)**3+kappa**3)),
                            null_slope=slope, slope_over_leading=slope/(-kappa*g/8),
                            grid_coefficient_difference=float(np.max(np.abs(b-b2))),
                            coordinate_check_difference=float(np.max(np.abs(b2-np.asarray(reference['coefficient'])[[0,2,4,6]]))),
                            oracle_null_score=float(score2[1]), minimum_eigenvalue=float(np.linalg.eigvalsh(covariance)[0])))
    return dict(records=records, seconds=time.perf_counter()-start,
                scope='four deterministic quadrature checks, not iid-data simulations or certified bounds')


if __name__ == '__main__':
    result=verify()
    (ROOT/'results/v10/quadrature.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
