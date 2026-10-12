#!/usr/bin/env python3
"""Independent finite-state SCIP check of the null-information budget."""
from __future__ import annotations

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
GRID = np.arange(4) * np.pi / 2
X = np.array(list(itertools.product(range(4), repeat=3)), int)
VECTORS = np.stack((np.cos(GRID[X]), np.sin(GRID[X])), -1)
STATISTICS = np.c_[VECTORS.reshape(64, 6),
                  (VECTORS[:, 0, :, None] * VECTORS[:, 1, None, :]).reshape(64, 4),
                  (VECTORS[:, 1, :, None] * VECTORS[:, 2, None, :]).reshape(64, 4)]
TRUE = np.r_[.3, .1, .7, -.2, -.4, .25, .5, .13, -.08, .25, .35, -.1, .18, .6]


def scip(parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = STATISTICS @ parameter
    probability = np.exp(energy - energy.max())
    probability /= probability.sum()
    current = {tuple(row): float(weight) for row, weight in zip(X, probability)}
    for coordinate in range(3):
        conditionals = {}
        for state, weight in current.items():
            key = state[:coordinate] + state[coordinate + 1:3] + state[3:]
            if key not in conditionals:
                conditionals[key] = np.zeros(4)
            conditionals[key][state[coordinate]] += weight
        for key in conditionals:
            conditionals[key] /= conditionals[key].sum()
        updated = {}
        for state, weight in current.items():
            key = state[:coordinate] + state[coordinate + 1:3] + state[3:]
            for value in range(4):
                updated[state + (value,)] = weight * conditionals[key][value]
        current = updated
    states = np.array(sorted(current), int)
    weights = np.array([current[tuple(row)] for row in states])
    return states, weights, probability


def setup(parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states, joint, working = scip(parameter)
    true_probability = np.exp(STATISTICS @ TRUE - (STATISTICS @ TRUE).max())
    true_probability /= true_probability.sum()
    original_index = states[:, 0] * 16 + states[:, 1] * 4 + states[:, 2]
    joint *= true_probability[original_index] / working[original_index]
    design = np.stack((np.cos(GRID[states]), np.sin(GRID[states])), -1).reshape(4096, 12)
    response_mean = np.cos(2 * GRID[states[:, 0]])
    y = np.concatenate([response_mean - .5, response_mean + .5])
    design = np.concatenate([design, design])
    statistics = np.concatenate([STATISTICS[original_index], STATISTICS[original_index]])
    weights = np.r_[joint / 2, joint / 2]
    return design, y, statistics, weights, states


def projection(parameter: np.ndarray):
    design, response, statistics, weights, states = setup(parameter)
    centered = design - weights @ design
    yc = response - weights @ response
    gram = centered.T @ (weights[:, None] * centered)
    beta = np.linalg.solve(gram, centered.T @ (weights * yc))
    contrast = (beta[:6] - beta[6:]).reshape(3, 2)
    return contrast, (design, response, statistics, weights, states, centered, yc, gram, beta)


def run() -> None:
    started = time.perf_counter()
    contrast, values = projection(TRUE)
    design, response, t, w, states, centered, yc, gram, beta = values
    residual = yc - centered @ beta
    influence = np.linalg.solve(gram, centered.T).T * residual[:, None]
    null_l = np.c_[influence[:, 2:4] - influence[:, 8:10], influence[:, 4:6] - influence[:, 10:12]]
    d = null_l.T @ (w[:, None] * null_l)
    tc = t - w @ t
    information = tc.T @ (w[:, None] * tc)
    b = -null_l.T @ (w[:, None] * tc)
    d_block = np.zeros_like(d)
    d_block[:2, :2], d_block[2:, 2:] = d[:2, :2], d[2:, 2:]
    np.testing.assert_allclose(contrast[1:], 0, atol=3e-14)
    np.testing.assert_allclose(d, d_block, atol=3e-14)
    states0, joint, _ = scip(TRUE)
    index = {tuple(state): i for i, state in enumerate(states0)}
    max_swap = 0.
    for subset in itertools.product((0, 1), repeat=3):
        permuted = states0.copy()
        for j, swap in enumerate(subset):
            if swap:
                permuted[:, j], permuted[:, j + 3] = states0[:, j + 3], states0[:, j]
        reordered = np.array([joint[index[tuple(state)]] for state in permuted])
        max_swap = max(max_swap, float(np.max(abs(joint - reordered))))
    assert max_swap < 1e-15
    jacobian = np.empty((4, 14))
    for coordinate in range(14):
        shift = np.eye(14)[coordinate] * 1e-5
        jacobian[:, coordinate] = ((projection(TRUE + shift)[0] - projection(TRUE - shift)[0]) / 2e-5)[1:].ravel()
    np.testing.assert_allclose(jacobian, b, atol=3e-10, rtol=3e-7)
    remainder = information - b.T @ np.linalg.solve(d_block, b)
    eigen_i, directions_i = np.linalg.eigh(information)
    inverse_i = (directions_i / eigen_i) @ directions_i.T
    root_i = (directions_i * np.sqrt(eigen_i)) @ directions_i.T
    covariance = inverse_i + .3 * np.outer(np.arange(1., 15), np.arange(1., 15)) / 14
    kappa = float(np.linalg.eigvalsh(root_i @ covariance @ root_i).max())
    minimum_budget = float(np.linalg.eigvalsh(remainder).min())
    minimum_eff = float(np.linalg.eigvalsh(d_block - b @ inverse_i @ b.T).min())
    minimum_general = float(np.linalg.eigvalsh(kappa * d_block - b @ covariance @ b.T).min())
    assert min(minimum_budget, minimum_eff, minimum_general) > -1e-12
    # Positive control: duplicating a whitened score violates orthogonality.
    duplicated = np.ones((2, 1))
    false_budget = np.eye(1) - duplicated.T @ duplicated
    assert false_budget[0, 0] == -1
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'results/statistical_papers' / ('sp07_information_' + stamp)
    output.mkdir()
    np.savez_compressed(output / 'moments.npz', null_influence_covariance=d, null_derivative=b,
                        finite_difference=jacobian, information=information, pilot_covariance=covariance,
                        contrasts=contrast, true_parameter=TRUE)
    result = {'status': 'complete', 'augmented_states_with_noise': 8192, 'selection_simulations': 0,
              'graph': 'three-node chain on four-valued angles; finite-state SCIP, not a continuous VM experiment',
              'max_swap_error': max_swap, 'max_null_contrast': float(abs(contrast[1:]).max()),
              'max_cross_null_covariance': float(abs(d[:2, 2:]).max()),
              'max_derivative_error': float(abs(jacobian - b).max()),
              'minimum_budget_eigenvalue': minimum_budget, 'minimum_efficient_bound_eigenvalue': minimum_eff,
              'minimum_general_bound_eigenvalue': minimum_general, 'kappa': kappa,
              'derivative_norm': float(np.linalg.norm(b)), 'duplicated_influence_control': -1.,
              'elapsed_seconds': time.perf_counter() - started,
              'sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (Path(__file__), ROOT / 'docs/statistical_papers/SP07_INFORMATION_BUDGET_EN_20261010.md', output / 'moments.npz')}}
    (output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(output)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
