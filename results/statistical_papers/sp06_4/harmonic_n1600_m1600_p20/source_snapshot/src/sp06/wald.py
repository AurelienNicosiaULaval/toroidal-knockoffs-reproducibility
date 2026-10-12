"""Block contrasts with estimated independent-pilot variance and BY selection.

This is an asymptotic testing procedure, not the finite-sample knockoff filter.
One p-value is calculated for each whole angle block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

Array = NDArray[np.float64]


@dataclass(frozen=True)
class WaldResult:
    contrasts: Array
    oracle_variance_component: Array
    pilot_derivative: Array
    total_variance: Array
    pvalues: Array
    naive_pvalues: Array
    selected: NDArray[np.int64]
    naive_selected: NDArray[np.int64]


def by_select(pvalues: Array, level: float) -> NDArray[np.int64]:
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("A nonempty vector of finite p-values is required")
    if np.any((values < 0) | (values > 1)) or not 0 < level < 1:
        raise ValueError("P-values must lie in [0,1] and the level in (0,1)")
    p = len(values)
    ordered = np.sort(values)
    threshold = level * np.arange(1, p + 1) / (p * np.sum(1 / np.arange(1, p + 1)))
    valid = np.flatnonzero(ordered <= threshold)
    return np.flatnonzero(values <= ordered[valid[-1]]) if len(valid) else np.empty(0, dtype=np.int64)


def calibrated_blocks(original: Array, witness: Array, response: Array,
                      sufficient_statistic: Array, pilot_covariance: Array,
                      pilot_size: int, block_width: int = 2, level: float = .1,
                      ridge_penalty: float = 1.) -> WaldResult:
    """V is the covariance of sqrt(m) times the fitted canonical parameter.

    The caller must provide compatible canonical sufficient statistics and V
    from an independent pilot. This function cannot certify that provenance.
    The fixed-dimensional theorem requires a correctly specified regular pilot.
    """
    x, xt, y, t, v = (np.asarray(item, dtype=float) for item in
                      (original, witness, response, sufficient_statistic, pilot_covariance))
    if x.ndim != 2 or xt.shape != x.shape or y.shape != (len(x),):
        raise ValueError("Original and witness matrices must match, with one response per row")
    if block_width < 1 or block_width % 2 or x.shape[1] % block_width:
        raise ValueError("Whole angle blocks require an even width dividing the feature count")
    if t.ndim != 2 or len(t) != len(x) or v.shape != (t.shape[1], t.shape[1]):
        raise ValueError("Canonical statistics and pilot covariance dimensions do not match")
    if not all(np.isfinite(item).all() for item in (x, xt, y, t, v)):
        raise ValueError("All supplied values must be finite; missing subjects are not silently removed")
    if pilot_size < 1 or int(pilot_size) != pilot_size or not np.isfinite(ridge_penalty) or ridge_penalty < 0:
        raise ValueError("A positive integer pilot size and a finite nonnegative fixed penalty are required")
    if not np.allclose(v, v.T, atol=1e-12, rtol=0) or np.linalg.eigvalsh(v).min() < -1e-12:
        raise ValueError("Pilot covariance must be symmetric positive semidefinite")
    n, features = x.shape
    if n <= 2 * features + 1:
        raise ValueError("This full-rank fixed-dimensional implementation requires more rows than design columns")
    d = np.c_[x, xt]
    d -= d.mean(axis=0)
    yc = y - y.mean()
    gram = d.T @ d / n
    if np.linalg.cond(gram) > 1e12:
        raise ArithmeticError("Unresolved design covariance; the theorem requires a nonsingular population design")
    beta = np.linalg.solve(gram + ridge_penalty / n * np.eye(2 * features), d.T @ yc / n)
    residual = yc - d @ beta
    precision_design = cho_solve(cho_factor(gram, lower=True), d.T).T
    p = features // block_width
    contrasts = (beta[:features] - beta[features:]).reshape(p, block_width)
    influences = (precision_design[:, :features] - precision_design[:, features:]) * residual[:, None]
    influences = influences.reshape(n, p, block_width)
    influences -= influences.mean(axis=0)
    tc = t - t.mean(axis=0)
    oracle = np.einsum("npi,npj->pij", influences, influences) / n ** 2
    derivative = -np.einsum("npi,nk->pik", influences, tc) / n
    total = oracle + np.einsum("pik,kl,pjl->pij", derivative, v, derivative) / pilot_size
    pvalues = np.empty(p)
    naive = np.empty(p)
    for j in range(p):
        # Singular contrast variances are errors, not zero p-values or dropped hypotheses.
        adjusted_statistic = contrasts[j] @ cho_solve(cho_factor(total[j], lower=True), contrasts[j])
        naive_statistic = contrasts[j] @ cho_solve(cho_factor(oracle[j], lower=True), contrasts[j])
        pvalues[j] = chi2.sf(adjusted_statistic, block_width)
        naive[j] = chi2.sf(naive_statistic, block_width)
    return WaldResult(contrasts, oracle, derivative, total, pvalues, naive,
                      by_select(pvalues, level), by_select(naive, level))
