"""Prespecified harmonics and a leverage correction for sensitivity analysis.

HC3 at zero ridge follows the classical OLS formula. With positive fixed ridge
we retain the original unpenalized influence precision and ridge leverages;
this is an explicitly labelled HC3-style variant, asymptotically equivalent.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

from sp06.wald import WaldResult, by_select, calibrated_blocks


def harmonic_features(angles: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(angles, float)
    if x.ndim != 2 or not np.isfinite(x).all() or int(order) != order or order < 1:
        raise ValueError("A finite angle matrix and positive integer harmonic order are required")
    frequency = np.arange(1, int(order) + 1)
    values = x[:, :, None] * frequency
    features = np.stack((np.cos(values), np.sin(values)), axis=-1)
    return features.reshape(len(x), -1)


def block_norm_scores(original: np.ndarray, witness: np.ndarray, y: np.ndarray,
                      block_width: int, penalty: float = 1.) -> np.ndarray:
    n, features = original.shape
    p = features // block_width
    design = np.c_[original, witness]
    design -= design.mean(axis=0)
    coefficient = np.linalg.solve(design.T @ design + penalty * np.eye(2 * features),
                                  design.T @ (y - y.mean()))
    return np.linalg.norm(coefficient[:features].reshape(p, block_width), axis=1) - np.linalg.norm(
        coefficient[features:].reshape(p, block_width), axis=1)


def hc_pair(original: np.ndarray, witness: np.ndarray, response: np.ndarray,
            sufficient: np.ndarray, pilot_covariance: np.ndarray, pilot_size: int,
            block_width: int = 2, level: float = .1, ridge_penalty: float = 1.) -> tuple[WaldResult, WaldResult, np.ndarray]:
    base = calibrated_blocks(original, witness, response, sufficient, pilot_covariance, pilot_size,
                             block_width, level, ridge_penalty)
    n, features = original.shape
    p = features // block_width
    design = np.c_[original, witness]
    design -= design.mean(axis=0)
    gram = design.T @ design / n
    penalized = gram + ridge_penalty / n * np.eye(2 * features)
    coef = cho_solve(cho_factor(penalized, lower=True), design.T @ (response - response.mean()) / n)
    residual = response - response.mean() - design @ coef
    ridge_precision_design = cho_solve(cho_factor(penalized, lower=True), design.T).T
    leverage = 1 / n + (design * ridge_precision_design).sum(axis=1) / n
    if not np.isfinite(leverage).all() or leverage.min() < -1e-10 or leverage.max() >= 1 - 1e-10:
        raise ArithmeticError("Leverage correction not numerically resolved")
    precision = cho_solve(cho_factor(gram, lower=True), design.T).T
    raw = (precision[:, :features] - precision[:, features:]) * (residual / (1 - leverage))[:, None]
    raw = raw.reshape(n, p, block_width)
    corrected_component = np.einsum("npi,npj->pij", raw, raw) / n ** 2
    total = base.total_variance + corrected_component - base.oracle_variance_component
    pvalue = np.empty(p)
    naive = np.empty(p)
    for j in range(p):
        adjusted = base.contrasts[j] @ cho_solve(cho_factor(total[j], lower=True), base.contrasts[j])
        unadjusted = base.contrasts[j] @ cho_solve(cho_factor(corrected_component[j], lower=True), base.contrasts[j])
        pvalue[j] = chi2.sf(adjusted, block_width)
        naive[j] = chi2.sf(unadjusted, block_width)
    corrected = WaldResult(base.contrasts, corrected_component, base.pilot_derivative, total,
                           pvalue, naive, by_select(pvalue, level), by_select(naive, level))
    return base, corrected, leverage
