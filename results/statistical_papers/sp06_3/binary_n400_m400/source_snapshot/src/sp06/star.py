"""Exact iid general-field star simulation and compatible pilot covariance.

The exactness refers to the rejection identity, with standard floating-point
von Mises sampling. This sampler uses no Markov chain.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import i0e

from toroidal import Model, rot, vec
from v3.fit_verified import fit_verified, hessian
from v3.population_pair import vm_moments


def true_star(config: dict) -> Model:
    p = int(config["p"])
    phi = config["field_phase"]
    fields = np.empty((p, 2))
    fields[0] = config["root_field_norm"] * np.array([np.cos(phi), np.sin(phi)])
    for j in range(1, p):
        phase = phi + 2 * np.pi * j / (p - 1)
        fields[j] = config["leaf_field_norm"] * np.array([np.cos(phase), np.sin(phase)])
    edges = np.c_[np.zeros(p - 1, dtype=int), np.arange(1, p)]
    matrices = np.array([rot(config["left_rotation_step"] * j)
                         @ np.diag(config["singular_values"])
                         @ rot(config["right_rotation_step"] * j).T for j in range(1, p)])
    return Model(fields, edges, matrices)


def log_i0(value: np.ndarray) -> np.ndarray:
    return np.log(i0e(value)) + np.abs(value)


def root_log_weight(model: Model, roots: np.ndarray) -> np.ndarray:
    expected = np.c_[np.zeros(model.p - 1, dtype=int), np.arange(1, model.p)]
    if not np.array_equal(model.edges, expected):
        raise ValueError("This exact sampler requires a star rooted at vertex zero in edge order")
    vectors = vec(np.asarray(roots, float))
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("A finite vector of root angles is required")
    eta = model.h[None, 1:, :] + np.einsum("na,eab->neb", vectors, model.M)
    return log_i0(np.linalg.norm(eta, axis=-1)).sum(axis=-1)


def exact_star(rng: np.random.Generator, model: Model, n: int) -> tuple[np.ndarray, dict]:
    if int(n) != n or n < 1:
        raise ValueError("A positive integer subject count is required")
    n = int(n)
    expected = np.c_[np.zeros(model.p - 1, dtype=int), np.arange(1, model.p)]
    if not np.array_equal(model.edges, expected):
        raise ValueError("Only a full star rooted at zero is supported")
    bound = float(log_i0(np.linalg.norm(model.h[1:], axis=1)
                        + np.linalg.svd(model.M, compute_uv=False)[:, 0]).sum())
    root_direction = np.arctan2(model.h[0, 1], model.h[0, 0])
    root_concentration = np.linalg.norm(model.h[0])
    accepted = []
    kept = proposals = raw_accepts = 0
    while kept < n:
        count = min(8192, max(256, 3 * (n - kept)))
        candidates = rng.vonmises(root_direction, root_concentration, count)
        log_acceptance = root_log_weight(model, candidates) - bound
        if np.max(log_acceptance) > 1e-11:
            raise ArithmeticError("Rejection envelope violated")
        mask = np.log(rng.random(count)) <= log_acceptance
        block = candidates[mask]
        accepted.append(block)
        kept += len(block)
        raw_accepts += len(block)
        proposals += count
        if proposals > max(50000, 500 * n):
            raise RuntimeError("Rejection budget exceeded; no approximate MCMC fallback used")
    roots = np.concatenate(accepted)[:n]
    x = np.empty((n, model.p))
    x[:, 0] = roots
    eta = model.h[None, 1:, :] + np.einsum("na,eab->neb", vec(roots), model.M)
    x[:, 1:] = rng.vonmises(np.arctan2(eta[..., 1], eta[..., 0]), np.linalg.norm(eta, axis=-1))
    return x, {"proposals": proposals, "accepted_including_surplus": raw_accepts,
               "retained_subjects": n, "acceptance": raw_accepts / proposals,
               "log_envelope": bound, "exact_iid_identity": True}


def canonical_statistics(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    v = vec(x)
    pairs = [np.einsum("na,nb->nab", v[:, j], v[:, k]).reshape(len(x), 4) for j, k in edges]
    return np.concatenate([v.reshape(len(x), -1)] + pairs, axis=1)


def pseudolikelihood_scores(model: Model, x: np.ndarray) -> np.ndarray:
    v = vec(x)
    mean, _ = vm_moments(model.fields(x))
    residual = v - mean
    edge_scores = []
    for j, k in model.edges:
        score = np.einsum("na,nb->nab", residual[:, j], v[:, k])
        score += np.einsum("na,nb->nab", v[:, j], residual[:, k])
        edge_scores.append(score.reshape(len(x), 4))
    return np.concatenate([residual.reshape(len(x), -1)] + edge_scores, axis=1)


def pilot_fit(x: np.ndarray, edges: np.ndarray, penalty_numerator: float = 1.) -> tuple[Model, np.ndarray, dict]:
    if not np.isfinite(penalty_numerator) or penalty_numerator <= 0:
        raise ValueError("A fixed positive penalty numerator is required")
    tau = penalty_numerator / len(x)
    fitted, diagnostics = fit_verified(x, edges, penalty=tau)
    covariance_started = time.perf_counter()
    parameters = np.r_[fitted.h.ravel(), fitted.M.ravel()]
    information = hessian(parameters, x, edges, tau)
    scores = pseudolikelihood_scores(fitted, x)
    scores -= scores.mean(axis=0)
    variability = scores.T @ scores / len(x)
    factor = cho_factor(information, lower=True)
    left = cho_solve(factor, variability)
    covariance = cho_solve(factor, left.T).T
    covariance = (covariance + covariance.T) / 2
    minimum = float(np.linalg.eigvalsh(covariance)[0])
    if minimum < -1e-10 or not np.isfinite(covariance).all():
        raise ArithmeticError("Unresolved joint pseudolikelihood sandwich")
    diagnostics.update(parameter_count=len(parameters), information_condition=float(np.linalg.cond(information)),
                       sandwich_minimum_eigenvalue=minimum, covariance_scope="sqrt(m) parameter error, joint subject scores",
                       covariance_seconds=time.perf_counter() - covariance_started)
    return fitted, covariance, diagnostics
