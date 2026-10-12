import numpy as np
from scipy.optimize._numdiff import approx_derivative

from fit_model import loss_grad
from sp06.star import (canonical_statistics, exact_star, log_i0, pilot_fit,
                      pseudolikelihood_scores, root_log_weight)
from toroidal import EdgeSampler, Model, rot, vec
from v3.fit_verified import hessian
from v3.population_pair import vm_moments


def small_star():
    return Model(np.array([[1.1, .8], [.2, -.1], [-.1, .25]]), np.array([[0, 1], [0, 2]]),
                 np.array([rot(.3) @ np.diag([.8, .5]), np.array([[.6, .2], [-.1, .4]])]))


def test_root_rejection_identity_by_independent_two_dimensional_quadrature():
    model = small_star()
    root = np.array([-.7, .2, 1.3])
    grid = np.linspace(-np.pi, np.pi, 8192, endpoint=False)
    for x in root:
        total = 0.
        for leaf, matrix in zip(model.h[1:], model.M):
            # Independently integrate the leaf interaction exp(v(root)^T M v(leaf)).
            energy = vec(grid) @ leaf + vec(x) @ matrix @ vec(grid).T
            total += np.log(np.exp(energy).mean())
        np.testing.assert_allclose(root_log_weight(model, np.array([x]))[0], total, atol=1e-13)
    bound = log_i0(np.linalg.norm(model.h[1:], axis=1) + np.linalg.svd(model.M, compute_uv=False)[:, 0]).sum()
    assert np.max(root_log_weight(model, grid)) <= bound


def test_exact_star_moments_and_conditional_residuals_against_quadrature():
    model = small_star()
    x, info = exact_star(np.random.default_rng(1061901), model, 25000)
    grid = np.linspace(-np.pi, np.pi, 16384, endpoint=False)
    weights = np.exp(vec(grid) @ model.h[0] + root_log_weight(model, grid))
    weights /= weights.sum()
    mean = weights @ vec(grid)
    error = vec(x[:, 0]) - mean
    np.testing.assert_array_less(np.abs(error.mean(axis=0)), 4 * error.std(axis=0, ddof=1) / np.sqrt(len(x)))
    for edge, leaf in enumerate([1, 2]):
        conditional, _ = vm_moments(model.h[leaf] + vec(x[:, 0]) @ model.M[edge])
        residual = vec(x[:, leaf]) - conditional
        np.testing.assert_array_less(np.abs(residual.mean(axis=0)), 4 * residual.std(axis=0, ddof=1) / np.sqrt(len(x)))
    assert info["exact_iid_identity"]
    assert info["retained_subjects"] == len(x)


def test_joint_scores_hessian_and_canonical_order_independently():
    model = small_star()
    x, _ = exact_star(np.random.default_rng(1061902), model, 180)
    par = np.r_[model.h.ravel(), model.M.ravel()]
    scores = pseudolikelihood_scores(model, x)
    np.testing.assert_allclose(-scores.mean(axis=0), loss_grad(par, vec(x), model.edges)[1], atol=1e-14)
    gradient = lambda theta: loss_grad(theta, vec(x), model.edges, .01)[1]
    numerical = approx_derivative(gradient, par, method="3-point")
    np.testing.assert_allclose(hessian(par, x, model.edges, .01), numerical, atol=2e-9)
    t = canonical_statistics(x, model.edges)
    np.testing.assert_allclose(t @ par, model.log_unnormalized(x), atol=1e-14)


def test_pilot_joint_sandwich_against_direct_inverse_and_reference_sampler():
    model = small_star()
    x, _ = exact_star(np.random.default_rng(1061903), model, 250)
    fitted, covariance, diagnostics = pilot_fit(x, model.edges)
    theta = np.r_[fitted.h.ravel(), fitted.M.ravel()]
    h = hessian(theta, x, model.edges, 1 / len(x))
    scores = pseudolikelihood_scores(fitted, x)
    scores -= scores.mean(axis=0)
    inverse = np.linalg.inv(h)
    np.testing.assert_allclose(covariance, inverse @ (scores.T @ scores / len(x)) @ inverse, atol=2e-12)
    assert diagnostics["gradient_l2"] <= diagnostics["tolerance"]
    xt = EdgeSampler(fitted).sample(np.random.default_rng(1061904), x)
    assert xt.shape == x.shape
    assert np.isfinite(xt).all()
