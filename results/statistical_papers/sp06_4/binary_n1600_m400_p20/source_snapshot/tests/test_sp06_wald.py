import numpy as np
import pytest
from scipy.stats import chi2

from sp06.wald import by_select, calibrated_blocks


def example():
    rng = np.random.default_rng(1060001)
    x = rng.normal(size=(500, 6))
    xt = .2 * x + rng.normal(size=x.shape)
    y = x[:, 0] + .3 * x[:, 1] ** 2 + rng.normal(size=len(x))
    t = np.c_[x[:, 1], x[:, 2] ** 2]
    return x, xt, y, t


def test_pilot_derivative_against_independent_weighted_normal_equations():
    x, xt, y, t = example()
    result = calibrated_blocks(x, xt, y, t, np.eye(2), 400, ridge_penalty=0)
    design = np.c_[np.ones(len(x)), x, xt]
    def weighted_contrast(coordinate, step):
        weights = np.exp(step * t[:, coordinate])
        b = np.linalg.solve(design.T @ (weights[:, None] * design), design.T @ (weights * y))
        return (b[1:7] - b[7:]).reshape(3, 2)
    for coordinate in range(2):
        finite = (weighted_contrast(coordinate, 1e-5) - weighted_contrast(coordinate, -1e-5)) / 2e-5
        np.testing.assert_allclose(result.pilot_derivative[:, :, coordinate], -finite, rtol=2e-7, atol=1e-9)


def test_added_variance_pvalues_and_one_hypothesis_per_block():
    x, xt, y, t = example()
    result = calibrated_blocks(x, xt, y, t, np.array([[2., .4], [.4, 1.]]), 100)
    assert result.pvalues.shape == (3,)
    assert np.all(result.pvalues >= result.naive_pvalues - 1e-14)
    assert set(result.selected) <= set(result.naive_selected)
    for j in range(3):
        statistic = result.contrasts[j] @ np.linalg.solve(result.total_variance[j], result.contrasts[j])
        np.testing.assert_allclose(result.pvalues[j], chi2.sf(statistic, 2), atol=1e-14)
        assert np.linalg.eigvalsh(result.total_variance[j] - result.oracle_variance_component[j]).min() > -1e-14


def test_oracle_variance_limit_and_by_reference():
    x, xt, y, t = example()
    result = calibrated_blocks(x, xt, y, t, np.zeros((2, 2)), 100)
    np.testing.assert_array_equal(result.pvalues, result.naive_pvalues)
    np.testing.assert_array_equal(by_select(np.array([.001, .01, .09, .3]), .1), [0, 1])
    assert len(by_select(np.ones(4), .1)) == 0


def test_errors_keep_invalid_inputs_visible():
    x, xt, y, t = example()
    with pytest.raises(ValueError):
        calibrated_blocks(x, xt, y, t, -np.eye(2), 100)
    with pytest.raises(ValueError):
        calibrated_blocks(x, xt, y, t, np.eye(2), 100, block_width=3)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        calibrated_blocks(x, xt, y, t, np.eye(2), 100)
