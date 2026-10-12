import numpy as np

from sp06.harmonics_hc import block_norm_scores, harmonic_features, hc_pair
from toroidal import knockstat, vec


def data():
    rng = np.random.default_rng(1063901)
    x = rng.uniform(-np.pi, np.pi, (150, 3))
    xt = rng.uniform(-np.pi, np.pi, x.shape)
    y = np.cos(x[:, 0]) + .2 * np.sin(2 * x[:, 1]) + rng.normal(size=len(x))
    return x, xt, y


def test_harmonic_order_matches_reference_and_whole_block_swaps():
    x, xt, y = data()
    np.testing.assert_array_equal(harmonic_features(x, 1), vec(x).reshape(len(x), -1))
    np.testing.assert_allclose(block_norm_scores(harmonic_features(x, 1), harmonic_features(xt, 1), y, 2),
                               knockstat(x, xt, y, penalty=1.), atol=1e-13)
    first, second = harmonic_features(x, 2), harmonic_features(xt, 2)
    expected = block_norm_scores(first, second, y, 4)
    a, b = first.copy(), second.copy()
    a[:, 4:8], b[:, 4:8] = second[:, 4:8], first[:, 4:8]
    swapped = block_norm_scores(a, b, y, 4)
    np.testing.assert_allclose(swapped, expected * np.array([1, -1, 1]), atol=1e-13)


def test_zero_ridge_hc3_against_independent_deleted_observation_fits():
    x, xt, y = data()
    original, witness = harmonic_features(x, 1), harmonic_features(xt, 1)
    zero, corrected, leverage = hc_pair(original, witness, y, np.c_[x[:, 0]], np.zeros((1, 1)), 100, ridge_penalty=0)
    design = np.c_[np.ones(len(x)), original, witness]
    all_coef = np.linalg.lstsq(design, y, rcond=None)[0]
    all_q = (all_coef[1:7] - all_coef[7:]).reshape(3, 2)
    changes = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        deleted = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        changes.append(all_q - (deleted[1:7] - deleted[7:]).reshape(3, 2))
    changes = np.array(changes)
    independent = np.einsum("npi,npj->pij", changes, changes)
    np.testing.assert_allclose(corrected.oracle_variance_component, independent, rtol=1e-11, atol=1e-13)
    assert np.all(corrected.pvalues >= zero.pvalues - 1e-14)
    assert leverage.min() > 0


def test_fixed_ridge_variant_increases_variance_preserves_derivative_and_angle_count():
    x, xt, y = data()
    for h in [1, 2]:
        first, second = harmonic_features(x, h), harmonic_features(xt, h)
        base, corrected, _ = hc_pair(first, second, y, np.c_[np.cos(x[:, 0]), np.sin(x[:, 1])], np.eye(2), 100, 2 * h)
        np.testing.assert_array_equal(base.pilot_derivative, corrected.pilot_derivative)
        assert corrected.pvalues.shape == (3,)
        assert np.all(corrected.pvalues >= base.pvalues - 1e-14)
        for j in range(3):
            assert np.linalg.eigvalsh(corrected.total_variance[j] - base.total_variance[j]).min() > -1e-13
