import numpy as np
from toroidal import Model,select,knockstat
from loading import AllocatedSampler
from v4.core import batch_metrics,ridge_scores,sample_pairs,pair_knockoffs,fit_common_gamma


def test_batch_threshold_matches_reference_with_ties_and_zero():
    rng=np.random.default_rng(109);w=rng.integers(-6,7,(300,20)).astype(float);w[0]=0
    active=rng.uniform(size=w.shape)<.4;fdp,power,empty=batch_metrics(w,active,.2)
    for i in range(len(w)):
        selected=select(w[i],.2);true=active[i,selected].sum()
        assert fdp[i]==(len(selected)-true)/max(1,len(selected))
        assert power[i]==true/max(1,active[i].sum())
        assert empty[i]==(len(selected)==0)


def test_v4_score_matches_retained_reference():
    rng=np.random.default_rng(128);x=sample_pairs(rng,160,5,1.,.7)
    xt=pair_knockoffs(rng,x,.8,.7);y=rng.normal(size=len(x))
    np.testing.assert_allclose(ridge_scores(x,xt,y),knockstat(x,xt,y),atol=1e-13,rtol=1e-13)


def test_common_parameter_estimator_uses_exact_increment_moments():
    x=sample_pairs(np.random.default_rng(876),40000,2,.8,1.)
    assert abs(fit_common_gamma(x)-.8)<.02


def test_shared_normal_run_matches_actual_filter():
    """The p=20 limiting run argument uses 2/19 > q, including q*p=2."""
    rng = np.random.default_rng(932)
    null = rng.normal(size=(3000, 10)) + rng.normal(size=(3000, 1))
    order = np.argsort(-np.abs(null), axis=1)
    positive = np.take_along_axis(null, order, axis=1) > 0
    run = np.cumprod(positive, axis=1).sum(axis=1)
    score = np.c_[np.full((len(null), 10), 100.), null]
    truth = np.r_[np.ones(10, bool), np.zeros(10, bool)]
    fdp, power, empty = batch_metrics(score, truth)
    np.testing.assert_array_equal(fdp, run / (10 + run))
    assert np.all(power == 1) and not np.any(empty)


def test_pair_generator_against_independent_gaussian_loading():
    """Compare empirical moments of the specialized and general samplers."""
    from toroidal import vec
    p = Model(np.array([[0., 0.], [.7, 0.]]),
              np.array([[0, 1]]), np.array([.8 * np.eye(2)]))
    x = sample_pairs(np.random.default_rng(341), 120000, 1, .8, .7)
    a = pair_knockoffs(np.random.default_rng(342), x, .8, .7)
    b = AllocatedSampler(p).sample(np.random.default_rng(343), x)
    moments_a = np.c_[vec(a).reshape(len(x), 4),
                      np.cos(a[:, 0] - a[:, 1]), np.cos(x - a)]
    moments_b = np.c_[vec(b).reshape(len(x), 4),
                      np.cos(b[:, 0] - b[:, 1]), np.cos(x - b)]
    difference = moments_a - moments_b
    se = difference.std(axis=0, ddof=1) / np.sqrt(len(x))
    assert np.all(np.abs(difference.mean(axis=0)) < 6 * se)


def test_local_covariance_formula_against_direct_sampling():
    """Independent empirical check of the matrix used for power calibration."""
    from toroidal import make_tree, sample_tree, vec
    from loading import endpoint_allocations
    from v4.calibrate_coupling import exact_sigma, weights
    from scipy.special import i0e, i1e
    p = make_tree('star', 5, .8)
    sampler = AllocatedSampler(p, endpoint_allocations(p, weights('star', 5)**.5))
    rng = np.random.default_rng(678)
    x = sample_tree(rng, p, 150000)
    xt = sampler.sample(rng, x)
    field = sampler.fields(rng, sample_tree(rng, p, 150000))
    radius = np.linalg.norm(field, axis=2)
    r = ((1 - (i1e(radius) / i0e(radius))**2) / 2).mean(axis=0)
    sigma = exact_sigma('star', 5, .8)
    residual = np.diag(np.repeat(r, 2))
    target = np.block([[sigma, sigma-residual], [sigma-residual, sigma]])
    design = np.c_[vec(x).reshape(len(x), 10), vec(xt).reshape(len(x), 10)]
    assert np.max(np.abs(np.cov(design, rowvar=False) - target)) < .008
