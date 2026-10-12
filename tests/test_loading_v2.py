import numpy as np
import pytest
from scipy.optimize import minimize
from toroidal import Model, make_tree, EdgeSampler
from loading import AllocatedSampler, endpoint_allocations, perron_vector, psd_root
from metro_circle import CircularMetro
from metro_variants import PeriodicMetro


def test_perron_minimax_independent_convex_solver():
    # Solve in independent edge log-ratios plus a common maximum, without using PF.
    rng = np.random.default_rng(701)
    for p in (4, 7):
        edges = np.array([(j,k) for j in range(p) for k in range(j+1,p)])
        model = Model(rng.normal(size=(p,2)), edges, rng.normal(size=(len(edges),2,2)))
        s = np.linalg.norm(model.M, ord=2, axis=(1,2))
        def loads(z):
            d = np.zeros(p)
            np.add.at(d, edges[:,0], s*np.exp(z[:-1]))
            np.add.at(d, edges[:,1], s*np.exp(-z[:-1]))
            return d
        result = minimize(lambda z:z[-1], np.r_[np.zeros(len(edges)), 20.],
                          method='SLSQP', constraints={'type':'ineq','fun':lambda z:z[-1]-loads(z)},
                          options={'ftol':1e-10,'maxiter':1000})
        assert result.success
        u, rho = perron_vector(model)
        sampler = AllocatedSampler(model, endpoint_allocations(model,u))
        np.testing.assert_allclose(sampler.loads, rho, atol=2e-12)
        np.testing.assert_allclose(result.fun, max(rho), atol=2e-7)
        assert np.linalg.eigvalsh(sampler.aggregate()).min() > -1e-10


def test_star_tradeoff_and_disconnected_components():
    model = make_tree('star', 10)
    u, rho = perron_vector(model)
    symmetric = AllocatedSampler(model)
    balanced = AllocatedSampler(model, endpoint_allocations(model,u))
    np.testing.assert_allclose(balanced.loads, 3., atol=1e-13)
    assert symmetric.loads.sum() == 18
    assert abs(balanced.loads.sum()-30) < 1e-12
    model = Model(np.zeros((5,2)), [[0,1],[2,3]], [np.eye(2),2*np.eye(2)])
    u, rho = perron_vector(model)
    np.testing.assert_allclose(AllocatedSampler(model,endpoint_allocations(model,u)).loads,rho)


def test_augmentation_identity_general_asymmetric():
    rng = np.random.default_rng(44)
    model = Model(rng.normal(size=(4,2)),[[0,1],[1,2],[2,3]],rng.normal(size=(3,2,2)))
    sam = AllocatedSampler(model, endpoint_allocations(model, np.exp(rng.normal(size=4))))
    full = np.zeros((8,8))
    for j,k,bj,bk in sam.blocks:
        indices = [2*j,2*j+1,2*k,2*k+1]
        b = np.vstack([bj,bk]); full[np.ix_(indices,indices)] += b@b.T
    np.testing.assert_allclose(full, sam.aggregate(), atol=3e-13)
    # Compare the quadratic identity at independent random points.
    from toroidal import vec
    x = rng.uniform(-np.pi,np.pi,(100,4));v=vec(x).reshape(100,8)
    energy=.5*np.einsum('ni,ij,nj->n',v,full,v)+v@model.h.ravel()
    np.testing.assert_allclose(energy-model.log_unnormalized(x),sam.loads.sum()/2,atol=1e-12)
    with pytest.raises(ValueError): AllocatedSampler(model,np.zeros((3,2)))
    with pytest.raises(ValueError): psd_root(np.diag([1.,-.01]))


def test_fixed_order_sparse_recursion():
    rng=np.random.default_rng(47);model=make_tree('star',7)
    order=np.arange(7)[::-1];x=rng.uniform(-3,3,(57,7))
    proposals=rng.uniform(-3,3,x.shape);forced=rng.uniform(size=x.shape)<.5
    sparse=PeriodicMetro(model,order=order)
    full=CircularMetro(model,tau=0.,order=order)
    a,la=sparse.sample(rng,x,proposals=proposals,forced=forced)
    b,lb=full.sample(rng,x,proposals=proposals,forced=forced)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(la,lb,atol=1e-12)


def test_metro_more_than_64_variables():
    rng=np.random.default_rng(812);model=make_tree('chain',80)
    x=rng.uniform(-np.pi,np.pi,(11,80))
    xt,branch=PeriodicMetro(model).sample(rng,x)
    assert xt.shape==x.shape and np.all(np.isfinite(branch))
    assert all(type(k) is int for front in PeriodicMetro(model).front for k in front)
