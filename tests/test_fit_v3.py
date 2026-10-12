import numpy as np
from scipy.optimize._numdiff import approx_derivative
from fit_model import loss_grad
from toroidal import vec
from v3.fit_verified import hessian,fit_verified


def test_analytic_joint_hessian_against_gradient_difference():
    rng=np.random.default_rng(319);x=rng.uniform(-np.pi,np.pi,(80,4));edges=np.array([[0,1],[1,2],[1,3]])
    theta=rng.normal(0,.3,20);penalty=.017
    numeric=approx_derivative(lambda b:loss_grad(b,vec(x),edges,penalty)[1],theta,method='3-point')
    actual=hessian(theta,x,edges,penalty)
    assert np.max(abs(numeric-actual))<2e-9
    assert np.linalg.eigvalsh(actual).min()>penalty


def test_regularized_fit_stationarity_and_translation_equivariance():
    rng=np.random.default_rng(31);x=rng.uniform(-np.pi,np.pi,(120,3));edges=np.array([[0,1],[1,2]])
    q,d=fit_verified(x,edges)
    assert d['gradient_l2']<=d['tolerance']
    assert d['global_objective_gap_upper']<=d['tolerance']**2/(2*d['penalty'])
    q2,_=fit_verified(x+2*np.pi,edges)
    assert np.allclose(q.h,q2.h,atol=1e-8)
    assert np.allclose(q.M,q2.M,atol=1e-8)
