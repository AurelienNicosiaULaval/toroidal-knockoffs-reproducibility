import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import i0e
from scipy.stats import multivariate_normal
from v4.core import sample_pairs,pair_knockoffs
from v6.core import fit_separate,witnesses


def test_constant_interaction_matches_retained_generator():
    x=sample_pairs(np.random.default_rng(106001),83,4,1.,.7)
    a=pair_knockoffs(np.random.default_rng(97),x,1.2,.7)
    b=witnesses(np.random.default_rng(97),x,np.full(4,1.2),.7)
    np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):witnesses(np.random.default_rng(1),x,np.ones(3),.7)


def test_separate_mle_against_independent_scalar_optimization():
    x=sample_pairs(np.random.default_rng(106002),100,3,1.,1.)
    fit=fit_separate(x)
    for j in range(3):
        mean=np.cos(x[:,2*j]-x[:,2*j+1]).mean()
        other=minimize_scalar(lambda g:np.log(i0e(g))+g-g*mean,bounds=(0,8),method='bounded')
        np.testing.assert_allclose(fit[j],other.x,rtol=3e-6,atol=3e-6)
    zeros=np.c_[np.zeros(8),np.full(8,np.pi)]
    assert fit_separate(zeros)[0]==0


def test_gaussian_conditional_sign_odds():
    k=10;t=.83;cov=np.eye(k)+t*t*np.ones((k,k));alpha=t*t/(1+k*t*t)
    w=np.random.default_rng(106003).normal(size=k);w[3]=abs(w[3]);swapped=w.copy();swapped[3]*=-1
    logratio=multivariate_normal.logpdf(w,cov=cov)-multivariate_normal.logpdf(swapped,cov=cov)
    np.testing.assert_allclose(logratio,2*alpha*w[3]*(w.sum()-w[3]),atol=1e-13)


def test_covariance_jackknife_against_direct_leave_one_out():
    from v6.aggregate import covariance_summary
    x=np.random.default_rng(106004).normal(size=(25,4))
    result,jack=covariance_summary(x)
    np.testing.assert_allclose(result['covariance'],np.cov(x,rowvar=False),atol=1e-14)
    for i in range(len(x)):
        c=np.cov(np.delete(x,i,axis=0),rowvar=False)
        correlation=np.corrcoef(np.delete(x,i,axis=0),rowvar=False)
        expected=[np.diag(c).mean(),c[~np.eye(4,dtype=bool)].mean(),correlation[~np.eye(4,dtype=bool)].mean()]
        np.testing.assert_allclose(jack[i],expected,atol=1e-14)
