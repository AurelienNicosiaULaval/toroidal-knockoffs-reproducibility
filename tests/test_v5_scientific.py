import numpy as np
from v4.core import sample_pairs,pair_knockoffs,ridge_scores
from v5.harmonic_score import features,Moments


def test_streamed_score_matches_retained_dense_ridge():
    rng=np.random.default_rng(915);x=sample_pairs(rng,300,4,1.,1.)
    xt=pair_knockoffs(rng,x,.9,1.);y=rng.normal(size=len(x))
    d=np.c_[features(x),features(xt)];moment=Moments(d.shape[1],1)
    for chunk in np.array_split(np.arange(len(x)),7):
        moment.update(d[chunk],y[chunk])
    np.testing.assert_allclose(moment.scores(8,1)[:,0],ridge_scores(x,xt,y),atol=2e-13,rtol=2e-13)


def test_second_harmonic_score_swaps_entire_angle_and_is_rotation_equivariant():
    rng=np.random.default_rng(716);x=sample_pairs(rng,400,3,1.,1.)
    xt=pair_knockoffs(rng,x,.8,1.);y=np.cos(2*x[:,0])+rng.normal(size=len(x))
    def calculate(x,xt):
        d=np.c_[features(x),features(xt)];m=Moments(d.shape[1],1);m.update(d,y)
        return m.scores(6,2)[:,0]
    original=calculate(x,xt);swapped_x=x.copy();swapped_xt=xt.copy();indices=[1,4]
    swapped_x[:,indices]=xt[:,indices];swapped_xt[:,indices]=x[:,indices]
    expected=original.copy();expected[indices]*=-1
    np.testing.assert_allclose(calculate(swapped_x,swapped_xt),expected,atol=2e-13)
    rotation=rng.uniform(-np.pi,np.pi,6)
    np.testing.assert_allclose(calculate(x+rotation,xt+rotation),original,atol=2e-13)


def test_second_harmonic_mean_has_common_population_target_in_finite_design():
    rng=np.random.default_rng(889);x=sample_pairs(rng,1000,2,1.,1.)
    y=np.cos(2*x[:,0])+2*np.sin(x[:,2])
    for working in [.1,1.,2.]:
        xt=pair_knockoffs(rng,x,working,1.)
        d=np.c_[features(x),features(xt)];d-=d.mean(axis=0)
        coefficient=np.linalg.solve(d.T@d,d.T@(y-y.mean()))
        truth=np.zeros(d.shape[1]);truth[2]=1.;truth[9]=2.
        np.testing.assert_allclose(coefficient,truth,atol=2e-12)


def test_normal_integral_matches_exact_fair_sign_limit():
    from fractions import Fraction
    from v5.normal_integral import integrate
    exact=sum(Fraction(r,10+r)*Fraction(1,2**(r+1)) for r in range(10))+Fraction(1,2**11)
    assert abs(integrate(0.,64)['fdr']-float(exact))<2e-13
    assert abs(integrate(.657,64)['fdr']-integrate(.657,128)['fdr'])<2e-11
