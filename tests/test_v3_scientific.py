from fractions import Fraction
from itertools import product
import numpy as np
from toroidal import select
from v3.sign_limit import sign_limit
from v3.independent_counterexample import moments,quadrature


def test_sign_limit_matches_exhaustive_knockoff_threshold():
    for s in [0,2,4]:
        p=7;fdps=[];nonempty=[]
        for signs in product([-1,1],repeat=p-s):
            scores=np.r_[np.full(s,100.),np.arange(1,p-s+1)*signs]
            chosen=select(scores,.25)
            fdps.append(np.sum(chosen>=s)/max(1,len(chosen)));nonempty.append(bool(len(chosen)))
        exact=sign_limit(p,s,Fraction(1,4))
        assert abs(np.mean(fdps)-exact['fdr'])<1e-14
        assert abs(np.mean(nonempty)-exact['selection_probability'])<1e-14


def test_population_moments_against_independent_angular_integration():
    exact=moments(1.,1.)
    np.testing.assert_allclose(exact['coefficient'],quadrature(1.,1.,128),atol=1e-12)
    # The second angle is conditionally null for cos(2 X1); omitted harmonics
    # and an independent working law create a nonzero population score.
    assert exact['population_score'][1]>.04
