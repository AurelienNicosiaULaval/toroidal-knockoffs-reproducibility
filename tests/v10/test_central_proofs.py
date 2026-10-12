from fractions import Fraction
from itertools import product
from pathlib import Path
import importlib.util
import numpy as np
from toroidal import select


def test_independent_exact_fourier_gaussian_coefficients():
    path=Path(__file__).resolve().parents[2]/'scripts/v10/verify_symbolic.py'
    spec=importlib.util.spec_from_file_location('verify_symbolic_v10',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result=module.verify()
    assert result['witness_cross_covariance_first_order']=='t/4'


def test_all_finite_sign_configurations_at_actual_knockoff_threshold():
    for p,signals in [(20,10),(18,10),(18,3),(18,0),(1,0),(1,1)]:
        nulls=p-signals
        total_fdp=Fraction(0)
        discovered=0
        for signs in product([-1,1],repeat=nulls):
            w=np.r_[np.full(signals,100.),np.array(signs)*np.arange(nulls,0,-1)]
            selected=select(w,.1)
            tp=int(np.sum(selected<signals))
            fp=len(selected)-tp
            total_fdp+=Fraction(fp,max(1,len(selected)))
            discovered+=int(tp>0)
        run_fdr=sum(Fraction(r,max(1,signals+r))*Fraction(1,2**(r+1))
                    for r in range(nulls) if signals+r>=10)
        if p>=10:
            run_fdr+=Fraction(nulls,p)*Fraction(1,2**nulls)
        assert total_fdp/Fraction(2**nulls)==run_fdr
        if signals==3:
            assert Fraction(discovered,2**nulls)==Fraction(1,128)
