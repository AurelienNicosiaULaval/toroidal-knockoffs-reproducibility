"""Exact strong-signal knockoff+ limit when q*p < 2, using fair null signs.

This is a consequence of the standard sign-flip property, not a new general
FDR theorem. No estimated covariance or Monte Carlo approximation is used.
"""
from __future__ import annotations
from fractions import Fraction
import json
from pathlib import Path


def sign_limit(p:int,s:int,q:Fraction=Fraction(1,10)) -> dict:
    if not 0<=s<=p or q<=0 or q>=1 or q*p>=2:raise ValueError('Requires 0 <= s <= p, 0 < q < 1, q*p < 2')
    floor=(q.denominator+q.numerator-1)//q.numerator;null=p-s
    fdr=Fraction(0);nonempty=Fraction(0)
    for run in range(null+1):
        probability=Fraction(1,2**(run+1)) if run<null else Fraction(1,2**null)
        discoveries=s+run
        if discoveries>=floor:
            fdr+=probability*Fraction(run,discoveries);nonempty+=probability
    return dict(p=p,signals=s,q=str(q),fdr=float(fdr),fdr_fraction=str(fdr),
        selection_probability=float(nonempty),selection_probability_fraction=str(nonempty),
        power=float(nonempty) if s else None)


if __name__=='__main__':
    result=[sign_limit(18,s) for s in [0,3,10]]
    Path('results/v3/validation/sign_limits.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
