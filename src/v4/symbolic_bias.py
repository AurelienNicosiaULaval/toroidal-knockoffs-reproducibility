"""Exact polynomial verification of the local population-score expansion.

SymPy rational algebra, Gaussian moments and uniform circular monomial moments.
No floating-point quadrature enters these coefficients.
"""
from __future__ import annotations
import json
from pathlib import Path
import sympy as S


def normal_expectation(expr,variables):
    result=0
    for powers,coefficient in S.Poly(S.expand(expr),*variables).terms():
        weight=S.Integer(1)
        for degree in powers:
            if degree%2:weight=0;break
            if degree:weight*=S.factorial2(degree-1)
        result+=coefficient*weight
    return S.expand(result)


def main():
    a,zx,zy,wx,wy,k,g,t=S.symbols('a zx zy wx wy k g t',real=True)
    ex=k+a*zx+a*a*wx;ey=a*zy+a*a*wy;rr=ex*ex+ey*ey
    mean_series=S.expand(ex/S.Integer(2)-ex*rr/16+ex*rr*rr/96)
    expansion=normal_expectation(mean_series,[zx,zy])
    expansion=S.expand(expansion).coeff(k,1)
    result=S.expand(sum(expansion.coeff(a,j)*a**j for j in range(0,5)))
    expected=S.Rational(1,2)-a*a/4+a**4*(S.Rational(1,4)-(3*wx*wx+wy*wy)/16)
    if S.expand(result-expected)!=0:raise AssertionError('Conditional VM expansion')
    # Uniform-circle E[cos(2 X1) (3 wx^2+wy^2)] = 1/2,
    # where wx=cos X1+cos X2, wy=sin X1+sin X2.
    # The nonzero terms are 3 E[cos(2X) cos^2 X]+E[cos(2X) sin^2 X].
    circular=S.Rational(3,4)-S.Rational(1,4)
    if circular!=S.Rational(1,2):raise AssertionError('Circular moment')
    cov_first=S.Matrix([[0,g/4,t/4,t/4],[g/4,0,t/4,t/4],
                        [t/4,t/4,0,t/4],[t/4,t/4,t/4,0]])
    cross=S.Matrix([k*g/8,k*g*g/32,k*t*g/16,k*t*g/16-k*t*t/32])
    coefficients=S.expand(2*cross-4*cov_first*S.Matrix([k*g/8,0,0,0]))
    expected_beta=S.Matrix([k*g/4,-k*g*g/16,0,-k*t*t/16])
    if S.simplify(coefficients-expected_beta)!=S.zeros(4,1):raise AssertionError('Population solve')
    # Both null cosine coefficients are negative near t=g>0.
    null_score=-coefficients[1]+coefficients[3]
    slope=S.diff(null_score,t).subs(t,g)
    if S.simplify(slope+k*g/8)!=0:raise AssertionError('Null score slope')
    output=dict(sympy=S.__version__,conditional_linear_k=str(result),
        cross_moments=[str(v) for v in cross],covariance_linear=[[str(v) for v in row] for row in cov_first.tolist()],
        coefficient_terms=[str(v) for v in coefficients],null_score=str(null_score),slope_at_truth=str(slope),
        scope='Exact coefficients only. Taylor remainder justification is in the proof, not supplied by symbolic algebra.')
    path=Path('results/v4/theory/symbolic_bias.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
