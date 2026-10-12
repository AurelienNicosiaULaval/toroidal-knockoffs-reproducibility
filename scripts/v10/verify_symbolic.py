"""Independent exact Fourier/Gaussian calculation of the small-parameter lemma.

This verifies finite Taylor coefficients, not the analytic remainder bounds.
No retained simulator or population integration routine is imported.
"""
from __future__ import annotations

import json
from pathlib import Path
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]


def verify() -> dict:
    x, y, wx, wy, t, g, k = sp.symbols('x y wx wy t g k', real=True)
    z1, z2 = sp.symbols('z1 z2', nonzero=True)
    cos1, cos2 = (z1 + 1 / z1) / 2, (z2 + 1 / z2) / 2
    sin1, sin2 = (z1 - 1 / z1) / (2 * sp.I), (z2 - 1 / z2) / (2 * sp.I)
    response = (z1**2 + z1**-2) / 2
    interaction = (z1 / z2 + z2 / z1) / 2

    def integrate_uniform(f):
        return sp.expand(f).coeff(z1, 0).coeff(z2, 0)

    def generator(f):
        return sp.expand(wx * sp.diff(f, x) + wy * sp.diff(f, y)
                         + (sp.diff(f, x, 2) + sp.diff(f, y, 2)) / 2)

    def at_zero(f):
        return sp.expand(f).subs({x: 0, y: 0})

    def semigroup_jet(f):
        return at_zero(f) + t * at_zero(generator(f)) + t**2 * at_zero(generator(generator(f))) / 2

    def truncate_total(f, degree=2):
        poly = sp.Poly(sp.expand(f), g, t)
        return sp.expand(sum(c * g**a * t**b for (a, b), c in poly.terms() if a + b <= degree))

    radius2 = x**2 + y**2
    vm_mean_jet = x / 2 - x * radius2 / 16 + x * radius2**2 / 96
    conditional_mean = semigroup_jet(vm_mean_jet)
    field_derivative = semigroup_jet(sp.diff(vm_mean_jet, x))
    assert sp.expand(conditional_mean - (t * wx / 2 - t**2 * wx / 4)) == 0
    assert sp.expand(field_derivative - (sp.Rational(1, 2) - t / 4
                     + t**2 * (sp.Rational(1, 4) - (3 * wx**2 + wy**2) / 16))) == 0
    # The common Gaussian field contributes t/4 to witness-witness covariance.
    common_covariance = semigroup_jet(vm_mean_jet**2)
    assert sp.expand(common_covariance).coeff(t, 1) == sp.Rational(1, 4)

    angular = {wx: cos1 + cos2, wy: sin1 + sin2}
    density_jet = 1 + g * interaction + g**2 * (interaction**2 / 2 - sp.Rational(1, 4))
    witness = conditional_mean.subs(angular)
    shifted_derivative = field_derivative.subs(angular)
    u = sp.Matrix([
        truncate_total(integrate_uniform(response * cos1 * cos2 * density_jet)),
        truncate_total(integrate_uniform(response * cos2**2 * density_jet)),
        truncate_total(integrate_uniform(response * witness * cos2 * density_jet)),
        truncate_total(integrate_uniform(response * (witness * cos2 + shifted_derivative) * density_jet)),
    ])
    expected_u = sp.Matrix([g / 8, g**2 / 32, t * g / 16, t * g / 16 - t**2 / 32])
    assert all(sp.simplify(a - b) == 0 for a, b in zip(u, expected_u))
    covariance_first_order = sp.Matrix([
        [0, g / 4, t / 4, t / 4],
        [g / 4, 0, t / 4, t / 4],
        [t / 4, t / 4, 0, t / 4],
        [t / 4, t / 4, t / 4, 0],
    ])
    coefficient = 2 * u - 4 * covariance_first_order * sp.Matrix([g / 8, 0, 0, 0])
    expected = sp.Matrix([g / 4, -g**2 / 16, 0, -t**2 / 16])
    assert all(sp.simplify(a - b) == 0 for a, b in zip(coefficient, expected))
    slope = sp.diff(-coefficient[1] + coefficient[3], t).subs(t, g)
    assert slope == -g / 8
    return dict(response_covariance_linear_field=[str(a * k) for a in u],
                coefficient_linear_field=[str(a * k) for a in coefficient],
                null_score_derivative=str(k * slope),
                witness_cross_covariance_first_order='t/4',
                scope='exact finite Taylor coefficients; remainders require the analytic proof',
                sympy_version=sp.__version__)


if __name__ == '__main__':
    result = verify()
    out = ROOT / 'results/v10/symbolic.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
