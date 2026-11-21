"""Exact algebra checks for THEORY_V21_ZH.md; no sampling or training."""
from pathlib import Path
import json
import sympy as S

HERE = Path(__file__).resolve().parent
checks = {}


def equal(name, lhs, rhs):
    assert S.cancel(S.expand(lhs - rhs)) == 0, name
    checks[name] = "passed"


mu, h, x, s, k, m4, alpha, lam = S.symbols(
    "mu h x s k m4 alpha lam", real=True
)
mom = {0: 1, 1: 0, 2: s, 3: k, 4: m4}


def expectation(poly, variables, moments):
    result = 0
    for powers, coefficient in S.Poly(S.expand(poly), *variables).terms():
        term = coefficient
        for index, power in enumerate(powers):
            term *= moments[index][power]
        result += term
    return S.expand(result)


g = mu + x
nu = mu**2 + s
e = h - mu
q = g**2 - 2 * alpha * h * (g - h)
V = 4 * mu**2 * s + 4 * mu * k + m4 - s**2
C = 2 * mu * s + k
P = h * C
Q = h**2 * (s + e**2)
equal("conditional_risk", expectation((q - nu)**2, [x], [mom]),
      V - 4 * P * alpha + 4 * Q * alpha**2)
equal("psd_factorization", q,
      (g - alpha * h)**2 + alpha * (2 - alpha) * h**2)
equal("gradient_square_lower_bound_identity", q,
      (1 - alpha / 2) * g**2 + 2 * alpha * (h - g / 2)**2)
q2 = (1 - lam) * g**2 + lam * h**2
equal("v2_common_target_risk", expectation((q2 - nu)**2, [x], [mom]),
      (1 - lam)**2 * V + lam**2 * (h**2 - nu)**2)
theta = mu**2 + (1 - lam) * s
qn = (1 - lam) * q + lam * h**2
bias0 = lam * (h**2 - mu**2)
equal("nested_risk", expectation((qn - theta)**2, [x], [mom]),
      (1 - lam)**2 * V + bias0**2
      - 4 * (1 - lam) * alpha * h * ((1 - lam) * C - bias0 * e)
      + 4 * (1 - lam)**2 * alpha**2 * Q)

# Independent symbolic derivation of contemporaneous and lagged covariance.
E, X = S.symbols("E X", real=True)
H, Ke, Je, Jx = S.symbols("H Ke Je Jx", real=True)
A, B, U, L, N = S.symbols("A B U L N", real=True)
p, b, d1, z = S.symbols("p b d1 z", real=True)
emom = {0: 1, 1: 0, 2: H, 3: Ke, 4: Je + H**2}
xmom = {0: 1, 1: 0, 2: s, 3: k, 4: Jx + s**2}
qpast = mu**2 + 2 * mu * (A * X + B * E) + U * X**2 + L * E * X + N * E**2
mean = mu**2 + U * s + N * H
gamma0 = (4 * mu**2 * (A**2 * s + B**2 * H)
          + U**2 * Jx + L**2 * H * s + N**2 * Je
          + 4 * mu * (A * U * k + B * N * Ke))
equal("stationary_input_variance",
      expectation((qpast - mean)**2, [E, X], [emom, xmom]), gamma0)
K1 = (4 * mu**2 * B * (p * B * H + d1 * A * s)
      + 2 * mu * B * (p * N * Ke + d1 * U * k))
K2 = (2 * mu * N * (p**2 * B * Ke + d1**2 * A * k)
      + N * (p**2 * N * Je + d1**2 * U * Jx + 2 * p * d1 * L * H * s))
# z = p**(lag-1). Independent intermediate innovations integrate to a
# constant and zero linear terms, which do not affect the covariance.
future_predictor = z * (p * E + d1 * X)
conditional_future = 2 * mu * B * future_predictor + N * future_predictor**2
equal("all_lag_covariances",
      expectation(conditional_future * (qpast - mean), [E, X], [emom, xmom]),
      z * K1 + z**2 * K2)

stationary_risk = ((U - 1) * s + N * H)**2 + (1 - b) / (1 + b) * (
    gamma0 + 2 * b * K1 / (1 - p * b) + 2 * b * K2 / (1 - p**2 * b)
)
r = S.symbols("r", nonnegative=True)
gaussian = {s: 1, k: 0, Ke: 0, Jx: 2, H: S.Rational(1, 19),
            Je: S.Rational(2, 361), p: S.Rational(9, 10),
            b: S.Rational(999, 1000), d1: S.Rational(1, 10)}
base = S.expand(stationary_risk.subs(gaussian)).subs(mu**2, r)
actual21 = base.subs({A: 1 - alpha, B: alpha, U: 1, L: -2 * alpha, N: 2 * alpha})
actual2 = base.subs({A: 1 - lam, B: lam, U: 1 - lam, L: 0, N: lam})
c0 = 2 * (2 * r + 1) / S.Integer(1999)
a21 = 80 * (3625390 * r + 1926381791) / S.Integer(13893520102831)
b21 = -8 * (190810 * r - 1007991) / S.Integer(38486205271)
a2 = 80 * (3625390 * r + 156029256671) / S.Integer(13893520102831)
b2 = -8 * (190810 * r + 9122369) / S.Integer(38486205271)
equal("gaussian_v21_exact_polynomial", actual21, c0 + b21 * alpha + a21 * alpha**2)
equal("gaussian_v2_exact_polynomial", actual2, c0 + b2 * lam + a2 * lam**2)
factor = (S.Integer(10240000) * (38162 * r + 1009)
          * (72673232270 * r - 872239446817)
          / S.Integer(20395356987851769822486980744719))
equal("oracle_comparison_factorization", b21**2 * a2 - b2**2 * a21, factor)
fixed_gap = (32 * (1812695 * r + 92199712853)
             / S.Integer(13893520102831))
equal("fixed_coefficients_gap", actual2.subs(lam, S.Rational(1, 2))
      - actual21.subs(alpha, 1), fixed_gap)
for name, expr in [("v21_oracle_below_one", 2 * a21 + b21),
                   ("v2_oracle_below_one", 2 * a2 + b2)]:
    assert all(coef > 0 for coef in S.Poly(expr, r).all_coeffs())
    checks[name] = "passed"

# Exact finite-support example: sum over the population, never draw samples.
values = [S.Integer(-2), S.Integer(1)]
probs = [S.Rational(1, 3), S.Rational(2, 3)]
assert sum(pr * val for pr, val in zip(probs, values)) == 0
assert sum(pr * val**2 for pr, val in zip(probs, values)) == 2
assert sum(pr * val**3 for pr, val in zip(probs, values)) == -2
assert sum(pr * val**4 for pr, val in zip(probs, values)) == 6
for value in values:
    assert q.subs({mu: 1, h: 1, alpha: S.Rational(1, 2), x: value}) == 3
checks["skewed_exact_zero_risk_example"] = "passed"

rho, omega = S.symbols("rho omega", real=True)
vv = 4 * r + omega
equal("symmetric_ema_phase_identity", vv - omega * (1 + rho * vv),
      4 * r * (1 - rho * omega) - rho * omega**2)
threshold = S.Rational(872239446817, 72673232270)
output = {"scope": "exact symbolic mathematics only; no random samples or training",
          "checks": checks, "num_checks": len(checks),
          "stationary_gaussian_threshold_exact": str(threshold),
          "stationary_gaussian_threshold_decimal": str(S.N(threshold, 16))}
(HERE / "theory_v21_symbolic_checks.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(output, indent=2))
