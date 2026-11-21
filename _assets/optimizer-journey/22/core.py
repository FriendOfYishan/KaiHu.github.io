"""Exact two-point oracle checks of theory/22; not a training benchmark."""
import json
import math
import random
from pathlib import Path

rng = random.Random(221110)
T = 128
eta0 = 0.3
eps = 0.3
dp = 1 / eps
alpha = [0.0] + [max(2 / (t + 1), 0.1 / t ** (1 / 3)) for t in range(1, T + 2)]
eta = [0.0] + [eta0 / math.sqrt(t) for t in range(1, T + 1)]
R = [0.0] * (T + 2)
for t in range(T, 0, -1):
    R[t] = eta[t] + (1 - alpha[t + 1]) * R[t + 1]

counts = dict(cases=0, tracking_checks=0, potential_checks=0,
              certificate_checks=0, actual_potential_worse=0,
              actual_objective_worse=0)
max_identity_residual = 0.0

def leq(lhs, rhs):
    assert lhs <= rhs + 1e-9 * (1 + abs(lhs) + abs(rhs)), (lhs, rhs)

for _ in range(3000):
    t = rng.randrange(1, T + 1)
    L = 10 ** rng.uniform(-1, 1)
    x = rng.uniform(-3, 3)
    m = rng.uniform(-3, 3)
    v = m * m + rng.uniform(0, 4)
    sigma = rng.uniform(0, 2)
    D = 1 / (math.sqrt(v) + eps)
    u = D * m
    e = m - L * x
    a = alpha[t + 1]
    c = (1 - a) ** 2 / a
    knext = dp * R[t + 1] / 2
    kt = dp * R[t] / 2
    K = L + 1 / (dp * eta[t]) + 2 * knext * L * L * c
    star = 1 / (dp * K)
    via_rule = eta[t] / (1 + dp * L * eta[t] + (dp * L) ** 2 * eta[t] * R[t + 1] * c)
    assert math.isclose(star, via_rule, rel_tol=1e-12)
    assert 0 < star <= eta[t]
    cert = lambda s: s / dp - K * s * s / 2
    gap = cert(star) - cert(eta[t])
    identity = K * (eta[t] - star) ** 2 / 2
    max_identity_residual = max(max_identity_residual, abs(gap - identity))
    assert math.isclose(gap, identity, rel_tol=1e-9, abs_tol=1e-12)
    leq(cert(eta[t]), cert(star))
    counts['certificate_checks'] += 1
    potentials, objectives = [], []
    for s in [eta[t], star]:
        xn = x - s * u
        # F(x)=L*x^2/2. Next gradient oracle is L*xn +/- sigma.
        errors = [(1 - a) * m + a * (L * xn + sign * sigma) - L * xn for sign in [-1, 1]]
        en2 = sum(q*q for q in errors) / 2
        tracking_rhs = (1 - a) * e*e + c * L*L * s*s * u*u + a*a * sigma*sigma
        leq(en2, tracking_rhs)
        counts['tracking_checks'] += 1
        actual_delta = L * (xn*xn - x*x) / 2 + knext * en2 - kt * e*e
        certificate_rhs = -cert(s) * u*u + knext * a*a * sigma*sigma
        leq(actual_delta, certificate_rhs)
        counts['potential_checks'] += 1
        potentials.append(actual_delta)
        objectives.append(L * xn*xn / 2)
    counts['actual_potential_worse'] += int(potentials[1] > potentials[0] + 1e-12)
    counts['actual_objective_worse'] += int(objectives[1] > objectives[0] + 1e-12)
    counts['cases'] += 1

result = dict(status='passed', seed=221110, **counts,
              max_certificate_identity_residual=max_identity_residual,
              scope='Scalar quadratic; exact expectation over a two-point next-gradient oracle. Random fixed states, not optimizer training trajectories.',
              interpretation='Certificate dominance is verified, but does not imply actual potential or objective dominance.')
Path(__file__).with_name('results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
