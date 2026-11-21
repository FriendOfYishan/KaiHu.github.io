"""CPU algebra checks for theory 34; not a training or runtime benchmark."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(12034)
error_cases = 0
max_identity_error = 0.0
for c in [1.01, 1.5, 2.0, 4.0, 10.0]:
    for _ in range(1000):
        g = rng.uniform(-1, 1, 31)
        weights = rng.dirichlet(np.ones(31))
        mean = weights @ g
        harmonic = 1 / (weights @ (1 / (c + g)))
        centered = weights @ (g / (c + g))
        reference = weights @ (c * g * g / (c + g))
        direct = c * mean - c * c + c**3 / harmonic
        reconstructed = c * (mean - c * centered)
        error = max(abs(reference - direct), abs(reference - reconstructed))
        max_identity_error = max(max_identity_error, error)
        assert error < 1e-10
        true_v = weights @ (g * g)
        signed_error = -(weights @ (g**3 / (c + g)))
        assert abs(reference - true_v - signed_error) < 1e-12
        assert c / (c + 1) * true_v - 1e-12 <= reference <= c / (c - 1) * true_v + 1e-12
        assert abs(reference - true_v) <= true_v / (c - 1) + 1e-12
        error_cases += 1

weight_cases = 0
max_reconstruction_error = 0.0
for a in [8.0, 10.0, 100.0, 10000.0]:
    c = 2.0
    m = np.zeros(5)
    h = np.zeros(5)
    raw_reference = np.zeros(5)
    n = 0.0
    for t in range(1, 301):
        beta = (t - 1) / (t + a - 2)
        n = beta * n + 1
        expected_n = (t + a - 1) / a
        assert np.isclose(n, expected_n, rtol=1e-12)
        g = rng.uniform(-1, 1, 5)
        m = beta * m + g
        h = beta * h + g / (c + g)
        raw_reference = beta * raw_reference + c * g * g / (c + g)
        reconstructed = c * (m - c * h)
        max_reconstruction_error = max(max_reconstruction_error,
                                       float(np.max(np.abs(raw_reference - reconstructed))))
        assert np.allclose(raw_reference, reconstructed, atol=1e-11)
        weight_cases += 1

# Check the parameter restrictions used in the Lyapunov proof across scales.
contraction_cases = 0
for _ in range(1000):
    smoothness = float(10 ** rng.uniform(-1, 1))
    pl = smoothness * float(rng.uniform(0.1, 1))
    epsilon = float(10 ** rng.uniform(-2, 0))
    c = 2.0
    lower = 1 / (np.sqrt(c / (c - 1)) + epsilon)
    upper = 1 / epsilon
    d0 = pl * lower / 2
    u = 2 * smoothness**2 * upper**2
    w = 4 * smoothness**3 * upper**2
    gamma = 0.8 * min(lower / (4 * smoothness * upper**2),
                      1 / (2 * np.sqrt(u)), np.sqrt(d0 / (4 * upper * w)))
    q = 2 * upper * gamma
    theta = min(d0 * gamma / 2, 0.25)
    assert d0 * gamma - q * w * gamma**2 >= theta - 1e-15
    assert 1 - u * gamma**2 - upper * gamma / q >= theta - 1e-15
    contraction_cases += 1

report = dict(moment_identity_and_bound_cases=error_cases,
              max_identity_abs_error=max_identity_error,
              cumulative_reconstruction_cases=weight_cases,
              max_reconstruction_abs_error=max_reconstruction_error,
              lyapunov_coefficient_cases=contraction_cases,
              all_checks_passed=True,
              scope="Algebra checks only; no claimed empirical rate or speedup")
Path(__file__).with_name("34_RECIPROCAL_RESIDUAL_VERIFICATION.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
