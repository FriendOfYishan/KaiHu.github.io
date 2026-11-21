"""Local algebra checks for theory 31, not training or a proof of rates."""

import json
from pathlib import Path

import numpy as np


rng = np.random.default_rng(12031)
moment_cases = 0
max_identity_error = 0.0
for c in [1.0001, 1.1, 2.0, 4.0, 10.0]:
    bound_ratio = (c + 1) / (c - 1)
    for _ in range(1000):
        g = rng.uniform(-1, 1, 25)
        p = rng.dirichlet(np.ones(25))
        m = p @ g
        variance = p @ ((g - m) ** 2)
        v = p @ (g * g)
        a = c + m
        harmonic = 1 / (p @ (1 / (c + g)))
        estimate = m * m + a * (a - harmonic)
        weighted_variance = p @ ((g - m) ** 2 * harmonic / (c + g))
        identity_error = abs(a * (a - harmonic) - weighted_variance)
        max_identity_error = max(max_identity_error, identity_error)
        assert identity_error < 1e-11
        assert abs(estimate - v) <= (bound_ratio - 1) * variance + 1e-11
        assert v / bound_ratio - 1e-11 <= estimate <= bound_ratio * v + 1e-11
        assert m * m - 1e-11 <= estimate <= bound_ratio + 1e-11
        epsilon = 0.01
        ratio = (np.sqrt(v) + epsilon) / (np.sqrt(estimate) + epsilon)
        assert 1 / np.sqrt(bound_ratio) - 1e-11 <= ratio <= np.sqrt(bound_ratio) + 1e-11
        moment_cases += 1

weight_steps = 0
for horizon in [1, 2, 10, 100, 1000, 10000]:
    forgetting = horizon ** -0.5
    z = 0.0
    square_sum = 0.0
    for t in range(1, horizon + 1):
        z = (1 - forgetting) * z + 1
        kappa = 1 / z
        closed = forgetting / (1 - (1 - forgetting) ** t)
        assert np.isclose(kappa, closed, rtol=1e-12)
        assert forgetting - 1e-12 <= kappa <= forgetting + 1 / t + 1e-12
        square_sum += kappa * kappa
        weight_steps += 1
    assert square_sum <= 2 * horizon * forgetting**2 + np.pi**2 / 3

tracking_cases = 0
for _ in range(5000):
    # Exact two-outcome conditional expectation; no independence assumption
    # between the present noise and the present preconditioner is used.
    kappa = rng.uniform(0.001, 1)
    old_error, drift, noise = rng.normal(size=(3, 8))
    base = (1 - kappa) * (old_error + drift)
    actual = (np.sum((base + kappa * noise) ** 2)
              + np.sum((base - kappa * noise) ** 2)) / 2
    bound = ((1 - kappa) * np.sum(old_error**2)
             + np.sum(drift**2) / kappa
             + kappa**2 * np.sum(noise**2))
    assert actual <= bound + 1e-10
    tracking_cases += 1

descent_cases = 0
dimension = 8
gradient_bound = 1.5
shift = 3.0
epsilon = 0.1
ratio_bound = (shift + gradient_bound) / (shift - gradient_bound)
lower = 1 / (gradient_bound * np.sqrt(ratio_bound) + epsilon)
upper = 1 / epsilon
for _ in range(3000):
    # F(x)=sqrt(1+||x||^2) is globally 1-smooth and lower bounded.
    x = rng.normal(size=dimension)
    function_value = np.sqrt(1 + x @ x)
    true_gradient = x / function_value
    current_gradient = true_gradient + rng.choice([-0.3, 0.3], size=dimension)
    history = rng.uniform(-gradient_bound, gradient_bound, (12, dimension))
    history[-1] = current_gradient
    weights = rng.dirichlet(np.ones(12))
    m = weights @ history
    arithmetic = shift + m
    harmonic = 1 / (weights @ (1 / (shift + history)))
    estimate = m * m + arithmetic * (arithmetic - harmonic)
    direction = m / (np.sqrt(estimate) + epsilon)
    error = m - true_gradient
    eta = float(rng.uniform(0.001, 0.5))
    next_x = x - eta * direction
    actual = np.sqrt(1 + next_x @ next_x)
    certificate = (function_value - lower * eta / 2 * (true_gradient @ true_gradient)
                   + upper * eta / 2 * (error @ error) + dimension * eta**2 / 2)
    assert actual <= certificate + 1e-11
    assert np.sum(direction**2) <= dimension + 1e-11
    descent_cases += 1

report = {
    "purpose": "CPU algebra checks only; no GPU, training, or empirical rate claim",
    "moment_bound_cases": moment_cases,
    "max_variance_identity_abs_error": max_identity_error,
    "normalized_weight_steps": weight_steps,
    "conditional_tracking_cases": tracking_cases,
    "pathwise_descent_cases": descent_cases,
    "all_checks_passed": True,
}
destination = Path(__file__).with_name("31_ARITHMETIC_HARMONIC_VERIFICATION.json")
destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
