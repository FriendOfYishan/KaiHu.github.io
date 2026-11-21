"""Local algebra checks for theory 33; no training or timing claims."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(12033)
p, c, g_bound, epsilon = 1 / 3, 2.0, 1.0, 0.01
k = (c + g_bound) / (c - g_bound)
cases = 0
max_update_error = 0.0
max_weight_sum_error = 0.0
for _ in range(20):
    m, h = np.zeros(7), np.zeros(7)
    weights, history = np.empty(0), []
    for t in range(1, 151):
        beta = 0.0 if t == 1 else (t**p - 1) / (t - 1)**p
        assert 0 <= beta < 1
        g = rng.uniform(-g_bound, g_bound, 7)
        m = beta * m + g
        h += 1 / (c + g)
        weights = np.r_[weights * beta, 1.0]
        history.append(g)
        samples = np.asarray(history)
        normalized_weights = weights / t**p
        max_weight_sum_error = max(max_weight_sum_error, abs(weights.sum() - t**p))
        assert np.isclose(weights.sum(), t**p)
        mean = normalized_weights @ samples
        a = c + mean
        ordinary_h = 1 / np.mean(1 / (c + samples), axis=0)
        weighted_h = 1 / (normalized_weights @ (1 / (c + samples)))
        estimate = mean**2 + a * np.maximum(a - ordinary_h, 0)
        raw_a = c * t**p + m
        raw_v = m**2 + raw_a * np.maximum(raw_a - t**(p + 1) / h, 0)
        raw_update = m / (np.sqrt(raw_v) + epsilon * t**p)
        normalized_update = mean / (np.sqrt(estimate) + epsilon)
        update_error = float(np.max(np.abs(raw_update - normalized_update)))
        max_update_error = max(max_update_error, update_error)
        assert update_error < 1e-10
        assert np.max(np.abs(raw_update)) <= 1 + 1e-10
        true_v = normalized_weights @ (samples**2)
        variance = normalized_weights @ ((samples - mean)**2)
        mismatch = np.abs(ordinary_h - weighted_h)
        bound = (k - 1) * variance + a * mismatch
        tv = np.abs(normalized_weights - 1 / t).sum() / 2
        global_bound = (k - 1) * variance + 2 * g_bound * (c + g_bound) * min(1, k * tv)
        assert np.all(np.abs(estimate - true_v) <= bound + 1e-10)
        assert np.all(np.abs(estimate - true_v) <= global_bound + 1e-10)
        assert np.all(estimate <= 3 * g_bound**2 + 2 * c * g_bound + 1e-10)
        cases += 1

g1, g2 = 1.0, -(2**p - 1)
ordinary_h2 = 2 / (1 / (c + g1) + 1 / (c + g2))
unprotected_estimate = c * (c - ordinary_h2)
assert unprotected_estimate < 0

drift_examples = []
for horizon in [1000, 10000, 100000]:
    recent = int(np.ceil(3 * horizon**p * np.log(horizon)))
    mean = 0.0
    true_v = 0.0
    for t in range(1, horizon + 1):
        g = float(t > horizon - recent)
        weight = t**(-p)
        mean = (1 - weight) * mean + weight * g
        true_v = (1 - weight) * true_v + weight * g * g
    ordinary_h = horizon / ((horizon - recent) / c + recent / (c + 1))
    estimate = mean**2 + (c + mean) * max(c + mean - ordinary_h, 0)
    drift_examples.append(dict(horizon=horizon, recent=recent, mean=mean,
                               true_second_moment=true_v, estimate=estimate,
                               absolute_error=abs(estimate - true_v)))

report = dict(algebra_cases=cases, max_update_error=max_update_error,
              max_raw_weight_sum_error=max_weight_sum_error,
              negative_without_protection=unprotected_estimate,
              horizon_dependent_drift_examples=drift_examples,
              all_checks_passed=True,
              scope="CPU algebra only; not an empirical optimization rate or timing benchmark")
Path(__file__).with_name("33_ORDINARY_HARMONIC_VERIFICATION.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
