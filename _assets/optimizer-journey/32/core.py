"""Check raw-state algebra against explicit weighted history, without training."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(12032)
maximum_error = 0.0
checks = 0
for _ in range(30):
    m = np.zeros(9)
    h = np.zeros(9)
    z = 0.0
    samples = []
    weights = np.empty(0)
    c, epsilon = 2.0, 0.01
    for t in range(1, 201):
        beta = 1 - t ** (-1 / 3)
        g = rng.uniform(-1, 1, 9)
        m = beta * m + g
        h = beta * h + 1 / (c + g)
        z = beta * z + 1
        s = c * z + m
        raw_v = m * m + s * (s - z * z / h)
        raw_direction = m / (np.sqrt(np.maximum(raw_v, 0)) + epsilon * z)
        samples.append(g)
        weights = np.r_[beta * weights, 1.0]
        p = weights / weights.sum()
        history = np.asarray(samples)
        mean = p @ history
        arithmetic = c + mean
        harmonic = 1 / (p @ (1 / (c + history)))
        estimate = mean * mean + arithmetic * (arithmetic - harmonic)
        expected = mean / (np.sqrt(np.maximum(estimate, 0)) + epsilon)
        assert np.allclose(m, weights @ history, atol=1e-12)
        assert np.allclose(raw_v / z**2, estimate, atol=1e-12)
        assert np.allclose(raw_direction, expected, atol=1e-10)
        maximum_error = max(maximum_error, float(np.max(np.abs(raw_direction - expected))))
        checks += 1
report = {"checks": checks, "maximum_update_abs_error": maximum_error,
          "all_passed": True, "scope": "Algebra verification only; no timing or training claim"}
Path(__file__).with_name("32_RAW_WEIGHTED_HARMONIC_VERIFICATION.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
