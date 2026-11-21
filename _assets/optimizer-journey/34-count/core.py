"""Deterministic Poisson-sum checks for the count-calibrated 34 descendant.

These are mathematical checks, not GPU benchmarks or training experiments.
Run from the project root: python theory/verify_34_count_calibration.py
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import poisson


def gate(s, k, width):
    return np.clip((np.asarray(s) - k) / width, 0.0, 1.0)


def setup(a, gap, block):
    return math.ceil(block * (a + gap / 2)), max(2, math.ceil(block * gap / 8))


def poisson_grid(mean):
    # The included upper tail extends well beyond the 1 - 1e-14 quantile.
    top = int(poisson.ppf(1 - 1e-14, mean) + 12 * math.sqrt(mean + 1) + 50)
    s = np.arange(top + 1, dtype=np.float64)
    prob = poisson.pmf(s, mean)
    # PMF sums have a few ulps of error; retain the actual weights, no renormalizing.
    return s, prob


def measure(a, gap, upper, block, rate):
    k, width = setup(a, gap, block)
    s, prob = poisson_grid(block * rate)
    f = gate(s, k, width)
    fm = gate(s - 1, k, width)
    fp = gate(s + 1, k, width)
    core = a + s / block * fm - a * f
    estimate = np.minimum(2 * upper, core)
    naive = a + (s / block - a) * f
    result = {
        "a": a, "gap": gap, "upper": upper, "block": block, "rate": rate,
        "k": k, "width": width,
        "pmf_mass": float(prob.sum()),
        "core_mean": float(prob @ core),
        "mean_identity": float(a + (rate - a) * (prob @ f)),
        "core_mse": float(prob @ ((core - rate) ** 2)),
        "clipped_mse": float(prob @ ((estimate - rate) ** 2)),
        "naive_bias": float(prob @ (naive - rate)),
        "min_core_on_grid": float(core.min()),
        "recovery_probability": float(prob @ (np.abs(estimate - rate) <= gap / 4)),
    }
    if rate == a:
        result["null_risk_identity"] = float(
            a / block * (prob @ (f * f)) + a * a * (prob @ ((fp - f) ** 2))
        )
        result["naive_bias_identity"] = float(a * (prob @ (fp - f)))
        result["null_risk_upper"] = float(
            (a / block + a * a / width ** 2)
            * math.exp(-block * gap * gap / (12 * a))
        )
    return result


def theorem_block(a, gap, upper, risk, failure):
    return math.ceil(max(
        64 * a / gap ** 2,
        12 * a / gap ** 2 * max(0, math.log(gap ** 2 / risk)),
        40 * upper / gap ** 2 * math.log(2 / failure),
    ))


def main():
    records = []
    for a, gap, upper in [(1.0, 0.5, 2.0), (4.0, 2.0, 6.0), (8.0, 8.0, 16.0)]:
        for block in [8, 32, 128, 512]:
            for rate in [a, a + gap / 2, a + gap]:
                row = measure(a, gap, upper, block, rate)
                assert abs(row["core_mean"] - row["mean_identity"]) < 2e-9, row
                assert row["min_core_on_grid"] >= a / 2 - 1e-12, row
                assert row["clipped_mse"] <= row["core_mse"] + 1e-12, row
                if rate == a:
                    assert math.isclose(row["core_mse"], row["null_risk_identity"],
                                        rel_tol=2e-8, abs_tol=1e-13), row
                    assert math.isclose(row["naive_bias"], row["naive_bias_identity"],
                                        rel_tol=2e-8, abs_tol=1e-13), row
                    assert row["core_mse"] <= row["null_risk_upper"] + 1e-12, row
                records.append(row)

    a, gap, upper, failure = 4.0, 2.0, 6.0, 0.05
    comparisons = []
    for risk in [1e-2, 1e-3, 1e-4, 1e-5, 1e-8]:
        block = theorem_block(a, gap, upper, risk, failure)
        null = measure(a, gap, upper, block, a)
        post = measure(a, gap, upper, block, a + gap)
        assert null["clipped_mse"] <= risk
        assert post["recovery_probability"] >= 1 - failure
        information = (a + gap) * math.log1p(gap / a) - gap
        universal_lower = max(1, ((1 - failure) * math.log(gap ** 2 / (4 * risk))
                                  - math.log(2)) / information)
        # Search the exact Poisson-sum criterion at this one post-change rate.
        # This is not a uniform-in-rate numerical proof or a tuning experiment.
        for exact_block in range(1, block + 1):
            n0 = measure(a, gap, upper, exact_block, a)
            n1 = measure(a, gap, upper, exact_block, a + gap)
            if n0["clipped_mse"] <= risk and n1["recovery_probability"] >= 1 - failure:
                break
        comparisons.append({
            "risk_target": risk, "failure_target": failure,
            "theorem_block": block, "theorem_recovery_upper": 2 * block,
            "actual_null_mse_at_theorem_block": null["clipped_mse"],
            "actual_recovery_probability_at_theorem_block": post["recovery_probability"],
            "fixed_linear_half_mean_delay_lower": a / (4 * risk),
            "all_estimator_delay_lower": universal_lower,
            "first_numerically_feasible_block_for_rate_6_only": exact_block,
        })

    output = {
        "description": "Deterministic Poisson probability summation; no Monte Carlo or training.",
        "identity_cases": len(records),
        "maximum_mean_identity_absolute_error": max(
            abs(r["core_mean"] - r["mean_identity"]) for r in records),
        "comparison_model": {"baseline": a, "gap": gap, "upper": upper, "failure": failure},
        "comparisons": comparisons,
        "identity_records": records,
    }
    path = Path(__file__).with_name("34_COUNT_CALIBRATION_CHECKS.json")
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "identity_records"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
