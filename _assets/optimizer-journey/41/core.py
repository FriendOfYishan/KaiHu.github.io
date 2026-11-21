"""Local algebra / finite-noise checks for candidate 41; no network or training.

Sources are the workspace derivations in 40 and 41. Full N and noise-tree
branches below are verification-only and are not optimizer state.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def gain(m, q, beta=.9, eps2=1e-16):
    return 1 - (1 + beta) / (4 * beta) * q / (m * m + q + eps2)


def layer_scale(v, step, rho=.999, eps2=1e-16):
    if step == 1:
        return np.ones_like(v)
    return np.clip(1 / np.sqrt(v / (1 - rho ** (step - 1)) + eps2),
                   1e-3, 1e4)


def main():
    rng = np.random.default_rng(411115)
    beta, rho, eps2 = .9, .999, 1e-16
    a, kappa = 1 - beta, beta / (1 - beta)
    c, floor = (1 + beta) / (4 * beta), (3 * beta - 1) / (4 * beta)
    out = {"seed": 411115, "beta": beta, "rho": rho,
           "gain_lower_bound": floor}

    # Endpoint maximization is exact: the regret is convex in theta.
    worst_minimax_violation = worst_cost_error = worst_certificate_error = 0.
    min_certificate_improvement = float("inf")
    for _ in range(2000):
        s = np.exp(rng.normal(0, 2, 17))
        n = np.exp(rng.normal(0, 2, 17))
        v, p = s + n, s / (s + n)
        kr = (1 + p) / 2
        opt_cost = n * n / (4 * v)
        test = rng.uniform(-.5, 1.5, 17)
        cost = v * np.maximum((test - p) ** 2, (test - 1) ** 2)
        worst_minimax_violation = max(worst_minimax_violation,
                                      float(np.max(opt_cost - cost)))
        e = rng.normal(0, .2, 17)
        actual_cost = v * np.maximum((kr + e - p) ** 2, (kr + e - 1) ** 2)
        cost_formula = opt_cost + n * np.abs(e) + v * e * e
        worst_cost_error = max(worst_cost_error,
                              float(np.max(np.abs(actual_cost - cost_formula))))

        W = v.sum()
        avg = s.sum() / W
        dispersion = np.dot(v, (p - avg) ** 2) / W
        score = np.dot(kr, s) ** 2 / np.dot(kr * kr, v)
        baseline = s.sum() ** 2 / W
        improvement = W * dispersion * (2 * avg + avg ** 2 + dispersion) / (
            (1 + avg) ** 2 + dispersion)
        worst_certificate_error = max(worst_certificate_error,
                                      abs(float(score - baseline - improvement)))
        min_certificate_improvement = min(min_certificate_improvement,
                                          float(score - baseline))
    assert worst_minimax_violation < 1e-10
    assert worst_cost_error < 1e-10
    assert worst_certificate_error < 1e-9
    assert min_certificate_improvement > 0
    out["minimax_and_certificate"] = {
        "cases": 2000, "coordinates_per_case": 17,
        "minimax_violation": worst_minimax_violation,
        "regret_cost_identity_max_error": worst_cost_error,
        "certificate_identity_max_error": worst_certificate_error,
        "minimum_certificate_improvement": min_certificate_improvement}

    # Degenerate zero/noise-free cases and the old .1-floor rule.
    p = np.linspace(0., 1. - 1e-6, 10001)
    new_cost = (1 - p) ** 2 / 4
    old = np.maximum(p, .1)
    old_cost = np.maximum((old - p) ** 2, (old - 1) ** 2)
    ratios = old_cost / new_cost
    assert ratios.min() >= 3.24 - 1e-10 and ratios.max() <= 4 + 1e-10
    assert np.all(gain(np.zeros(10), np.zeros(10)) == 1)
    assert np.all(gain(np.ones(10), np.zeros(10)) == 1)
    out["old_floored_oracle_regret_ratio_range"] = [float(ratios.min()),
                                                   float(ratios.max())]

    # Exact stationary first/second moments with fixed gains, no Monte Carlo.
    mu = np.array([.0, .01, .4, -1.2, .1])
    nu = np.array([1., 2., .5, .1, 2.])
    fixed = np.array([.3, .4, .2, .7, .9])
    mean, var, en = np.zeros(5), np.zeros(5), np.zeros(5)
    for _ in range(40000):
        en = rho * en + (1 - rho) * beta * (
            (fixed * mu - mean) ** 2 + fixed ** 2 * nu + var)
        mean = beta * mean + a * fixed * mu
        var = beta ** 2 * var + a * a * fixed ** 2 * nu
    em2 = mean * mean + var
    calibrated = 1 - c * en / (em2 + en)
    ideal = 1 - nu / (2 * (mu * mu + nu))
    err = float(np.max(np.abs(calibrated - ideal)))
    assert err < 1e-11
    out["ratio_of_expectations_calibration_error"] = err

    # Compression sensitivity, including nonzero epsilon.
    comp_error = 0.
    for _ in range(2000):
        m = rng.normal(size=31)
        n = np.exp(rng.normal(size=31))
        q = np.full(31, n.mean())
        e2 = 10 ** rng.uniform(-16, 1)
        lhs = np.abs(gain(m, n, eps2=e2) - gain(m, q, eps2=e2))
        rhs = c * (m * m + e2) * np.abs(n - q) / (
            (m * m + n + e2) * (m * m + q + e2))
        comp_error = max(comp_error, float(np.max(np.abs(lhs - rhs))))
        k = gain(m, q, eps2=e2)
        assert np.all(k >= floor - 1e-14) and np.all(k <= 1.)
    assert comp_error < 1e-12
    out["compression_sensitivity_identity_max_error"] = comp_error

    # Actual state recurrence: q=P N, virtual point, and summed EMA energy.
    m, n = np.zeros((4, 13)), np.zeros((4, 13))
    q, v = np.zeros((4, 1)), np.zeros((4, 1))
    x = rng.normal(size=m.shape)
    eta = .002
    comp_path_error = virtual_error = 0.
    input_energy = history_energy = 0.
    for t in range(1, 1001):
        g = rng.normal(size=m.shape) + np.sin(t / 31) * np.arange(4)[:, None]
        A = layer_scale(v, t) * gain(m, q)
        y, z = A * g, x - eta * kappa * m
        e = y - m
        history_energy += float(np.sum(m * m))
        input_energy += float(np.sum(y * y))
        new_m = beta * m + a * y
        new_x = x - eta * new_m
        new_z = new_x - eta * kappa * new_m
        virtual_error = max(virtual_error, float(np.max(np.abs(new_z - z + eta * y))))
        n = rho * n + (1 - rho) * beta * e * e
        q = rho * q + (1 - rho) * beta * np.mean(e * e, axis=1, keepdims=True)
        v = rho * v + (1 - rho) * np.mean(g * g, axis=1, keepdims=True)
        comp_path_error = max(comp_path_error,
                              float(np.max(np.abs(q - n.mean(axis=1, keepdims=True)))))
        x, m = new_x, new_m
    assert comp_path_error < 1e-12 and virtual_error < 1e-12
    assert history_energy <= input_energy
    out["pathwise_checks"] = {
        "steps": 1000, "residual_group_identity_error": comp_path_error,
        "virtual_iterate_identity_error": virtual_error,
        "summed_old_state_over_input_energy": history_energy / input_energy}

    # All 4^8 outcomes, with the actual adaptive candidate, on F=sum(1-cos x).
    T, eta = 8, .005
    x0 = np.array([.6, -.8])
    x, m = x0[None, :].copy(), np.zeros((1, 2))
    q, v = np.zeros((1, 1)), np.zeros((1, 1))
    noises = .3 * np.array(list(itertools.product([-1., 1.], repeat=2)))
    sumD = sumV = sumF2 = 0.
    H_tree = 1.
    for t in range(1, T + 1):
        h = layer_scale(v, t)
        H_tree = max(H_tree, float(h.max()))
        A = h * gain(m, q)
        f = np.sin(x)
        sumD += float(np.mean(np.sum(A * f * f, axis=1)))
        sumV += float(np.mean(np.sum(A * A * .09, axis=1)))
        sumF2 += float(np.mean(np.sum(f * f, axis=1)))
        g = (f[:, None, :] + noises[None, :, :]).reshape(-1, 2)
        old = np.repeat(m, 4, axis=0)
        y = np.repeat(A, 4, axis=0) * g
        e = y - old
        q = rho * np.repeat(q, 4, axis=0) + (1 - rho) * beta * np.mean(
            e * e, axis=1, keepdims=True)
        v = rho * np.repeat(v, 4, axis=0) + (1 - rho) * np.mean(
            g * g, axis=1, keepdims=True)
        m = beta * old + a * y
        x = np.repeat(x, 4, axis=0) - eta * m
    alpha, variance = sumD / sumF2, sumV / T
    deltaF = float(np.sum(1 - np.cos(x0)))
    eta_limit = min(1 / (4 * H_tree), 1 / (2 * np.sqrt(2) * H_tree * kappa))
    bound = 2 * deltaF / (alpha * eta * T) + (
        eta + 2 * H_tree * eta ** 2 * kappa ** 2) * variance / alpha
    assert eta <= eta_limit and sumF2 / T <= bound + 1e-10
    out["adaptive_exact_noise_tree"] = {
        "horizon": T, "outcomes": len(x), "L": 1., "eta": eta,
        "eta_limit_using_entire_tree_scale_bound": eta_limit,
        "entire_tree_scale_bound": H_tree, "effective_alpha": alpha,
        "average_gradient_squared": sumF2 / T,
        "average_filtered_variance": variance, "theorem_bound": bound,
        "note": "Conservative finite-tree bound check, not empirical convergence rate."}

    # Actual loss comparison with Adam at matched signal displacement.
    d, etaA = 32, .1
    old = np.zeros(d)
    old[0] = 1.
    k = gain(old, 1., eps2=0.)
    signal = beta + a * k[0]
    eta41 = etaA / signal
    lossA = 1 - np.sin(etaA) + .5 * (d - 1) * (etaA * a) ** 2
    loss41 = 1 - np.sin(etaA) + .5 * (d - 1) * (eta41 * a * floor) ** 2
    exact_gap = .5 * (d - 1) * (etaA * a) ** 2 * ((floor / signal) ** 2 - 1)
    assert loss41 < lossA and abs(loss41 - lossA - exact_gap) < 1e-12
    assert abs(k[0] - 53 / 72) < 1e-14 and abs(signal - 701 / 720) < 1e-14
    out["actual_nonconvex_adam_one_step"] = {
        "dimension": d, "active_gain": float(k[0]), "noise_gain": floor,
        "signal_direction": float(signal), "adam_lr": etaA,
        "candidate_lr": float(eta41), "adam_expected_loss": float(lossA),
        "candidate_expected_loss": float(loss41), "loss_difference": float(exact_gap),
        "noise_cost_ratio": float((floor / signal) ** 2),
        "scope": "One prescribed common state, not full training trajectories."}

    # A positive structural case and a negative comparison to full second moment.
    d, r = 100, 7
    out["sparse_signal_certificate_factor"] = d / (r + (d - r) / 4)
    s, v = np.array([.5, 2.]), np.array([1., 16.])
    kr, ka = (1 + s / v) / 2, 1 / np.sqrt(v)
    score = lambda k: float(np.dot(k, s) ** 2 / np.dot(k * k, v))
    counterexample = {"s": s.tolist(), "v": v.tolist(),
                      "identity_certificate": score(np.ones(2)),
                      "candidate_oracle_certificate": score(kr),
                      "full_second_moment_certificate": score(ka)}
    assert counterexample["identity_certificate"] < score(kr) < score(ka)
    out["full_second_moment_wins_counterexample"] = counterexample
    out["conditional_complexity_factor_at_delta_002"] = (.25 + .04) / .98 ** 2

    report = Path(__file__).parents[1] / "experiments/20261114_40s_12h/final_summary.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))["groups"]["lm"]
        out["existing_lm_val_minus_train_probe"] = {
            name: row["val_mean"] - row["train_eval"] for name, row in data.items()}
    out["scope"] = ("Algebra, exact moments, and constructed finite-noise objectives only; "
                    "no new CIFAR/LM runs, timing, or external novelty search.")
    dest = Path(__file__).with_name("41_RESIDUAL_AMBIGUITY_CHECKS.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
