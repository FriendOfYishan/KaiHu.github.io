"""Local algebra/representation checks, not Transformer training or a speed benchmark."""
from pathlib import Path
import json
import numpy as np


def reconstruct(r, c):
    total = r.mean()
    if total == 0.0:
        return np.zeros((r.size, c.size))
    return r[:, None] * (c[None, :] / total)


def compress(z):
    return reconstruct(z.mean(axis=1), z.mean(axis=0))


def direction(m, n, eps=1e-3):
    return m / np.sqrt(m * m + n + eps * eps)


def main():
    rng = np.random.default_rng(20261114)
    checks = {}
    identity_error = marginal_error = direction_identity_error = 0.0
    max_update = 0.0
    path_steps = 0
    for _ in range(80):
        nr, nc = rng.integers(2, 12, size=2)
        beta = rng.uniform(0.05, 0.99)
        m = np.zeros((nr, nc))
        v = np.zeros_like(m)
        n = np.zeros_like(m)
        r, c = np.zeros(nr), np.zeros(nc)
        for t in range(60):
            # Signed, nonstationary paths, including isolated large gradients.
            g = rng.normal(size=m.shape) + np.sin(t / 8) * rng.normal(size=(nr, 1))
            if t % 19 == 0:
                g[0, -1] *= 15
            e = g - m
            m = beta * m + (1 - beta) * g
            v = beta * v + (1 - beta) * g * g
            innovation = beta * e * e
            n = beta * n + (1 - beta) * innovation
            r = beta * r + (1 - beta) * innovation.mean(axis=1)
            c = beta * c + (1 - beta) * innovation.mean(axis=0)
            nhat = reconstruct(r, c)
            identity_error = max(identity_error, float(np.max(np.abs(v - m * m - n))))
            marginal_error = max(marginal_error, float(np.max(np.abs(r - n.mean(axis=1)))),
                                 float(np.max(np.abs(c - n.mean(axis=0)))))
            eps2 = 1e-6
            s = m * m + n + eps2
            shat = m * m + nhat + eps2
            direct = np.abs(direction(m, n) - direction(m, nhat))
            formula = np.abs(m) * np.abs(n - nhat) / (
                np.sqrt(s) * np.sqrt(shat) * (np.sqrt(s) + np.sqrt(shat)))
            direction_identity_error = max(direction_identity_error,
                                           float(np.max(np.abs(direct - formula))))
            max_update = max(max_update, float(np.max(np.abs(direction(m, nhat)))))
            path_steps += 1
    assert identity_error < 1e-11
    assert marginal_error < 1e-11
    assert direction_identity_error < 1e-12
    assert max_update <= 1.0
    checks['arbitrary_signed_paths'] = dict(
        steps=path_steps, full_moment_identity_max_error=identity_error,
        residual_marginal_max_error=marginal_error,
        direction_error_identity_max_error=direction_identity_error,
        max_abs_normalized_coordinate_update=max_update)

    # Different beta/rho: marginals still exactly track the defined residual statistic.
    m = np.zeros((7, 9))
    n = np.zeros_like(m)
    r, c = np.zeros(7), np.zeros(9)
    for _ in range(300):
        g = rng.normal(size=m.shape)
        e = g - m
        m = .9 * m + .1 * g
        z = .9 * e * e
        n = .97 * n + .03 * z
        r = .97 * r + .03 * z.mean(axis=1)
        c = .97 * c + .03 * z.mean(axis=0)
    err = max(np.max(np.abs(r-n.mean(axis=1))), np.max(np.abs(c-n.mean(axis=0))))
    assert err < 1e-12
    checks['separate_memory_coefficients'] = {'marginal_max_error': float(err)}

    cases = []
    for name, m, n in [
        ('separable_residual_nonseparable_signal',
         np.array([[1., 4.], [4., 1.]]), np.ones((2, 2))),
        ('separable_total_nonseparable_residual',
         np.sqrt(np.array([[17., 2.], [2., 17.]])),
         np.array([[1., 16.], [16., 1.]])),
    ]:
        v = m*m+n
        u = direction(m, n)
        residual_u = direction(m, compress(n))
        raw_u = m/np.sqrt(compress(v)+1e-6)
        scalar = float(np.sum(m*u)/np.sum(m*m))
        cases.append(dict(
            case=name, reference_second_moment=v.tolist(),
            residual_reconstruction=(m*m+compress(n)).tolist(),
            raw_reconstruction=compress(v).tolist(),
            relative_direction_error_residual=float(np.linalg.norm(residual_u-u)/np.linalg.norm(u)),
            relative_direction_error_raw=float(np.linalg.norm(raw_u-u)/np.linalg.norm(u)),
            relative_direction_error_best_scalar=float(np.linalg.norm(scalar*m-u)/np.linalg.norm(u))))
    assert cases[0]['relative_direction_error_residual'] < 1e-12
    assert cases[0]['relative_direction_error_raw'] > .1
    assert cases[1]['relative_direction_error_raw'] < 1e-12
    assert cases[1]['relative_direction_error_residual'] > .1
    checks['representation_examples_not_training'] = cases

    # Equal row/column marginals do not identify an interaction pattern.
    delta = np.array([[2., -2.], [-2., 2.]])
    plus, minus = 3 + delta, 3 - delta
    assert np.array_equal(compress(plus), compress(minus))
    checks['indistinguishable_residuals'] = {
        'plus': plus.tolist(), 'minus': minus.tolist(),
        'common_reconstruction': compress(plus).tolist(),
        'minimax_squared_frobenius_error_lower_bound': float(np.sum(delta*delta))}

    # Stationary calibration of the UNCOMPRESSED statistic, not of the nonlinear reconstruction.
    replicas = 16000
    beta, rho = .9, .97
    mu, sigma = np.array([-1., .4, 2.]), np.array([.3, 1., 2.])
    m = np.zeros((replicas, 3))
    n = np.zeros_like(m)
    for _ in range(700):
        # Non-Gaussian, skewed, zero-mean, unit-variance noise.
        noise = rng.exponential(size=m.shape) - 1
        g = mu + sigma * noise
        e = g - m
        m = beta*m + (1-beta)*g
        n = rho*n + (1-rho)*beta*e*e
    z = m*m+n
    target = mu*mu+sigma*sigma
    se = z.std(axis=0, ddof=1)/np.sqrt(replicas)
    score = (z.mean(axis=0)-target)/se
    assert np.max(np.abs(score)) < 6
    checks['stationary_uncompressed_calibration'] = {
        'independent_replicas': replicas, 'steps': 700, 'beta': beta, 'rho': rho,
        'target': target.tolist(), 'observed_mean': z.mean(axis=0).tolist(),
        'monte_carlo_standard_errors': se.tolist(), 'z_scores': score.tolist(),
        'scope': 'Fixed mean, iid skewed noise; no calibration claim for row-column ratios.'}

    # Current-sample normalization has conditional bias even at a stationary point.
    beta, rho = .9, .95
    support = np.array([2., -1.])
    probs = np.array([1/3, 2/3])
    m = (1-beta)*support
    n = (1-rho)*beta*support**2
    checks['current_sample_bias_example'] = {
        'gradient_mean': float(probs@support),
        'normalized_update_mean_from_zero_state': float(probs@direction(m, n)),
        'scope': 'A failed martingale cancellation, not by itself a nonconvergence proof.'}

    nr = nc = 4096
    checks['fp32_state_example'] = {
        'shape': [nr, nc], 'adam_two_moments_bytes': 8*nr*nc,
        'one_moment_plus_marginals_bytes': 4*(nr*nc+nr+nc),
        'marginal_bytes': 4*(nr+nc),
        'scope': 'Persistent numeric state only; excludes parameters, gradients, activations, scratch.'}
    out = Path(__file__).with_name('38_RESIDUAL_GEOMETRY_CHECKS.json')
    out.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
