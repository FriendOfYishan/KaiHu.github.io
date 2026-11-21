"""Candidate 42: local algebra checks and a NumPy reference; no neural training."""
import itertools
import json
from pathlib import Path
import numpy as np


def step(x, m, q, g, eta, t, beta=.9, eps=1e-8):
    """Rows are fixed groups; equal group sizes in this reference only.

    Persistent arrays: x (parameters), m (raw EMA), q (one scalar per row).
    Real kernels must divide a tail group's sum by its actual, not padded, size.
    """
    residual = g - m
    new_m = beta * m + (1 - beta) * g
    new_q = q + beta * np.mean(residual ** 2, axis=-1, keepdims=True)
    scale_q = new_q if t == 1 else q
    return x - eta * new_m / np.sqrt(scale_q + eps ** 2), new_m, new_q


def main():
    rng = np.random.default_rng(421115)
    beta, a, kap = .9, .1, 9.
    out = {'seed': 421115}
    # The cumulative full-energy / signal / residual decomposition, arbitrary signs.
    m = np.zeros((4, 13)); q = np.zeros((4, 1)); n = np.zeros_like(m)
    full = np.zeros_like(m); explained = np.zeros_like(m); x = np.zeros_like(m)
    energy_err = compressed_err = 0.
    for t in range(1, 1001):
        g = rng.normal(size=m.shape) + np.sin(t / 17)
        old = m.copy()
        explained += old ** 2
        full += g ** 2
        n += beta * (g - old) ** 2
        x, m, qnew = step(x, m, q, g, .001, t)
        assert np.all(qnew >= q)
        q = qnew
        energy_err = max(energy_err, float(np.max(np.abs(full - explained - m*m/a - n))))
        compressed_err = max(compressed_err, float(np.max(abs(q - n.mean(-1, keepdims=True)))))
    assert energy_err < 1e-9 and compressed_err < 1e-9
    out['arbitrary_gradient_path'] = dict(steps=1000, energy_identity_error=energy_err,
                                         compressed_residual_error=compressed_err)

    # Constant signal: residual budget saturates, full-gradient budget grows linearly.
    g = np.array([[1., -2., .1, .3]])
    m = np.zeros_like(g); q = np.zeros((1, 1)); x = np.zeros_like(g)
    err = 0.
    for t in range(1, 1001):
        x, m, q = step(x, m, q, g, .001, t)
        closed = beta * (1-beta**(2*t)) / (1-beta*beta) * np.mean(g*g)
        err = max(err, abs(float(q.item()) - closed))
    assert err < 1e-12
    out['constant_signal'] = dict(max_budget_formula_error=err,
        residual_budget_at_1000=float(q.item()), full_gradient_budget_at_1000=float(1000*np.mean(g*g)),
        scope='Prescribed gradient stream; not a bounded-below objective trajectory.')

    # First-step displacement bound, including zero gradients.
    first_violation = 0.
    for _ in range(500):
        g = rng.normal(size=(3, 19)) * 10**rng.uniform(-8, 6)
        g[0] = 0
        x, _, _ = step(np.zeros_like(g), np.zeros_like(g), np.zeros((3,1)), g, .01, 1)
        first_violation = max(first_violation, float(np.max(np.linalg.norm(x, axis=1) - .01*a*np.sqrt(19/beta))))
    assert first_violation < 1e-12
    out['first_step_bound_max_violation'] = first_violation

    # Exact conditional-noise inequalities at independently selected history states.
    noise = np.array(list(itertools.product([-1., 1.], repeat=2))) * .2
    max_noise_violation = max_log_violation = 0.
    for _ in range(1000):
        f = rng.uniform(-1, 1, 2); old = rng.uniform(-1, 1, 2)
        qold = rng.uniform(.1, 5)
        g = f + noise
        inc = beta * np.mean((g-old)**2, axis=1)
        nu = .08
        max_noise_violation = max(max_noise_violation, nu/qold - 2/beta*np.mean(inc/qold))
        G = 1.2*np.sqrt(2)
        K = 1 + 4*beta*G*G/(2*qold)
        max_log_violation = max(max_log_violation, float(np.max(inc/qold-K*np.log1p(inc/qold))))
    assert max_noise_violation < 1e-12 and max_log_violation < 1e-12
    out['noise_budget_checks'] = dict(cases=1000, conditional_variance_violation=max_noise_violation,
                                     logarithmic_payment_violation=max_log_violation)

    # Entire adaptive noise tree, conditional on a legal first bootstrap observation.
    eta, eps, T = .002, .1, 7
    xpre = np.array([[.6, -.8]])
    x, m, q = step(xpre, np.zeros_like(xpre), np.zeros((1,1)), np.sin(xpre), eta, 1, eps=eps)
    xinit, minit, qinit = x.copy(), m.copy(), q.copy()
    hprev = 1/np.sqrt(q+eps*eps)
    H = float(hprev.item()); R = H; G = 1.2*np.sqrt(2)
    zinit = x-eta*kap*hprev*m
    delta = float(np.sum(1-np.cos(zinit)))
    assert eta <= min(1/(8*H), 1/(2*np.sqrt(2)*H*kap))
    C0 = (2*delta/eta + 2*H**3*eta**2*kap**2*np.sum(minit**2)/a
          + 2*kap*G*G*R + 4*eta*kap**2*H*G*G*R)
    C1 = 2*(eta+H*eta**2*kap**2)
    sumD = sumV = 0.
    cumulative_f2 = np.zeros(1)
    virt_err = 0.
    for t in range(T):
        h = 1/np.sqrt(q+eps*eps)
        f = np.sin(x)
        sumD += float(np.mean(np.sum(h*f*f, axis=1)))
        sumV += float(np.mean(np.sum(np.broadcast_to(h*h, f.shape)*.04, axis=1)))
        cumulative_f2 += np.sum(f*f,axis=1)
        oldz = x-eta*kap*hprev*m
        w = (hprev-h)*m
        g = (f[:,None,:]+noise[None,:,:]).reshape(-1,2)
        oldm = np.repeat(m,4,axis=0)
        oldq = np.repeat(q,4,axis=0)
        oldx = np.repeat(x,4,axis=0)
        # Each branch is one group of dimension two; the grouped step is exact.
        x, m, q = step(oldx,oldm,oldq,g,eta,t+2,eps=eps)
        hn = np.repeat(h,4,axis=0)
        newz = x-eta*kap*hn*m
        expected = -eta*hn*g+eta*kap*np.repeat(w,4,axis=0)
        virt_err = max(virt_err,float(np.max(abs(newz-np.repeat(oldz,4,axis=0)-expected))))
        cumulative_f2 = np.repeat(cumulative_f2,4)
        hprev = hn
    Q = q[:,0]+eps*eps
    lhs = float(np.mean(cumulative_f2/np.sqrt(Q)))
    K = 1+4*beta*G*G/(2*float(qinit.item()+eps*eps))
    noise_bound = 2*K/beta*float(np.mean(np.log(Q/(qinit.item()+eps*eps))))
    bound = float(C0+C1*noise_bound)
    assert virt_err < 1e-12 and lhs <= sumD+1e-10
    assert sumV <= noise_bound+1e-10 and sumD <= C0+C1*sumV+1e-10
    out['exact_nonconvex_tree'] = dict(objective='sum(1-cos x)', steps_after_bootstrap=T,
        outcomes=len(x), virtual_identity_error=virt_err, normalized_stationarity_lhs=lhs,
        theorem_rhs=bound, actual_weighted_signal_sum=sumD, actual_filtered_noise_sum=sumV,
        noise_log_upper_bound=noise_bound, scope='Bound is conservative; no empirical rate claim.')

    # Persistent noise is deliberately not classified as a fast-rate regime.
    m=np.zeros((1,64));q=np.zeros((1,1));x=np.zeros_like(m)
    for t in range(1,16385):
        g=rng.choice([-1.,1.],size=m.shape)
        x,m,q=step(x,m,q,g,.001,t)
    observed=float(q.item()/16384);expected=2*beta/(1+beta)
    assert abs(observed-expected) < .02
    out['persistent_noise_negative_control'] = dict(budget_per_step=observed,
        stationary_prediction=expected, conclusion='Linear residual growth; p<1 is not automatic.')
    out['scope']='Local mathematical checks only; no GPU timing, neural quality, or generalization result.'
    dest=Path(__file__).with_name('42_RESIDUAL_CLOCK_CHECKS.json')
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
