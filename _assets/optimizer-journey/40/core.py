"""Local algebra checks for theory 40; no external data or training runs."""
import itertools
import json
from pathlib import Path

import numpy as np


def gate(m, q, beta=.9, floor=.1, eps2=1e-16):
    c = (1-beta)/(2*beta)
    return np.clip((m*m-c*q)/(m*m+q+eps2), floor, 1.)


def main():
    rng = np.random.default_rng(401114)
    out = {}
    beta, rho, floor, eps2 = .9, .999, .1, 1e-16
    a, kappa = 1-beta, beta/(1-beta)

    # The deployed state is m, group q, and group v; full n is verifier-only.
    m = np.zeros((4, 13))
    n = np.zeros_like(m)
    q = np.zeros((4, 1))
    v = np.zeros_like(q)
    worst = 0.
    for t in range(1000):
        g = rng.normal(size=m.shape) + np.sin(t/31)*np.arange(4)[:, None]
        layer = np.ones_like(q) if t == 0 else np.clip(
            1/np.sqrt(v/(1-rho**t)+eps2),1e-3,1e4)
        h = layer*gate(m, q)
        y = h*g
        e = y-m
        n = rho*n+(1-rho)*beta*e*e
        q = rho*q+(1-rho)*beta*np.mean(e*e, axis=1, keepdims=True)
        v = rho*v+(1-rho)*np.mean(g*g,axis=1,keepdims=True)
        m = beta*m+a*y
        worst = max(worst, float(np.max(np.abs(q-n.mean(axis=1, keepdims=True)))))
    out['compressed_residual_identity_max_error'] = worst
    assert worst < 1e-12

    max_violation = -np.inf
    c = a/(2*beta)
    for _ in range(2000):
        m = rng.normal(size=24)
        n = np.exp(rng.normal(size=24))
        q = np.full(24, np.mean(n))
        actual = np.abs(gate(m, q)-gate(m, n))
        bound = ((1+c)*m*m+c*eps2)*np.abs(q-n)/(m*m+np.minimum(q,n)+eps2)**2
        max_violation = max(max_violation, float(np.max(actual-bound)))
    out['gate_compression_bound_max_violation'] = max_violation
    assert max_violation <= 1e-12

    # Exact finite-time means/variances in a fixed-input stationary model.
    mu = np.array([.4, -1.2, .1])
    nu = np.array([.5, .1, 2.])
    gain = np.array([.2, .7, .9])
    mean = np.zeros(3)
    var = np.zeros(3)
    en = np.zeros(3)
    for _ in range(40000):
        en = rho*en+(1-rho)*beta*((gain*mu-mean)**2+gain**2*nu+var)
        mean = beta*mean+a*gain*mu
        var = beta**2*var+a*a*gain**2*nu
    em2 = mean**2+var
    expected_signal = gain**2*mu**2
    expected_total = gain**2*(mu**2+nu)
    stationarity_error = float(max(np.max(abs(em2-c*en-expected_signal)),
                                   np.max(abs(em2+en-expected_total))))
    ratio_error = float(np.max(abs((em2-c*en)/(em2+en)-mu**2/(mu**2+nu))))
    out['fixed_gain_calibration_max_error'] = stationarity_error
    out['ratio_of_expectations_invariance_error'] = ratio_error
    assert max(stationarity_error, ratio_error) < 1e-11

    # Cauchy optimality is a descent-certificate statement, not a loss ordering.
    cauchy_violation, regret_error = -np.inf, 0.
    for _ in range(2000):
        f = rng.normal(size=30)
        nu = np.exp(rng.normal(size=30))
        w = f*f+nu
        p = f*f/w
        b = np.exp(rng.normal(size=30))
        score = np.dot(b,f*f)**2/np.dot(b*b,w)
        optimum = np.sum(f**4/w)
        cauchy_violation = max(cauchy_violation,float(score-optimum))
        h = np.clip(p+rng.normal(0,.1,30),floor,1)
        risk = lambda v: np.dot(w,v*v)-2*np.dot(f*f,v)
        regret_error = max(regret_error,float(abs(risk(h)-risk(p)-np.dot(w,(h-p)**2))))
    out['certificate_optimality_max_violation'] = cauchy_violation
    out['certificate_regret_identity_max_error'] = regret_error
    assert cauchy_violation <= 1e-10 and regret_error < 1e-10

    d, r = 100, 7
    f2 = np.zeros(d)
    f2[:r] = 1.
    total = np.ones(d)
    ideal = f2/total
    baseline = 1/np.sqrt(total)
    score = lambda h: float(np.dot(h,f2)**2/np.dot(h*h,total))
    ideal_factor = score(ideal)/score(baseline)
    floored_factor = score(np.maximum(ideal,floor))/score(baseline)
    out['frozen_metric_certificate_example'] = dict(d=d,r=r,
        oracle_factor=ideal_factor,floored_factor=floored_factor)
    assert abs(ideal_factor-d/r)<1e-12
    assert abs(floored_factor-d/(r+floor*floor*(d-r)))<1e-12

    # Exact conditional enumeration on a genuinely nonconvex cosine objective.
    signs = np.array(list(itertools.product([-1.,1.],repeat=4)))
    F = lambda x: np.sum(1-np.cos(x),axis=-1)
    one_step_violation, virtual_error = -np.inf, 0.
    eta = .01
    for _ in range(1200):
        x, old = rng.normal(size=(2,4))
        q = np.exp(rng.normal())
        layer = rng.uniform(.25,3.,4)
        H = float(np.max(layer))
        h = layer*gate(old,q)
        f = np.sin(x)
        scale = rng.uniform(.05,.8,4)
        g = f+signs*scale
        y = h*g
        new = beta*old+a*y
        xn = x-eta*new
        z = x-eta*kappa*old
        zn = xn-eta*kappa*new
        virtual_error = max(virtual_error,float(np.max(abs(zn-(z-eta*y)))))
        D = np.dot(f*f,h)
        V = np.dot(h*h,scale*scale)
        lhs = float(np.mean(F(zn))-F(z))
        rhs = -eta*(.75-H*eta/2)*D+H*eta**3*kappa**2*np.dot(old,old)+eta**2*V/2
        one_step_violation = max(one_step_violation,lhs-rhs)
    out['nonconvex_conditional_states'] = 1200
    out['noise_outcomes_per_state'] = len(signs)
    out['nonconvex_one_step_bound_max_violation'] = one_step_violation
    out['virtual_iterate_identity_max_error'] = virtual_error
    assert one_step_violation <= 1e-10 and virtual_error < 1e-12

    adam_transport_error = 0.
    for _ in range(1000):
        raw_old = rng.normal(size=19)
        g = rng.normal(size=19)
        h_old = np.exp(rng.normal(size=19))
        # New scale is deliberately dependent on the current gradient.
        h_new = 1/np.sqrt(.999/h_old**2+.001*g*g+1e-12)
        old_velocity = h_old*raw_old
        new_velocity = h_new*(beta*raw_old+a*g)
        effective_input = h_new*g+kappa*(h_new-h_old)*raw_old
        adam_transport_error = max(adam_transport_error,
            float(np.max(abs(new_velocity-beta*old_velocity-a*effective_input))))
    out['actual_adam_transport_identity_max_error'] = adam_transport_error
    assert adam_transport_error < 1e-11

    # The temporal summation eliminates a separate uniform gradient bound.
    old = np.zeros(17)
    input_energy, old_state_energy = 0., 0.
    for _ in range(1000):
        y = rng.normal(size=17)
        old_state_energy += np.dot(old,old)
        input_energy += np.dot(y,y)
        old = beta*old+a*y
    out['ema_summed_energy_ratio'] = float(old_state_energy/input_energy)
    assert old_state_energy <= input_energy+1e-10

    # Complete tree: 4^8 outcomes, with the actual adaptive state and q.
    T = 8
    eta = .005
    x0 = np.array([.6,-.8])
    x = x0[None,:].copy()
    m = np.zeros_like(x)
    q = np.zeros((1,1))
    v = np.zeros_like(q)
    noises = .3*np.array(list(itertools.product([-1.,1.],repeat=2)))
    sumD = sumV = sumF2 = 0.
    H_tree = 1.
    for t in range(T):
        layer = np.ones_like(q) if t == 0 else np.clip(
            1/np.sqrt(v/(1-rho**t)+eps2),1e-3,1e4)
        H_tree = max(H_tree,float(np.max(layer)))
        h = layer*gate(m,q)
        f = np.sin(x)
        sumD += float(np.mean(np.sum(h*f*f,axis=1)))
        sumV += float(np.mean(np.sum(h*h*.09,axis=1)))
        sumF2 += float(np.mean(np.sum(f*f,axis=1)))
        g = (f[:,None,:]+noises[None,:,:]).reshape(-1,2)
        old = np.repeat(m,4,axis=0)
        y = np.repeat(h,4,axis=0)*g
        e = y-old
        q = rho*np.repeat(q,4,axis=0)+(1-rho)*beta*np.mean(e*e,axis=1,keepdims=True)
        v = rho*np.repeat(v,4,axis=0)+(1-rho)*np.mean(g*g,axis=1,keepdims=True)
        m = beta*old+a*y
        x = np.repeat(x,4,axis=0)-eta*m
    alpha = sumD/sumF2
    bound = 2*float(F(x0))/(alpha*eta*T)+(eta+2*H_tree*eta*eta*kappa*kappa)*(sumV/T)/alpha
    out['adaptive_exact_tree'] = dict(horizon=T,outcomes=len(x),alpha=alpha,
                                    average_gradient_squared=sumF2/T,bound=bound,
                                    average_filtered_variance=sumV/T,
                                    max_layer_scale_over_entire_tree=H_tree,eta=eta,
                                    states='M, q, v; full final candidate')
    assert eta <= min(1/(4*H_tree),1/(2*np.sqrt(2)*H_tree*kappa))
    assert sumF2/T <= bound+1e-10

    # Actual Adam-type one-step directions at a common prescribed state.
    # Any common bias-correction/epsilon scalar is absorbed in effective eta.
    # This is a one-state comparison, explicitly not two full training paths.
    d, etaA = 32, .1
    old = np.zeros(d)
    old[0] = 1.
    h = gate(old,1.,eps2=0.)
    effective_signal = beta+a*h[0]
    etaC = etaA/effective_signal
    base_loss = 1-np.sin(etaA)+.5*(d-1)*(etaA*a)**2
    cand_loss = 1-np.sin(etaA)+.5*(d-1)*(etaC*a*floor)**2
    exact_gap = .5*(d-1)*(etaA*a)**2*((floor/effective_signal)**2-1)
    out['actual_nonconvex_adam_one_step'] = dict(
        objective='1-cos(x[0]) + sum(x[1:]**2)/2',dimension=d,
        initial_x0='pi/2',old_m0=1.,old_other_m=0.,old_v_all=1.,old_q=1.,
        active_gate=float(h[0]),matched_signal=float(effective_signal),
        adam_effective_lr=etaA,candidate_lr=float(etaC),
        adam_expected_loss=float(base_loss),candidate_expected_loss=float(cand_loss),
        exact_loss_difference=float(exact_gap),noise_cost_ratio=float((floor/effective_signal)**2))
    assert cand_loss < base_loss and abs(cand_loss-base_loss-exact_gap)<1e-12

    # Equal-geometry structural example for the leading noise/retention factor.
    tau, delta = .1, .02
    out['illustrative_oracle_error_comparison'] = dict(
        floor=tau,uniform_gain_error=delta,
        effective_complexity_factor_upper=(tau*tau+2*delta)/(1-delta)**2,
        interpretation='conditional bound constant versus identity metric; not observed training gain')
    out['scope'] = 'Algebra and finite nonconvex examples only; no new neural training, speed, or novelty claim.'
    target = Path(__file__).with_name('40_RESIDUAL_NOISE_ALLOCATION_CHECKS.json')
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
