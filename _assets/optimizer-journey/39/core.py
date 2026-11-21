"""Algebra, exact polynomial certificates, and constructed examples. No model training."""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import json
import numpy as np


def add(a, b):
    out = [F(0)] * max(len(a), len(b))
    for i, v in enumerate(a): out[i] += v
    for i, v in enumerate(b): out[i] += v
    return out


def mul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b): out[i+j] += x*y
    return out


def bernstein_on_interval(poly, left, right):
    degree = len(poly)-1
    power = [sum(poly[j]*comb(j, i)*left**(j-i)*(right-left)**i
                 for j in range(i, degree+1)) for i in range(degree+1)]
    return [sum(power[i]*F(comb(k, i), comb(degree, i))
                for i in range(k+1)) for k in range(degree+1)]


def exact_positive_certificate(poly, lo, hi, pieces=128):
    smallest = None
    for j in range(pieces):
        a, b = lo+(hi-lo)*F(j, pieces), lo+(hi-lo)*F(j+1, pieces)
        coeff = bernstein_on_interval(poly, a, b)
        assert min(coeff) > 0, (j, min(coeff))
        smallest = min(coeff) if smallest is None else min(smallest, min(coeff))
    return {'pieces': pieces, 'degree': len(poly)-1,
            'minimum_Bernstein_coefficient_exact': str(smallest),
            'minimum_Bernstein_coefficient_float': float(smallest)}


def decode(m, q, eps=1e-3):
    z = np.clip((5/12)*m/np.sqrt(q+eps*eps), -1, 1)
    return z*(2-np.abs(z))


def regret(m, v, u):
    d = np.sqrt(v)
    oracle = m/d
    return .5*np.sum(d*(u-oracle)**2)


def main():
    out = {}
    # q(z)=2z-z², phi(z)=(12z)/sqrt(25+144z²).
    # Exact rational positivity proves a uniform absolute error <= 1/12.
    square_root_denominator = [F(25), F(0), F(144)]
    phi_numerator_squared = [F(0), F(0), F(144)]
    p_lower = add(mul(mul([F(1,12), F(2), F(-1)],
                          [F(1,12), F(2), F(-1)]), square_root_denominator),
                  [-v for v in phi_numerator_squared])
    p_upper = add(phi_numerator_squared,
                  [-v for v in mul(mul([F(-1,12), F(2), F(-1)],
                                      [F(-1,12), F(2), F(-1)]), square_root_denominator)])
    out['exact_uniform_decoder_error_certificate'] = {
        'bound': '1/12', 'kappa': '12/5',
        'lower_error_polynomial_ascending_coefficients': [str(x) for x in p_lower],
        'upper_error_polynomial_ascending_coefficients': [str(x) for x in p_upper],
        'lower_error_on_0_1': exact_positive_certificate(p_lower, F(0), F(1)),
        'upper_error_on_1over12_1': exact_positive_certificate(p_upper, F(1,12), F(1)),
        'other_intervals': '0<=z<=1/12: phi>=2z>=q; z>=1: 1-phi<=1/13<1/12.'}
    z = np.linspace(0, 1, 200001)
    e = z*(2-z)-12*z/np.sqrt(25+144*z*z)
    out['decoder_numerical_extrema'] = {'minimum': float(e.min()), 'maximum': float(e.max()),
                                       'z_at_min': float(z[e.argmin()]), 'z_at_max': float(z[e.argmax()])}

    rng = np.random.default_rng(20261115)
    max_identity_error = 0.
    checked = strict = 0
    for _ in range(2000):
        m = rng.normal(size=(4, 5))
        n = rng.uniform(.5, 2.) + rng.uniform(-1e-5, 1e-5, size=m.shape)
        s = m*m
        v = s+n+.01
        vr = s+n.mean()+.01
        vf = np.full_like(v, (s+n).mean()+.01)
        ur, uf = m/np.sqrt(vr), m/np.sqrt(vf)
        er, ef = regret(m, v, ur), regret(m, v, uf)
        a = np.linalg.norm(m*(s-s.mean()))
        b = np.linalg.norm(m*(n-n.mean()))
        lo = min(v.min(), vr.min(), vf.min())
        hi = max(v.max(), vr.max(), vf.max())
        cr = hi**.25/(2*lo**1.5)
        cf = lo**.25/(2*hi**1.5)
        assert np.sqrt(2*er) <= cr*b+1e-12
        assert np.sqrt(2*ef)+1e-12 >= cf*max(0, a-b)
        if a > (1+(hi/lo)**1.75)*b:
            checked += 1
            assert er < ef
            strict += 1
        # Exact equality of regret and the difference in the quadratic proxy.
        oracle = m/np.sqrt(v)
        model = lambda u: -np.sum(m*u)+.5*np.sum(np.sqrt(v)*u*u)
        max_identity_error = max(max_identity_error, abs((model(ur)-model(oracle))-er))
    out['structured_residual_theorem_checks'] = {
        'draws': 2000, 'sufficient_condition_satisfied': checked,
        'strictly_smaller_proxy_regret': strict,
        'proxy_identity_max_absolute_error': max_identity_error}

    cases = []
    for name, m, n in [
        ('homogeneous_residual_rich_signal', np.array([[1.,4.],[4.,1.]]), np.ones((2,2))),
        ('failure_when_total_is_homogeneous', np.sqrt(np.array([[17.,2.],[2.,17.]])),
         np.array([[1.,16.],[16.,1.]]))]:
        v = m*m+n+1e-6
        reference = m/np.sqrt(v)
        full = m/np.sqrt(v.mean())
        residual = m/np.sqrt(m*m+n.mean()+1e-6)
        candidate = decode(m, n.mean())
        errors = {key: float(np.linalg.norm(u-reference)/np.linalg.norm(reference))
                  for key, u in [('full_compression',full),('residual_exact_decode',residual),
                                 ('candidate_quadratic_decode',candidate)]}
        regrets = {key: float(regret(m,v,u)) for key,u in [('full_compression',full),
                   ('residual_exact_decode',residual),('candidate_quadratic_decode',candidate)]}
        aa, bb, cc = np.sum(m*m), np.sum(np.sqrt(v)*m*m), np.sum(m*m/np.sqrt(v))
        hbest = aa/bb
        scalar_lower_bound = .5*(cc-aa*aa/bb)
        assert abs(regret(m,v,hbest*m)-scalar_lower_bound) < 1e-12
        decode_bound = np.sum(np.sqrt(v))/(2*144)
        # This explicitly specified quadratic objective realizes the proxy regret as actual loss.
        loss = lambda u: -np.sum(m*u)+.5*np.sum(np.sqrt(v)*u*u)
        cases.append({'name':name, 'relative_direction_errors':errors, 'proxy_regrets':regrets,
                      'best_scalar_proxy_regret':float(scalar_lower_bound),
                      'decoder_regret_bound_when_residual_exact':float(decode_bound),
                      'strictly_beats_every_scalar_certified_when_residual_exact':bool(
                          np.all(n == n.mean()) and decode_bound < scalar_lower_bound),
                      'candidate_minus_full_loss_on_explicit_quadratic':float(loss(candidate)-loss(full)),
                      'scope':'Constructed local state; not a trained Transformer result.'})
    out['success_and_failure_examples'] = cases

    # One persistent h encodes a scalar mean residual. Its use is delayed by one step.
    beta, eps = .9, .001
    m = np.zeros((11,13))
    n = np.full_like(m, 1-eps*eps)
    v = n.copy()
    h = 1.
    max_h_error = max_v_error = max_step = 0.
    for t in range(500):
        g = rng.normal(size=m.shape)+np.sin(t/19)
        innovation = g-m
        m = beta*m+(1-beta)*g
        zz = np.clip((5/12)*h*m,-1,1)
        u = zz*(2-np.abs(zz))
        max_step = max(max_step,float(np.max(np.abs(u))))
        h = h/np.sqrt(beta+(1-beta)*h*h*(beta*np.mean(innovation*innovation)+eps*eps))
        n = beta*n+beta*(1-beta)*innovation*innovation
        v = beta*v+(1-beta)*g*g
        max_h_error = max(max_h_error,abs(1/(h*h)-(n.mean()+eps*eps)))
        max_v_error = max(max_v_error,float(np.max(np.abs(v-m*m-n))))
    assert max_h_error < 1e-11 and max_v_error < 1e-11 and max_step <= 1
    out['one_state_one_scalar_streaming_checks'] = {
        'steps':500,'h_scalar_identity_max_error':max_h_error,
        'moment_identity_max_error':max_v_error,'maximum_abs_update':max_step}

    path = Path(__file__).with_name('39_RESIDUAL_RECONSTRUCTION_CHECKS.json')
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
