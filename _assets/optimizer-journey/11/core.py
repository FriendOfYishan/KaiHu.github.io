"""Exact scalar quadratic risks; no Monte Carlo ranking or NN claims."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
T=4096
kappa=1.; sigma=1.; initial=3.
rows=[]
curves={}
for q in [.05,.1,.2,.5,1.]:
    rho=1-q
    for method in ['constant','numerator','transport','PAGE-1','PAGE-2']:
        bias=initial; noise=0.; cross=0.; curve=[]; ev=0.; cost=0.
        for t in range(1,T+1):
            if method=='constant':
                bias*=rho
                noise=rho*rho*noise+(q/kappa)**2*sigma*sigma
            elif method=='numerator':
                c=1/q
                bias=initial*c/(t+c)
                noise=sigma*sigma*t/(kappa*kappa*(t+c)**2)
            elif method=='transport':
                # Cov(e_{t-1}, mean noise_t)=(t-1)/t Cov(e_{t-1},mean noise_{t-1})
                prior_cross=(t-1)/t*cross
                noise=rho*rho*noise+(q/kappa)**2*sigma*sigma/t-2*rho*q/kappa*prior_cross
                cross=rho*prior_cross-q/kappa*sigma*sigma/t
                bias*=rho
            else:
                refresh=min(1.,(1 if method=='PAGE-1' else 2)/t)
                ev=(1-refresh)*ev+refresh*sigma*sigma/t
                prior_cross=(1-refresh)*cross
                noise=rho*rho*noise+(q/kappa)**2*ev-2*rho*q/kappa*prior_cross
                cross=rho*prior_cross-q/kappa*ev
                bias*=rho
                assert ev <= (sum(1/j for j in range(1,t+1))/t if method=='PAGE-1' else 2/t)+1e-12
            cost+=refresh*t+2*(1-refresh) if method.startswith('PAGE') else 1
            curve.append(dict(t=t,mse=bias*bias+noise,bias_squared=bias*bias,noise_variance=noise,expected_gradient_calls=cost))
        rows.append(dict(method=method,q=q,mse_32=curve[31]['mse'],mse_128=curve[127]['mse'],
                         mse_4096=curve[-1]['mse'],scaled_tail=T*curve[-1]['mse'],
                         first_below_01=next((r['t'] for r in curve if r['mse']<=.1),None),
                         mse_at_expected_budget_4096=next(r['mse'] for r in reversed(curve) if r['expected_gradient_calls']<=4096)))
        curves[f'{method}_{q}']=curve
# Direct random-path check of x recursion against transported moment.
rng=np.random.default_rng(1109)
for q in [.1,.5,1.]:
    x=3.; oldx=3.; direct=3.; olddirect=3.; m=0.
    for t in range(1,1025):
        xi=rng.normal()
        g=kappa*x+xi
        m=(1-1/t)*(m+kappa*(x-oldx))+g/t
        nxt=x-q/kappa*m
        direct_next=direct+(1-1/t)*(1-q)*(direct-olddirect)-q/(kappa*t)*(kappa*direct+xi)
        assert abs(nxt-direct_next)<1e-11
        oldx,x=x,nxt
        olddirect,direct=direct,direct_next
# Verify the scalar numerator exact risk and uniform 1/T theorem.
for q in [.05,.1,.2,.5,1.]:
    c=1/q
    for r in curves[f'numerator_{q}']:
        assert r['mse']<=(c*initial**2/4+sigma**2/kappa**2)/r['t']+1e-12
result=dict(config=dict(T=T,kappa=kappa,sigma=sigma,initial=initial,
    definition='F=kappa*x^2/2, g=kappa*x+xi, iid mean-zero noise variance sigma^2',
    gradient_calls_per_step='1 except PAGE: p_t*b_t+2*(1-p_t)',known_curvature=True,frozen_denominator=True,
    selection='all 5 q values; exact expectation recurrences, no stochastic tuning'),
    rows=rows,curves=curves,recursion_checks_passed=True)
(HERE/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for r in rows:print(r)
