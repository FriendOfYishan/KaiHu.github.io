"""Deterministic filter identities, not neural-network optimization benchmarks."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
def kernel(name,n):
    k=np.arange(1,n+1,dtype=float)
    lw={'uniform':np.zeros(n),'linear':np.log(k),'log':np.log(np.log1p(k)),
        'exp_fixed':.01*k,'exp_horizon':5*k/n,'sqrt_exp':np.sqrt(k)}[name]
    p=np.zeros(n);z=np.zeros(n);logS=-np.inf;A=B=0.
    err=0.;coeff=0.
    for i in range(n):
        logS=np.logaddexp(logS,lw[i]);a=np.exp(lw[i]-logS)
        p*=1-a;p[i]+=a
        z=(1-a)*z+a*p
        A=(1-a)*(A+1);B=(1-a)*(B+1)+a*A
        if i:
            coeff=A/(B-A);h=(1+coeff)*p-coeff*z
            err=max(err,abs(h.sum()-1),abs(np.dot(h,k)-(i+1)))
    return dict(method=name,age=A,n_eff=1/np.dot(p,p),
                double_correction_coefficient=coeff,
                corrected_noise_factor=float(np.dot(h,h)),
                raw_noise_factor=float(np.dot(p,p)),
                quadratic_bias=float(np.dot(h,(n-k)**2)),
                max_affine_error=err)
rows=[kernel(name,1000) for name in ['uniform','linear','log','exp_fixed','exp_horizon','sqrt_exp']]
assert max(r['max_affine_error'] for r in rows)<1e-8
assert abs(rows[1]['double_correction_coefficient']-1.5)<1e-10
errors=[]
for n in [1,2,3,10,100,1000]:
    k=np.arange(1,n+1,dtype=float)
    p=2*k/(n*(n+1));s=np.ones(n)/n;h=3*p-2*s
    errors.extend([abs(h.sum()-1),abs(h@k-n),
                   abs(h@h-(4*n-2)/(n*(n+1))),
                   abs(h@(n-k)**2+(n-1)*(n-2)/6)])
assert max(errors)<1e-8
# Check correlated-noise accounting for a partially corrected direction.
gate_errors=[]
for n in [3,10,100,1000]:
    k=np.arange(1,n+1,dtype=float);p=2*k/(n*(n+1));s=np.ones(n)/n
    V0=p@p;Dn=(n-1)/(3*n*(n+1))
    for gamma in [0.,.1,.5,1.]:
        h=p+2*gamma*(p-s)
        gate_errors.append(abs(h@h-(V0+4*gamma*(1+gamma)*Dn)))
    g=rng_g=np.random.default_rng(n).normal(size=(n,3))
    mm=p@g;ss=s@g;rr=s@(g*g)
    residual=rr-ss*ss-3*(n+1)/(n-1)*(mm-ss)**2
    X=np.stack([np.ones(n),k],axis=1);fit=X@np.linalg.lstsq(X,g,rcond=None)[0]
    gate_errors.append(float(np.max(np.abs(residual-np.mean((g-fit)**2,axis=0)))))
assert max(gate_errors)<1e-12
# Independent check of weighted-regression recurrences versus batch fit.
rng=np.random.default_rng(82);m=np.zeros(3);D=np.zeros(3);A=V=0.;lw=[];observations=[];err=0.
for t in range(1,301):
    lw.append(np.sqrt(t));logs=np.array(lw);weights=np.exp(logs-np.logaddexp.reduce(logs));a=weights[-1]
    g=rng.normal(size=3);observations.append(g)
    old_m=m.copy();old_A=A
    D=(1-a)*D+a*(1-a)*(old_A+1)*(g-old_m)
    V=(1-a)*V+a*(1-a)*(old_A+1)**2
    A=(1-a)*(old_A+1);m=(1-a)*m+a*g
    if t>1:
        X=np.stack([np.ones(t),np.arange(1,t+1)-t],axis=1)
        batch=np.linalg.solve(X.T@(weights[:,None]*X),X.T@(weights[:,None]*np.array(observations)))[0]
        err=max(err,float(np.max(np.abs(m+A*D/V-batch))))
assert err<1e-10
out=dict(kind='filter identities only; no GPU training',T=1000,
         weights={'log':'log(k+1)','exp_fixed':'exp(.01*k)','exp_horizon':'exp(5*k/T)','sqrt_exp':'exp(sqrt(k))'},
         filters=rows,two_average_max_error=max(errors),weighted_regression_max_error=err,
         shrinkage_and_residual_max_error=max(gate_errors))
(P/'verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
