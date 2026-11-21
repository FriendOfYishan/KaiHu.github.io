"""CPU mathematical checks only; no training, GPU access, or remote operations."""
import json
from pathlib import Path
import numpy as np

rng=np.random.default_rng(12030)
max_recurrence_error=0.;checks=0
for _ in range(100):
    history=[];weights=np.empty(0);Z=0.;U=0.;Q=0.;H=1.
    for t in range(1,101):
        alpha=max(2/(t+1),.1*t**(-1/3));beta=1-alpha;s=float(np.exp(rng.normal()))
        oldZ=Z;Z=beta*Z+1;U=beta*U+s;Q=beta*Q+1/s
        H=Z/(beta*oldZ/H+1/s)
        history.append(s);weights=np.r_[weights*beta,1.]
        direct=weights.sum()/np.dot(weights,1/np.asarray(history))
        max_recurrence_error=max(max_recurrence_error,abs(H-direct))
        assert np.isclose(H,direct,rtol=1e-12)
        assert np.isclose(H,Z/Q,rtol=1e-12)
        checks+=1
bound_checks=0
for rho in [.01,.1,.5,.8]:
    for _ in range(250):
        p=rng.dirichlet(np.ones(30));e=rng.uniform(-1,1,30);e-=p@e;e*=rho/np.max(np.abs(e))
        s=1+e;A=p@s;H=1/(p@(1/s));v=p@((s-A)**2);q=v/A**2
        R=p@(s*s);hat=2*A*A-A*H
        bound=v*(rho+q)/(1-rho)
        assert abs(hat-R)<=bound+1e-12
        ratio=A*(A-H)/v
        assert 1/(1+rho+q)-1e-10<=ratio<=1/(1-rho+q)+1e-10
        bound_checks+=1
for _ in range(500):
    a,b=np.exp(rng.normal(size=2));A=(a+b)/2;H=2*a*b/(a+b)
    assert np.isclose(2*A*A-A*H,(a*a+b*b)/2,rtol=1e-12)
for _ in range(500):
    g=rng.uniform(-1,1,30);p=rng.dirichlet(np.ones(30));m=p@g;c=1.1
    s=g+c;A=p@s;H=1/(p@(1/s));hat=m*m+A*(A-H)
    assert np.isclose(hat,2*A*A-A*H-2*c*A+c*c,atol=1e-12)
    assert hat>=m*m-1e-12
    assert abs(m)/(np.sqrt(max(hat,0))+1e-8)<=1+1e-12
    delta=.01;s=np.abs(g)+delta;A=p@s;H=1/(p@(1/s));hat=(A-delta)**2+A*(A-H)
    assert hat>=m*m-1e-12
for alpha_rule,z_rule in [(lambda t:2/(t+1),lambda t:(t+2)/3),(lambda t:1/t,lambda t:(t+1)/2)]:
    z=0.
    for t in range(1,101):
        z=(1-alpha_rule(t))*z+1;assert np.isclose(z,z_rule(t))
p=np.array([.99,.01]);s=np.array([1.,100.]);A=p@s;H=1/(p@(1/s));R=p@(s*s);hat=2*A*A-A*H
g32=np.array([-1e-4,1e-4],dtype=np.float32);c32=np.float32(1)
m32=np.mean(g32);a32=c32+m32;h32=np.float32(1)/np.mean(np.float32(1)/(c32+g32))
v32=m32*m32+a32*(a32-h32)
report=dict(recurrence_comparisons=checks,max_abs_recurrence_error=max_recurrence_error,
    error_bound_cases=bound_checks,two_point_identity_cases=500,signed_restoration_cases=500,
    tail_counterexample=dict(A=A,H=H,true_R=R,estimated_R=hat,true_step=A/np.sqrt(R),estimated_step=A/np.sqrt(hat)),
    float32_cancellation=dict(A=float(a32),H=float(h32),estimated_v=float(v32),true_v=float(np.mean(g32*g32))),
    same_A_H_distinct_R=dict(A=2,H=18/11,R_three_point=14/3,R_two_point=52/11))
path=Path(__file__).with_name('30_ARITHMETIC_HARMONIC_VERIFICATION.json')
path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
