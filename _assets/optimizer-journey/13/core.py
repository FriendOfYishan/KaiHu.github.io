from pathlib import Path
import json
import sympy as sp
import numpy as np

B,C,delta=sp.symbols('B C delta',positive=True)
Q=B+C; V=B+delta*C; W=B+delta**2*C
assert sp.simplify(W/V-(delta+(1-delta)*B/V))==0
assert sp.simplify(W*Q/V**2-1-(1-delta)**2*B*C/V**2)==0
R,b,t=sp.symbols('R b t',positive=True)
ratio=(1+R*(1-R)/(1+R)**2)/(1+b*(1-R))
assert sp.simplify(ratio-1-(1-R)*(R/(1+R)**2-b)/(1+b*(1-R)))==0
# Exact Beta(1/2,(t-1)/2) moments in the Gaussian model.
er1=1/t;er2=3/(t*(t+2));er3=15/(t*(t+2)*(t+4))
assert sp.simplify((er2-er3)-3/(t+4)*(er1-er2))==0
assert sp.simplify((3/(t+1))/(1+3/(t+1))-3/(t+4))==0

# Verify frozen-scale parameter difference is exactly inverse-time SGD.
rng=np.random.default_rng(1203)
x=3.; ref=3.; s=0.; p=0.; eta=1.3; scale=.7; err=0.
for step in range(1,1001):
    noise=rng.normal();g=x+noise
    s+=g;newp=scale*s
    x-=eta/step*(newp-p);p=newp
    ref-=eta*scale/step*(ref+noise)
    err=max(err,abs(x-ref))
assert err<1e-11
result=dict(symbolic_compensation=True,gaussian_finite_time_bound_algebra=True,
    constant_signal_leverage=1,frozen_scale_equivalence_error=err,
    note='Finite Gaussian noise theorem applies to U and unweighted potential differences; not automatically to the time-weighted parameter recursion.')
(Path(__file__).resolve().parent/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
