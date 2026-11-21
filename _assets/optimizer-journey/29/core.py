"""Check the proposed risk surrogate and invariant, not training performance."""
import json
import random
from pathlib import Path

rng=random.Random(291111)
risk_checks=0;state_checks=0
for _ in range(20000):
    a=10**rng.uniform(-5,-.01)
    floor=rng.uniform(1e-5,1)
    z=rng.random()
    base=max(floor,a);new=max(floor,a*(2+z)/3)
    risk=lambda w:w+a**3*z/(2*w*w)
    assert 2*base/3-1e-14<=new<=base+1e-14
    assert risk(new)<=risk(base)+1e-12
    risk_checks+=1

for seed in range(10):
    m=v=0.;eps=.01
    for t in range(1,1001):
        z=m*m/(v+eps*eps)
        assert 0<=z<=1+1e-12
        a=max(2/(t+1),.1/t**(1/3)*(2+z)/3)
        g=rng.gauss(0,1)+.5
        m=(1-a)*m+a*g;v=(1-a)*v+a*g*g
        assert m*m<=v+1e-12
        assert abs(m/(v**.5+eps))<=1+1e-12
        state_checks+=1

out=dict(status='passed',risk_surrogate_checks=risk_checks,adaptive_state_checks=state_checks,
         scope='Surrogate dominance, coordinate-weight range, and same-weight moment invariant. No optimization-advantage claim.')
Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
