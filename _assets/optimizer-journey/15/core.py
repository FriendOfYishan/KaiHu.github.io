import json
from pathlib import Path
import numpy as np
import torch
from preview import Preview
P=Path(__file__).resolve().parent
rng=np.random.default_rng(23);err=0.
for name in ['A-plain','A-mix','A-mix-lever','A-mix-diff','A-mix-rawdiff','H-mix','EMA-mix','Preview-only']:
 p=torch.nn.Parameter(torch.tensor([.7,-.4],dtype=torch.float64));o=Preview([p],lr=.1,name=name)
 x=p.detach().numpy().copy();m=np.zeros(2);v=m.copy();up=m.copy()
 for t in range(1,101):
  q=rng.normal(size=2);a=1024/(t+1023) if name=='H-mix' else 2/(t+1)
  if name in ['EMA-mix','Preview-only']:
   m=.9*m+.1*q;v=.999*v+.001*q*q;mm=m/(1-.9**t);vv=v/(1-.999**t)
  else:m=(1-a)*m+a*q;v=(1-a)*v+a*q*q;mm=m;vv=v
  assert np.min(vv-mm*mm)>-1e-12 if name not in ['EMA-mix','Preview-only'] else True
  u=mm/(np.sqrt(vv)+1e-8)
  if name in ['A-mix-lever','A-mix-diff','A-mix-rawdiff']:u*=.5+.5*mm*mm/(vv+1e-16)
  d=.1*(u-(1-1/t)*up) if name=='A-mix-diff' else (.1/t*(u-up) if name=='A-mix-rawdiff' else .1*u)
  x-=d;up=u.copy();p.grad=torch.tensor(q);o.step();err=max(err,np.max(np.abs(x-p.detach().numpy())))
assert err<1e-10
# Same-sample quadratic blend retains exactly the original noise.
xi=rng.normal(size=10000);g=2*.7+xi;h=2*(.7-.2)+xi
assert np.max(np.abs(.5*(g+h)-(2*.7-.2)-xi))<1e-12
result=dict(implementation_max_error=float(err),T10_v2=-40/33,
 same_sample_noise_variance_ratio=float(np.var(.5*(g+h))/np.var(xi)))
(P/'verification.json').write_text(json.dumps(result,indent=2));print(result)
