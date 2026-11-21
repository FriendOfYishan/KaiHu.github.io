"""Numerical checks of newly derived identities; not training evidence."""
import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(371114)
a=.1; beta=1-a; eta=.01; theta=.05; zeta=.5; c=2*a*theta/beta;q=eta*beta/(2*a);L=1.1
def f(x):return np.sum(1-np.cos(x)+.05*x*x)
def refresh(h,m,b,trial,C,j):
 trial=np.clip(trial,np.maximum(.05,h-C/j),np.minimum(5,h+C/j))
 d=np.maximum(trial-h,0);B=h*(c*m*m+zeta*(m-b)**2);A=d*m*m
 lam=np.ones_like(A);ix=A>B;lam[ix]=B[ix]/A[ix]
 return np.minimum(trial,h)+lam*d, np.count_nonzero(ix)
mx=dict(identity=0.,budget_violation=0.,path_descent_violation=0.,expected_descent_violation=0.,virtual_identity=0.,variation_violation=0.)
active=0;cases=0
for trial in range(2000):
 d=int(rng.choice([1,7,31]));x=rng.normal(size=d);mu=np.sin(x)+.1*x;b=rng.normal(size=d);h=rng.uniform(.05,5,size=d);v=rng.normal(size=d)*.2;j=int(rng.integers(1,50));C=2.
 noise=[v,-v,2*v,-2*v];vals=[];rights=[]
 for xi in noise:
  g=mu+xi;m=beta*b+a*g;ht=h/np.sqrt(.95+.05*(h*g)**2);hn,n=refresh(h,m,b,ht,C,j);active+=n;cases+=1
  M=np.sum(h*m*m);I=np.sum(h*(m-b)**2);V=np.sum((hn-h)*m*m)
  lhs=-eta*np.dot(g,h*m)+q*np.sum(hn*m*m)-q*np.sum(h*b*b)
  rhs=-eta*M-q*I+q*V
  mx['identity']=max(mx['identity'],abs(lhs-rhs))
  mx['budget_violation']=max(mx['budget_violation'],np.max((hn-h)*m*m-h*(c*m*m+zeta*(m-b)**2)))
  mx['variation_violation']=max(mx['variation_violation'],np.max(abs(hn-h))-C/j)
  xp=x-eta*h*m;delta=f(xp)-f(x)+q*np.sum(hn*m*m)-q*np.sum(h*b*b)
  deterministic=-(1-theta)*eta*M-(1-zeta)*q*I+L*eta**2/2*np.sum((h*m)**2)
  mx['path_descent_violation']=max(mx['path_descent_violation'],delta-deterministic-eta*np.dot(xi,h*m))
  vals.append(delta);rights.append(deterministic)
  hp=rng.uniform(.05,5,size=d);k=beta/a;z=x-eta*k*hp*b;zp=xp-eta*k*h*m
  expected=-eta*h*g+eta*k*(hp-h)*b
  mx['virtual_identity']=max(mx['virtual_identity'],np.max(abs(zp-z-expected)))
 nu=2.5*v*v
 mx['expected_descent_violation']=max(mx['expected_descent_violation'],np.mean(vals)-np.mean(rights)-eta*a*np.sum(h*nu))
assert max(mx.values())<1e-10,mx
# The zero momentum branch must permit every finite positive proposed scale.
zero,_=refresh(np.ones(3),np.zeros(3),np.ones(3),np.array([2.,.5,1.5]),10.,1)
assert np.allclose(zero,[2,.5,1.5])
out=dict(cases=cases,energy_limited_coordinates=int(active),maxima=mx,zero_momentum_case=zero.tolist(),scope='algebra and finite-distribution nonconvex inequalities only; no training or timing claim')
Path(__file__).with_name('37_MATH_CHECKS.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
