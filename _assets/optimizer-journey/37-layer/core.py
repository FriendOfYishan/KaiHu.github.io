"""Block version: verify identities and inequalities, not empirical generalization."""
import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(3711142)
a=.1; beta=.9; theta=.05; zeta=.5; c=2*a*theta/beta;eta=.01;q=eta*beta/(2*a);L=1.1
sizes=[1,3,8,17];cuts=np.cumsum([0]+sizes);groups=[slice(cuts[i],cuts[i+1]) for i in range(len(sizes))];d=sum(sizes)
def energy(x):return np.sum(1-np.cos(x)+.05*x*x)
def stats(x):return np.array([np.dot(x[s],x[s]) for s in groups])
def expand(h):return np.repeat(h,sizes)
mx={k:0. for k in ['identity_error','budget_violation','conditional_descent_violation','variation_violation','compression_identity_error']};active=0
for state in range(1000):
 x=rng.normal(size=d);mu=np.sin(x)+.1*x;old=rng.normal(size=d);h=rng.uniform(.1,3,size=4);noise=rng.normal(size=d)*.2
 diffs=[];bounds=[];j=int(rng.integers(1,40));C=2.
 for xi in [noise,-noise,2*noise,-2*noise]:
  g=mu+xi;m=beta*old+a*g;M=stats(m);J=stats(m-old);S=stats(g)/sizes
  trial=h/np.sqrt(.95+.05*h*h*S);trial=np.clip(trial,np.maximum(.05,h-C/j),np.minimum(5,h+C/j))
  increase=np.maximum(trial-h,0);cost=increase*M;budget=h*(c*M+zeta*J);limited=cost>budget
  hn=trial.copy();hn[limited]=h[limited]+budget[limited]/M[limited];active+=int(np.sum(limited))
  H=expand(h);xn=x-eta*H*m
  lhs=-eta*np.dot(g,H*m)+q*np.dot(hn,M)-q*np.dot(h,stats(old));rhs=-eta*np.dot(h,M)-q*np.dot(h,J)+q*np.dot(hn-h,M)
  mx['identity_error']=max(mx['identity_error'],abs(lhs-rhs))
  mx['budget_violation']=max(mx['budget_violation'],np.max((hn-h)*M-budget))
  mx['variation_violation']=max(mx['variation_violation'],np.max(abs(hn-h))-C/j)
  diffs.append(energy(xn)-energy(x)+q*np.dot(hn,M)-q*np.dot(h,stats(old)))
  bounds.append(-(1-theta)*eta*np.dot(h,M)-(1-zeta)*q*np.dot(h,J)+L*eta**2/2*np.dot(H*m,H*m))
 mx['conditional_descent_violation']=max(mx['conditional_descent_violation'],np.mean(diffs)-np.mean(bounds)-eta*a*np.dot(h,2.5*stats(noise)))
 # Exact loss of representational capacity from a layer scalar.
 diag=rng.uniform(.05,5,size=d)
 for sl in groups:
  b=old[sl];u=diag[sl];best=np.dot(b*b,u)/np.dot(b,b)
  mx['compression_identity_error']=max(mx['compression_identity_error'],abs(np.dot(u*b-best*b,u*b-best*b)-np.dot(b*b,(u-best)**2)))
assert max(mx.values())<1e-10,mx
assert active>0
out=dict(states=1000,finite_noise_cases=4000,layer_sizes=sizes,limited_layer_events=active,maxima=mx,
 persistent_state_floats=d+len(sizes),adam_two_moment_floats=2*d,
 note='State counts are mathematical; peak GPU allocation and training speed were not measured.')
Path(__file__).with_name('37_LAYER_CHECKS.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
