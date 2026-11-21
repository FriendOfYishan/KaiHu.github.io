"""Finite-noise checks of actual descent bounds; no training experiment."""
from pathlib import Path
import itertools,json
import numpy as np

def axis(m,q,eps=.001):
 z=np.clip((5/12)*m/np.sqrt(q+eps*eps),-1,1)
 c=m/np.sqrt(np.mean(m*m)+q+eps*eps)
 d=z*(2-np.abs(z))-c
 n=d@d+c@c
 return np.outer(d,d)/n if n>0 else np.zeros((m.size,m.size))

def main():
 rng=np.random.default_rng(3939);beta=.9;a=1-beta
 signs=np.array(list(itertools.product([-1.,1.],repeat=5)))
 passed=0;max_violation=-np.inf;best_margin=0.
 for _ in range(600):
  x=rng.uniform(-2.7,2.7,5);f=np.sin(x);m=.5*f+rng.normal(0,.2,5)
  p=axis(m,.1);eta=10**rng.uniform(-4,-.5);noise=.15*signs;g=f+noise
  projected=g@p.T;m0=beta*m+a*g;mb=m0+a*projected
  loss=lambda y:(1-np.cos(y)).sum(axis=-1)
  actual=np.mean(loss(x-eta*mb)-loss(x-eta*m0))
  s=f@p@f;q=np.mean(np.sum(projected**2,axis=1));b0=np.mean(np.sum(m0**2,axis=1))
  bound=-eta*a*s+eta*eta*a*np.sqrt(b0*q)+.5*eta*eta*a*a*q
  max_violation=max(max_violation,actual-bound)
  assert actual<=bound+1e-12
  if s>eta*(np.sqrt(b0*q)+a*q/2):
   passed+=1;assert actual<1e-12;best_margin=max(best_margin,-actual)
 result={'nonconvex_cosine_exact_noise_enumeration':dict(states=600,noise_outcomes_per_state=32,condition_satisfied=passed,max_actual_minus_upper_bound=float(max_violation),largest_observed_expected_advantage=float(best_margin))}
 # Orthogonal mean, noise on the chosen axis: the extra projection is harmful.
 m=np.array([1.,3.]);p=axis(m,.2);eig,vec=np.linalg.eigh(p);d=vec[:,-1];f=np.array([-d[1],d[0]])
 g=np.stack([f+.3*d,f-.3*d]);eta=.1;m0=beta*m+a*g;mb=m0+a*(g@p.T)
 actual=np.mean(.5*np.sum((f-eta*mb)**2,axis=1)-.5*np.sum((f-eta*m0)**2,axis=1))
 lam=eig[-1];formula=eta**2*a*a*.3**2*(lam+.5*lam*lam)
 assert abs(actual-formula)<1e-13 and actual>0
 result['orthogonal_noise_counterexample']=dict(actual_expected_harm=float(actual),formula=float(formula))
 # One vector and scalar recurrences reproduce the implicit full transformed moments.
 m=np.zeros(9);q=.2;n=np.full(9,q);v=n.copy();x=rng.normal(size=9);eta=.01;kappa=beta/a
 virtual_error=moment_error=mean_error=0.
 for _ in range(500):
  g=rng.normal(size=9);p=axis(m,q);aa=np.eye(9)+p;old=m.copy();oldx=x.copy();oldz=x-eta*kappa*old
  y=aa@g;m=beta*old+a*y;x=x-eta*m;q=beta*q+beta*a*np.mean((y-old)**2)
  n=beta*n+beta*a*(y-old)**2;v=beta*v+a*y*y
  virtual_error=max(virtual_error,np.max(np.abs((x-eta*kappa*m)-oldz+eta*y)))
  moment_error=max(moment_error,np.max(np.abs(v-m*m-n)));mean_error=max(mean_error,abs(q-n.mean()))
  ev=np.linalg.eigvalsh(aa);assert ev.min()>1-1e-12 and ev.max()<2+1e-12
 assert max(virtual_error,moment_error,mean_error)<1e-12
 result['streaming_identities']=dict(steps=500,virtual_point_max_error=float(virtual_error),moment_identity_max_error=float(moment_error),scalar_residual_max_error=float(mean_error))
 path=Path(__file__).with_name('39B_CHECKS.json');path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
