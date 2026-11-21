import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(20);err=0.;normalized_err=0.
m=np.zeros(9);v=np.zeros(9);s=np.zeros(9);rawm=np.zeros(9);rawv=np.zeros(9);mn=np.zeros(9);vn=np.zeros(9);z=0.
for t in range(1,2001):
 a=.1*t**(-.4);g=rng.normal(size=9);an=1 if t==1 else a
 sn=(1-an)*s+an*(1-an)*(g-m)**2
 m=(1-an)*m+an*g;v=(1-an)*v+an*g*g
 err=max(err,float(np.max(np.abs(sn-(v-m*m)))));s=sn
 z=(1-a)*z+a;rawm=(1-a)*rawm+a*g;rawv=(1-a)*rawv+a*g*g
 w=a/z;mn=(1-w)*mn+w*g;vn=(1-w)*vn+w*g*g
 normalized_err=max(normalized_err,float(np.max(np.abs(mn-rawm/z))),float(np.max(np.abs(vn-rawv/z))))
 assert np.all(mn*mn<=vn+1e-12)
assert err<1e-12 and normalized_err<1e-12
cov_err=0.
for a in [.01,.1,.5,1.]:
 for n in [1,3,50,200]:
  cov=a/(2-a)*(1-a)**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
  formula=n-2*(1-a)*(1-(1-a)**n)/(a*(2-a))
  cov_err=max(cov_err,abs(cov.sum()-formula))
assert cov_err<1e-9
for _ in range(1000):
 h=rng.uniform(.1,3,size=9);x=rng.normal(size=9);mu=h*x;d=np.exp(rng.normal(size=9));e=rng.normal(size=9);m=mu+e;u=d*m;eta=.1
 A=np.dot(mu,d*mu);R=np.dot(e,d*e)/A;K=max(h)*eta*np.dot(u,u)/(2*A)
 lhs=.5*np.dot(h*(x-eta*u),x-eta*u);rhs=.5*np.dot(h*x,x)-eta*A*(1-np.sqrt(R)-K)
 assert lhs<=rhs+1e-10
weights=[]
for p in [.2,1/3,.4,.5,.6,.8,1.]:
 t=np.arange(2,1025);a=.1*t**(-p);b=float(np.prod(1-a));weights.append(dict(p=p,first_weight_pure=b,first_weight_normalized=.1*b/(1-.9*b),first_weight_max=float(np.prod(1-np.maximum(2/(t+1),a)))))
out=dict(variance_identity_max_error=err,normalized_moments_max_error=normalized_err,cumulative_covariance_max_error=cov_err,descent_certificate_checks=1000,weights=weights)
Path(__file__).with_name('verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
