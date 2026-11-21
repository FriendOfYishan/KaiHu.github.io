"""Analytical predictions: no observed GPU outcomes enter these calculations."""
import numpy as np
def weights(t,beta=.9):
 k=np.arange(1,t+1,dtype=float)
 return 2*k/(t*(t+1)),(1-beta)*beta**(t-k)/(1-beta**t)
def moments(p):
 n=len(p);return dict(A=float(p@np.arange(n-1,-1,-1)),V=float(p@p),R=float(np.sum(np.cumsum(p)[:-1]**2)))
def risk(p,d,q,sigma):
 c=moments(p);return d*d*c['A']**2+q*q*c['R']+sigma*sigma*c['V']
def feedback(eta,decay,sigma,T=1024,x0=1.):
 out={}
 for kind in ['AWA','EMA']:
  mu=np.array([x0,0.]);C=np.zeros((2,2))
  for t in range(1,T+1):
   a=2/(t+1) if kind=='AWA' else .1/(1-.9**t);lr=eta/t**decay
   M=np.array([[1-lr*a,-lr*(1-a)],[a,1-a]]);b=sigma*a*np.array([-lr,1.])
   mu=M@mu;C=M@C@M.T+np.outer(b,b)
  out[kind]=dict(mean=float(mu[0]),bias2=float(mu[0]**2),variance=float(C[0,0]),mse=float(mu[0]**2+C[0,0]))
 return out
def predictions():
 T=1024;p,e=weights(T);a,b=moments(p),moments(e)
 dc=np.sqrt((b['V']-a['V'])/(a['A']**2-b['A']**2));qc=np.sqrt((b['V']-a['V'])/(a['R']-b['R']))
 cases=[('stationary',0.,0.),('slow_coherent',.25*dc,0.),('fast_coherent',4*dc,0.),('slow_randomwalk',0.,.25*qc),('fast_randomwalk',0.,4*qc)]
 tracking=[dict(name=n,d=d,q=q,sigma=1.,AWA=risk(p,d,q,1),EMA=risk(e,d,q,1)) for n,d,q in cases]
 fb=[dict(name=f'decay{decay}_sigma{sig}',eta=.03,decay=decay,sigma=sig,prediction=feedback(.03,decay,sig)) for decay,sig in [(0,1),(.5,.25),(.5,1),(.5,4),(.5,8),(.5,16)]]
 return dict(T=T,beta=.9,AWA_moments=a,EMA_moments=b,critical_d_over_sigma=float(dc),critical_q_over_sigma=float(qc),tracking=tracking,feedback=fb)
