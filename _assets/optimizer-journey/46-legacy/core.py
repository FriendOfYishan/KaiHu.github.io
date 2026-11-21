import lab45
# Optimizer-only excerpt; see SOURCE_MANIFEST.json for original SHA-256.
import math

import torch

import lab45

B1 = lab45.BETA

TAU = 4.0

C_L1 = math.sqrt(math.pi / 2)

ARMS = ['46a', '46b']

class Optim46(lab45.Optim):
 def __init__(self,e,method,lr,wd):
  if method not in ARMS:
   super().__init__(e,method,lr,wd);return
  self.e=e;self.method=method;self.t=0;self.wd=wd;self.peak=lr;self.opt=None
  self.unscale_proxy=torch.optim.SGD(e.params,lr=0.)
  self.m=torch.zeros(e.n,device='cuda')
  if method=='46a':self.q=torch.zeros(e.ntiles,device='cuda')
  else:self.s=torch.zeros(e.n,device='cuda')
 def step(self,lr):
  self.t+=1;e=self.e;g=e.g
  m_old=self.m.clone()
  self.m.mul_(B1).add_(g,alpha=1-B1)
  if self.method=='46a':
   qhat=(self.q/(1-.999**self.t) if self.t>1 else self.q).repeat_interleave(lab45.TILE)
   den=self.m.abs()+qhat+1e-8
   u=(self.m/den).clamp(-1.,1.)
  else:
   scale=C_L1*(self.s/(1-.999**self.t) if self.t>1 else self.s)
   u=(self.m/torch.sqrt(self.m*self.m+scale*scale+1e-12)).clamp_(-TAU,TAU)
  e.p.mul_(1-lr*self.wd).add_(u,alpha=-lr)
  if self.method=='46a':
   resid=(g-m_old).abs().view(e.ntiles,lab45.TILE).sum(1)/e.tile_valid_count
   self.q.mul_(.999).add_(resid,alpha=.001)
  else:
   self.s.mul_(.999).add_((g-m_old).abs(),alpha=.001)
 def state_bytes(self):
  if self.method=='46a':return self.m.numel()*4+self.q.numel()*4
  if self.method=='46b':return self.m.numel()*4+self.s.numel()*4
  return super().state_bytes()
 def diagnostics(self):
  if self.method not in ARMS:return super().diagnostics()
  with torch.no_grad():
   mask=self.e.valid
   if self.method=='46a':
    scale=(self.q/(1-.999**self.t) if self.t>1 else self.q).repeat_interleave(lab45.TILE);u=(self.m/(self.m.abs()+scale+1e-8)).clamp(-1.,1.)
   else:
    scale=C_L1*(self.s/(1-.999**self.t) if self.t>1 else self.s);u=(self.m/torch.sqrt(self.m*self.m+scale*scale)).clamp_(-TAU,TAU)
   sc=scale[mask];uu=u[mask];mm=self.m[mask].abs()
   sub=sc[::max(1,sc.numel()//200000)]
   qs=torch.quantile(sub.float(),torch.tensor([.05,.5,.95],device='cuda')).tolist()
   return dict(a_mean=float(sc.mean()),a_p05=qs[0],a_p50=qs[1],a_p95=qs[2],
    clip_hi=float((uu.abs()>=min(1.,TAU)-1e-6).float().mean()),
    resid_share=float((sc/(mm+sc)).mean()),upd_rms=float(uu.pow(2).mean().sqrt()))
