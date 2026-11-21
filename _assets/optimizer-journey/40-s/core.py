# Optimizer-only excerpt; see SOURCE_MANIFEST.json for original SHA-256.
import math

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

import triton

import triton.language as tl

TILE = 1024

FALLBACK_CONFIG = {}

class Packed:
 def __init__(self,model,task):
  self.task=task
  named=list(model.named_parameters())
  def hidden(n,p):return p.ndim>=2 and not n.startswith(('embed.','pos.','fc.'))
  named.sort(key=lambda z:not hidden(*z));self.params=[];self.names=[];self.shapes=[]
  sizes=[p.numel() for _,p in named];pad=[triton.cdiv(n,TILE)*TILE for n in sizes];starts=np.cumsum([0]+pad[:-1]);self.n=sum(pad)
  self.p=torch.zeros(self.n,device='cuda');self.g=torch.zeros_like(self.p)
  for start,(name,p) in zip(starts,named):
   n=p.numel();self.p[start:start+n].copy_(p.detach().flatten());p.data=self.p[start:start+n].view(p.shape);p.grad=self.g[start:start+n].view(p.shape)
   self.params.append(p);self.names.append(name);self.shapes.append(tuple(p.shape))
  self.actual=sum(sizes);self.ng=len(sizes);self.nhidden=sum(hidden(*z) for z in named);self.hidden_n=sum(pad[:self.nhidden])
  self.ids=torch.repeat_interleave(torch.arange(self.ng,device='cuda',dtype=torch.int32),torch.tensor(np.array(pad)//TILE,device='cuda'))
  self.starts=torch.tensor(starts//TILE,device='cuda',dtype=torch.int32);self.count=torch.tensor(np.array(pad)//TILE,device='cuda',dtype=torch.int32);self.sizes=torch.tensor(sizes,device='cuda',dtype=torch.int32)
  self.decay=torch.tensor([int(p.ndim>=2) for _,p in named],device='cuda',dtype=torch.int32);self.maxpart=triton.next_power_of_2(max(pad)//TILE)
 def groups(self,params,wd):
  return [dict(params=[p for p in params if p.ndim>=2],weight_decay=wd),dict(params=[p for p in params if p.ndim<2],weight_decay=0.)]

@triton.jit
def apply_s(P,G,M,Q,IDS,DEC,PART,LR,WD,BC1,BC2,FIRST:tl.constexpr,N:tl.constexpr,W:tl.constexpr):
 tile=tl.program_id(0);i=tile*W+tl.arange(0,W);mask=i<N;b=tl.load(IDS+tile)
 old=tl.load(M+i,mask,0);g=tl.load(G+i,mask,0);p=tl.load(P+i,mask,0);q=tl.load(Q+b);dec=tl.load(DEC+b)
 e=g-old;m=.9*old+.1*g;mh=m/BC1
 nr=0.
 if not FIRST:nr=q/BC2
 u=mh/(tl.sqrt(mh*mh+nr)+1e-8)
 tl.store(M+i,m,mask);tl.store(P+i,p*(1-LR*WD*dec)-LR*u,mask);tl.store(PART+tile,tl.sum(e*e,0))

@triton.jit
def stats_s(Q,PART,START,COUNT,SIZE,W:tl.constexpr):
 b=tl.program_id(0);start=tl.load(START+b);count=tl.load(COUNT+b);n=tl.load(SIZE+b);i=tl.arange(0,W)
 ss=tl.sum(tl.load(PART+start+i,i<count,0),0)/n
 tl.store(Q+b,.999*tl.load(Q+b)+.0009*ss)

class Optim:
 def __init__(self,e,method,lr,wd):
  self.e=e;self.method=method;self.wd=wd;self.t=0;self.opt=None;self.peak=lr
  if method in ['AdamW','SGD']:
   groups=e.groups(e.params,wd)
   self.opt=torch.optim.AdamW(groups,lr=lr,betas=(.9,.999),eps=1e-8,fused=True) if method=='AdamW' else torch.optim.SGD(groups,lr=lr,momentum=.9,foreach=True)
   return
  hybrid=method=='40S-hybrid';self.n=e.hidden_n if hybrid else e.n;self.ng=e.nhidden if hybrid else e.ng
  self.m=torch.zeros(self.n,device='cuda');self.q=torch.zeros(self.ng,device='cuda');self.partial=torch.zeros(self.n//TILE,device='cuda')
  if hybrid:
   self.fallback=FALLBACK_CONFIG.get(e.task,dict(lr=lr,wd=wd))
   self.opt=torch.optim.AdamW(e.groups(e.params[e.nhidden:],self.fallback['wd']),lr=self.fallback['lr'],betas=(.9,.999),eps=1e-8,fused=True)
 def step(self,lr):
  self.t+=1;e=self.e
  if hasattr(self,'m'):
   apply_s[(self.n//TILE,)](e.p,e.g,self.m,self.q,e.ids,e.decay,self.partial,lr,self.wd,1-.9**self.t,1-.999**(self.t-1),FIRST=self.t==1,N=self.n,W=TILE)
   stats_s[(self.ng,)](self.q,self.partial,e.starts,e.count,e.sizes,W=e.maxpart)
  if self.opt:
   for g in self.opt.param_groups:g['lr']=lr/self.peak*self.fallback['lr'] if hasattr(self,'fallback') else lr
   self.opt.step()
 def state_bytes(self):
  own=sum(getattr(self,k).numel()*4 for k in ['m','q'] if hasattr(self,k))
  if self.opt:own+=sum(v.numel()*v.element_size() for s in self.opt.state.values() for v in s.values() if torch.is_tensor(v))
  return own
 def workspace_bytes(self):return self.partial.numel()*4 if hasattr(self,'partial') else 0
