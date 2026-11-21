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

@triton.jit
def apply_41(P,G,M,Q,V,IDS,DEC,PART,LR,WD,BC2,FIRST:tl.constexpr,MODE:tl.constexpr,N:tl.constexpr,W:tl.constexpr):
 tile=tl.program_id(0);i=tile*W+tl.arange(0,W);mask=i<N;b=tl.load(IDS+tile)
 old=tl.load(M+i,mask,0);g=tl.load(G+i,mask,0);p=tl.load(P+i,mask,0);q=tl.load(Q+b);v=tl.load(V+b);dec=tl.load(DEC+b)
 h=1.
 if not FIRST:h=tl.minimum(1e4,tl.maximum(1e-3,tl.rsqrt(v/BC2+1e-16)))
 k=1.-(19./36.)*q/(old*old+q+1e-16)
 if MODE==1:k=tl.minimum(1.,tl.maximum(.1,(old*old-q/18.)/(old*old+q+1e-16)))
 if MODE==2:k=1.
 if MODE==3:k=17./36.
 y=h*k*g;e=y-old;m=.9*old+.1*y
 tl.store(M+i,m,mask);tl.store(P+i,p*(1-LR*WD*dec)-LR*m,mask)
 tl.store(PART+tile,tl.sum(e*e,0));tl.store(PART+N//W+tile,tl.sum(g*g,0))

@triton.jit
def stats_41(Q,V,PART,START,COUNT,SIZE,NT:tl.constexpr,W:tl.constexpr):
 b=tl.program_id(0);start=tl.load(START+b);count=tl.load(COUNT+b);n=tl.load(SIZE+b);i=tl.arange(0,W)
 se=tl.sum(tl.load(PART+start+i,i<count,0),0)/n
 sg=tl.sum(tl.load(PART+NT+start+i,i<count,0),0)/n
 tl.store(Q+b,.999*tl.load(Q+b)+.0009*se)
 tl.store(V+b,.999*tl.load(V+b)+.001*sg)

@triton.jit
def apply42(P,G,M,Q,IDS,START,SIZE,LR,FIRST:tl.constexpr,MODE:tl.constexpr,N:tl.constexpr,W:tl.constexpr):
 tile=tl.program_id(0);i=tile*W+tl.arange(0,W);b=tl.load(IDS+tile)
 begin=tl.load(START+b)*W;n=tl.load(SIZE+b);mask=(i<begin+n)&(i<N)
 g=tl.load(G+i,mask,0);old=tl.load(M+i,mask,0);p=tl.load(P+i,mask,0);q=tl.load(Q+tile)
 e=g-old;m=.9*old+.1*g;count=tl.minimum(W,begin+n-tile*W)
 inc=.9*tl.sum(e*e,0)/count
 if MODE==1:inc=.9*tl.sum(g*g,0)/count
 newq=q+inc
 if MODE==2:newq=.999*q+.0009*tl.sum(e*e,0)/count
 use=q
 if FIRST:use=newq
 tl.store(P+i,p-LR*m*tl.rsqrt(use+1e-16),mask);tl.store(M+i,m,mask);tl.store(Q+tile,newq)

@triton.jit
def apply43(P,G,M,Q,IDS,START,SIZE,LR,WD,DEC,BC1,BC2,N:tl.constexpr,W:tl.constexpr):
 tile=tl.program_id(0);i=tile*W+tl.arange(0,W);b=tl.load(IDS+tile);begin=tl.load(START+b)*W;n=tl.load(SIZE+b)
 mask=(i<begin+n)&(i<N);g=tl.load(G+i,mask,0);old=tl.load(M+i,mask,0);p=tl.load(P+i,mask,0);q=tl.load(Q+tile)
 e=g-old;m=.9*old+.1*g;r=.999*q+.0009*e*e;mh=m/BC1
 u=mh*tl.rsqrt(mh*mh+r/BC2+1e-16);dec=tl.load(DEC+b)
 tl.store(P+i,p*(1-LR*WD*dec)-LR*u,mask);tl.store(M+i,m,mask)
 count=tl.minimum(W,begin+n-tile*W);tl.store(Q+tile,tl.sum(tl.where(mask,r,0.),0)/count)

class Optim:
 def __init__(self,e,method,lr,wd):
  self.e=e;self.method=method;self.wd=wd;self.t=0;self.opt=None;self.peak=lr
  if method=='43':
   self.n=e.n;self.m=torch.zeros(self.n,device='cuda');self.q=torch.zeros(self.n//TILE,device='cuda');self.unscale_proxy=torch.optim.SGD(e.params,lr=0.);return
  if method in ['Muon','Lion','SOAP','Adafactor','Layer37']:
   groups=e.groups(e.params,wd)
   if method=='Lion':
    from lion_pytorch import Lion
    self.opt=Lion(groups,lr=lr,betas=(.9,.99))
   if method=='SOAP':
    from soap_soap import SOAP
    self.opt=SOAP(groups,lr=lr,betas=(.95,.95),precondition_frequency=10,max_precond_dim=1024)
   if method=='Adafactor':self.opt=torch.optim.Adafactor(groups,lr=lr)
   if method=='Muon':
    from muon_muon import SingleDeviceMuonWithAuxAdam
    hidden=e.params[:e.nhidden];aux=e.params[e.nhidden:]
    self.opt=SingleDeviceMuonWithAuxAdam([dict(params=hidden,use_muon=True,lr=lr,weight_decay=wd,momentum=.95)]+[dict(params=g['params'],use_muon=False,lr=lr*.015,weight_decay=g['weight_decay'],betas=(.9,.95),eps=1e-8) for g in e.groups(aux,wd) if g['params']])
   if method=='Layer37':
    import layer_source
    self.inner=layer_source.Optim(e,'Layer37',lr);self.inner.wd=wd;self.unscale_proxy=torch.optim.SGD(e.params,lr=0.)
   return
  if method in ['AdamW','SGD']:
   groups=e.groups(e.params,wd)
   self.opt=torch.optim.AdamW(groups,lr=lr,betas=(.9,.999),eps=1e-8,fused=True) if method=='AdamW' else torch.optim.SGD(groups,lr=lr,momentum=.9,foreach=True)
   return
  if method in ['42','FullEnergy','ResidualEMA']:
   self.n=e.n;self.m=torch.zeros(self.n,device='cuda');self.q=torch.zeros(self.n//TILE,device='cuda');self.unscale_proxy=torch.optim.SGD(e.params,lr=0.);return
  if method in ['41','40','NoGate','FixedGate']:
   self.n=e.n;self.ng=e.ng;self.m=torch.zeros(self.n,device='cuda');self.q=torch.zeros(self.ng,device='cuda');self.v=torch.zeros_like(self.q);self.partial=torch.zeros((2,self.n//TILE),device='cuda');self.unscale_proxy=torch.optim.SGD(e.params,lr=0.);return
  hybrid=method=='40S-hybrid';self.n=e.hidden_n if hybrid else e.n;self.ng=e.nhidden if hybrid else e.ng
  self.m=torch.zeros(self.n,device='cuda');self.q=torch.zeros(self.ng,device='cuda');self.partial=torch.zeros(self.n//TILE,device='cuda')
  if hybrid:
   self.fallback=FALLBACK_CONFIG.get(e.task,dict(lr=lr,wd=wd))
   self.opt=torch.optim.AdamW(e.groups(e.params[e.nhidden:],self.fallback['wd']),lr=self.fallback['lr'],betas=(.9,.999),eps=1e-8,fused=True)
 def step(self,lr):
  self.t+=1;e=self.e
  if self.method=='Layer37':self.inner.step(lr);return
  if self.method=='43':
   apply43[(self.n//TILE,)](e.p,e.g,self.m,self.q,e.ids,e.starts,e.sizes,lr,self.wd,e.decay,1-.9**self.t,1-.999**self.t,N=self.n,W=TILE)
  elif self.method in ['42','FullEnergy','ResidualEMA']:
   apply42[(self.n//TILE,)](e.p,e.g,self.m,self.q,e.ids,e.starts,e.sizes,lr,FIRST=self.t==1,MODE={'42':0,'FullEnergy':1,'ResidualEMA':2}[self.method],N=self.n,W=TILE)
  elif hasattr(self,'v'):
   mode={'41':0,'40':1,'NoGate':2,'FixedGate':3}[self.method]
   apply_41[(self.n//TILE,)](e.p,e.g,self.m,self.q,self.v,e.ids,e.decay,self.partial,lr,self.wd,1-.999**(self.t-1),FIRST=self.t==1,MODE=mode,N=self.n,W=TILE)
   stats_41[(self.ng,)](self.q,self.v,self.partial,e.starts,e.count,e.sizes,NT=self.n//TILE,W=e.maxpart)
  elif hasattr(self,'m'):
   apply_s[(self.n//TILE,)](e.p,e.g,self.m,self.q,e.ids,e.decay,self.partial,lr,self.wd,1-.9**self.t,1-.999**(self.t-1),FIRST=self.t==1,N=self.n,W=TILE)
   stats_s[(self.ng,)](self.q,self.partial,e.starts,e.count,e.sizes,W=e.maxpart)
  if self.opt:
   for g in self.opt.param_groups:g['lr']=(lr if g.get('use_muon',True) else lr*.015) if self.method=='Muon' else lr/self.peak*self.fallback['lr'] if hasattr(self,'fallback') else lr
   self.opt.step()
 def state_bytes(self):
  if hasattr(self,'inner'):
   return sum(v.numel()*v.element_size() for k,v in vars(self.inner).items() if k in ['m','h','hits'] and torch.is_tensor(v))
  own=sum(getattr(self,k).numel()*4 for k in ['m','q','v'] if hasattr(self,k))
  if self.opt:
   def count(v):
    if torch.is_tensor(v):return v.numel()*v.element_size()
    if isinstance(v,dict):return sum(count(x) for x in v.values())
    if isinstance(v,(list,tuple)):return sum(count(x) for x in v)
    return 0
   own+=count(self.opt.state)
  return own
 def workspace_bytes(self):
  if hasattr(self,'inner'):return self.inner.partial.numel()*4+self.inner.act.numel()*4
  return self.partial.numel()*4 if hasattr(self,'partial') else 0
