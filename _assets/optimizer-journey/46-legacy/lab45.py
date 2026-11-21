# Optimizer-only excerpt; see SOURCE_MANIFEST.json for original SHA-256.
import math

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

import triton

import triton.language as tl

TILE = 1024

BETA = 0.9

A_LO = 0.001

A_HI = 10000.0

H_SCALE = {'44': A_HI, '44-fp32': A_HI, '44-cur': A_HI, '44-nos': A_HI, '44-h1e5': 100000.0, '44-h1e6': 1000000.0}

OPT45 = ['45', '45-nocap', '45-b2fast', '45-mbf16', '45-t2', '45-t4', '45-t8', '45-t4mbf16', '45-t2mbf16']

class Packed:
 """Flatten every parameter tensor into one FP32 buffer. Candidate-44 groups are 1024-coordinate
 chunks that never cross a tensor boundary; padding coordinates at a tensor tail belong to no group."""
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
  self.ntiles=self.n//TILE;self.nhidden_tiles=self.hidden_n//TILE
  tile_start=self.starts[self.ids].long()*TILE;tile_size=self.sizes[self.ids].long()
  off=torch.arange(self.ntiles,device='cuda',dtype=torch.long)*TILE-tile_start
  self.tile_valid_count=torch.clamp(tile_size-off,min=0,max=TILE).float()
  self.valid=((torch.arange(self.n,device='cuda')-tile_start.repeat_interleave(TILE))<tile_size.repeat_interleave(TILE))
 def groups(self,params,wd):
  return [dict(params=[p for p in params if p.ndim>=2],weight_decay=wd),dict(params=[p for p in params if p.ndim<2],weight_decay=0.)]

@triton.jit
def apply44(P,G,S,U,Q,IDS,START,SIZE,LR,BC1,BC2,ALO,AHI,FIRST:tl.constexpr,MODE:tl.constexpr,N:tl.constexpr,W:tl.constexpr):
 """MODE 0: A from history, S EMA + group residual EMA  -> candidate 44 / 44-fp32 / 44-h1e5
    MODE 1: A from the current gradient and current group residual -> ablation 44-cur
    MODE 2: A from history S only, residual term removed -> ablation 44-nos
    S is BF16 for 44 and FP32 for 44-fp32; the store rounds to S's element type (Q)."""
 tile=tl.program_id(0);i=tile*W+tl.arange(0,W);mask=i<N;b=tl.load(IDS+tile)
 begin=tl.load(START+b)*W;n=tl.load(SIZE+b);mask=mask&(i<begin+n)
 g=tl.load(G+i,mask,0.).to(tl.float32);s_old=tl.load(S+i,mask,0.).to(tl.float32)
 u=tl.load(U+i,mask,0.);p=tl.load(P+i,mask,0.);q=tl.load(Q+tile)
 resid=g-s_old
 count=tl.maximum(tl.minimum(W,begin+n-tile*W).to(tl.float32),1.)
 q_inc=tl.sum(tl.where(mask,resid*resid,0.),0)/count
 a=1.
 if not FIRST:
  if MODE==1:
   a=tl.rsqrt(g*g+q_inc+1e-16)
  else:
   sa=s_old/BC1;r=q/BC2
   a=tl.rsqrt(sa*sa+(0. if MODE==2 else r)+1e-16)
  a=tl.minimum(tl.maximum(a,ALO),AHI)
 u_new=.9*u+.1*a*g
 tl.store(U+i,u_new,mask);tl.store(P+i,p-LR*u_new,mask)
 tl.store(Q+tile,.999*q+.0009*q_inc)
 tl.store(S+i,(.9*s_old+.1*g).to(S.dtype.element_ty),mask)

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
   self.m=torch.zeros(e.n,device='cuda');self.q=torch.zeros(e.n//TILE,device='cuda');self.unscale_proxy=torch.optim.SGD(e.params,lr=0.);return
  if method in H_SCALE:
   self.u=torch.zeros(e.n,device='cuda')
   self.s=torch.zeros(e.n,device='cuda',dtype=torch.float32 if method=='44-fp32' else torch.bfloat16)
   self.q=torch.zeros(e.n//TILE,device='cuda');self.unscale_proxy=torch.optim.SGD(e.params,lr=0.);return
  if method=='AdamW':
   self.opt=torch.optim.AdamW(e.groups(e.params,wd),lr=lr,betas=(.9,.999),eps=1e-8,fused=True);return
  if method in OPT45:
   self.b1=.9;self.b2=.9 if 'b2fast' in method else .999
   store=method.endswith('mbf16')
   self.m=torch.zeros(e.n,device='cuda',dtype=torch.bfloat16 if store else torch.float32)
   self.v=torch.zeros(e.n,device='cuda',dtype=torch.bfloat16 if (store or 'vbf16' in method) else torch.float32)
   self.unscale_proxy=torch.optim.SGD(e.params,lr=0.);return
  raise ValueError(method)
 def step(self,lr):
  self.t+=1;e=self.e
  if self.method=='43':
   apply43[(e.n//TILE,)](e.p,e.g,self.m,self.q,e.ids,e.starts,e.sizes,lr,self.wd,e.decay,1-.9**self.t,1-.999**self.t,N=e.n,W=TILE);return
  if self.method in H_SCALE:
   apply44[(e.n//TILE,)](e.p,e.g,self.s,self.u,self.q,e.ids,e.starts,e.sizes,lr,1-.9**(self.t-1),1-.999**(self.t-1),
    A_LO,H_SCALE[self.method],FIRST=self.t==1,MODE={'44':0,'44-fp32':0,'44-cur':1,'44-nos':2,'44-h1e5':0,'44-h1e6':0}[self.method],N=e.n,W=TILE);return
  if self.method in OPT45:
   g=e.g;bf16=self.m.dtype==torch.bfloat16
   m=self.m.float() if bf16 else self.m
   m.mul_(self.b1).add_(g,alpha=1-self.b1)
   v=self.v.float().mul_(self.b2).addcmul_(g,g,value=1-self.b2)
   mh=m/(1-self.b1**self.t);vh=v/(1-self.b2**self.t)
   den=(vh-mh*mh).clamp_min(1e-32).sqrt()
   u=mh/den
   if 'nocap' not in self.method:
    cap=8. if 't8' in self.method else (4. if 't4' in self.method else (2. if 't2' in self.method else 1.))
    u=u.clamp(-cap,cap)
   e.p.mul_(1-lr*self.wd).add_(u,alpha=-lr)
   if bf16:self.m.copy_(m)
   self.v.copy_(v)
   return
  for g in self.opt.param_groups:g['lr']=lr   # the scheduled lr must reach the torch optimizer too
  self.opt.step()
 def state_bytes(self):
  if self.method=='43':return self.m.numel()*4+self.q.numel()*4
  if self.method.startswith('44'):return self.u.numel()*4+self.s.numel()*self.s.element_size()+self.q.numel()*4
  if self.method in OPT45:return self.m.numel()*self.m.element_size()+self.v.numel()*self.v.element_size()
  def count(v):
   if torch.is_tensor(v):return v.numel()*v.element_size()
   if isinstance(v,dict):return sum(count(x) for x in v.values())
   if isinstance(v,(list,tuple)):return sum(count(x) for x in v)
   return 0
  return count(self.opt.state)
 def workspace_bytes(self):return 0
 def diagnostics(self):
  """Candidate-45 read-out: how much of the second moment is residual, and how much of the step is capped."""
  e=self.e;t=self.t
  if self.method in OPT45:
   if t<=1:return dict(a_mean=0.,a_p05=0.,a_p50=0.,a_p95=0.,clip_lo=0.,clip_hi=0.,resid_share=1.,upd_rms=1.,den_rms=0.,raw_p50=0.)
   m=self.m.float()/(1-self.b1**t);v=self.v.float()/(1-self.b2**t)
   den=(v-m*m).clamp_min(1e-32);a=den.rsqrt();u=(m*a)
   v_=e.valid;au=a[v_];du=den[v_];sub=au[::max(1,au.numel()//200000)]
   qs=torch.quantile(sub.float(),torch.tensor([.05,.5,.95],device='cuda')).tolist()
   return dict(a_mean=float(au.mean()),a_p05=qs[0],a_p50=qs[1],a_p95=qs[2],
    clip_lo=0.,clip_hi=float((u[v_].abs()>=1.).float().mean()),
    resid_share=float(((v-m*m)[v_]/v[v_]).mean()),upd_rms=float(u[v_].pow(2).mean().sqrt()),
    den_rms=float(du.sqrt().mean()),raw_p50=qs[1])
  if not (self.method.startswith('44') or self.method=='43'):return None
  if t<=1:return dict(a_mean=1.,a_p05=1.,a_p50=1.,a_p95=1.,clip_lo=0.,clip_hi=0.,resid_share=0.,upd_rms=0.,den_rms=0.,raw_p50=1.)
  v=e.valid
  hi=H_SCALE.get(self.method,A_HI)
  with torch.no_grad():
   if self.method=='43':
    sig=self.m/(1-.9**t);rb=(self.q/(1-.999**t)).repeat_interleave(TILE)
    den=sig*sig+rb;upd=(sig*den.rsqrt());a=den.rsqrt()
   else:
    sig=self.s.float()/(1-.9**(t-1));rb=(self.q.float()/(1-.999**(t-1))).repeat_interleave(TILE)
    den=sig*sig+rb;a=den.rsqrt();upd=self.u
   au=a[v];du=den[v]
   sub=au[::max(1,au.numel()//200000)]
   raw=torch.quantile(sub.float(),torch.tensor([.05,.5,.95],device='cuda')).tolist()
   cl=au.clamp(A_LO,hi)
   qs=torch.quantile(cl[::max(1,cl.numel()//200000)].float(),torch.tensor([.05,.5,.95],device='cuda')).tolist()
   return dict(a_mean=float(cl.mean()),a_p05=qs[0],a_p50=qs[1],a_p95=qs[2],
    clip_lo=float((au<A_LO).float().mean()),clip_hi=float((au>hi).float().mean()),
    resid_share=float((rb[v]/du).mean()),upd_rms=float(upd[v].pow(2).mean().sqrt()),
    den_rms=float(du.sqrt().mean()),raw_p50=raw[1])
