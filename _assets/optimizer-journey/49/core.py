# Optimizer-only excerpt; see SOURCE_MANIFEST.json for original SHA-256.
import math

import torch

from torch import nn

from torch.nn import functional as F

B1 = 0.9

RHO = 0.999

EPS2 = 1e-16

class C49(torch.optim.Optimizer):
    """48 + asymmetric signal-weighted denominator A = 1/sqrt(S/4 + max(N, S/4) + eps^2)
    + per-tensor RMS protection U = Z/max(1, RMS(Z)).  1D params keep a per-coordinate residual EMA."""
    def __init__(self,params,lr=1e-3,beta=B1,rho=RHO,eps2=EPS2):
        super().__init__(params,dict(lr=lr,beta=beta,rho=rho,eps2=eps2))
        for grp in self.param_groups:
            for p in grp['params']:
                st=self.state[p];st['m']=torch.zeros_like(p);st['t']=0
                if p.ndim>=2:
                    st['r']=torch.zeros(p.shape[0],device=p.device)
                    st['c']=torch.zeros(p.numel()//p.shape[0],device=p.device)
                else:
                    st['n']=torch.zeros_like(p)
    @torch.no_grad()
    def step(self,closure=None):
        loss=closure() if closure is not None else None
        for grp in self.param_groups:
            lr=grp['lr'];beta=grp['beta'];rho=grp['rho'];eps2=grp['eps2']
            for p in grp['params']:
                if p.grad is None:continue
                st=self.state[p];st['t']+=1;t=st['t']
                G=p.grad;m=st['m']
                if p.ndim>=2:
                    G2=G.reshape(p.shape[0],-1)
                    if t==1:
                        rm=G2.square().mean(1);cm=G2.square().mean(0)
                        R=rm[:,None]*cm[None,:]/rm.mean().clamp_min(1e-30)
                        Z=G2/torch.sqrt(torch.maximum(G2.square(),R)+eps2)
                    else:
                        m2=m.reshape(p.shape[0],-1)
                        S=(m2/(1-beta**(t-1))).square()
                        rh=st['r']/(1-rho**(t-1));ch=st['c']/(1-rho**(t-1))
                        N=rh[:,None]*ch[None,:]/rh.mean().clamp_min(1e-30)
                        A=torch.rsqrt(S*0.25+torch.maximum(N,S*0.25)+eps2)
                        E2=G2-m2
                        m.mul_(beta).add_(G,alpha=1-beta)
                        Z=A*(m.reshape(p.shape[0],-1)/(1-beta**t))
                        rms=Z.square().mean().sqrt()
                        Z=Z/max(1.0,float(rms))
                        p.add_(Z.reshape(p.shape),alpha=-lr)
                        st['r'].mul_(rho).add_(E2.square().mean(1),alpha=(1-rho)*beta)
                        st['c'].mul_(rho).add_(E2.square().mean(0),alpha=(1-rho)*beta)
                        continue
                    m.mul_(beta).add_(G,alpha=1-beta)
                    rms=Z.square().mean().sqrt();Z=Z/max(1.0,float(rms))
                    p.add_(Z.reshape(p.shape),alpha=-lr)
                    st['r'].mul_(rho).add_((G2-m.reshape(p.shape[0],-1)).square().mean(1),alpha=(1-rho)*beta)
                    st['c'].mul_(rho).add_((G2-m.reshape(p.shape[0],-1)).square().mean(0),alpha=(1-rho)*beta)
                else:
                    if t==1:
                        Z=G/torch.sqrt(torch.maximum(G.square(),G.square())+eps2)
                    else:
                        S=(m/(1-beta**(t-1))).square()
                        N=st['n']/(1-rho**(t-1))
                        A=torch.rsqrt(S*0.25+torch.maximum(N,S*0.25)+eps2)
                        E=G-m
                        m.mul_(beta).add_(G,alpha=1-beta)
                        Z=A*(m/(1-beta**t))
                        rms=Z.square().mean().sqrt();Z=Z/max(1.0,float(rms))
                        p.add_(Z,alpha=-lr)
                        st['n'].mul_(rho).add_(E.square(),alpha=(1-rho)*beta)
                        continue
                    m.mul_(beta).add_(G,alpha=1-beta)
                    rms=Z.square().mean().sqrt();Z=Z/max(1.0,float(rms))
                    p.add_(Z,alpha=-lr)
                    st['n'].mul_(rho).add_((G-m).square(),alpha=(1-rho)*beta)
        return loss
