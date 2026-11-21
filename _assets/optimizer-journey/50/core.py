# Optimizer-only excerpt; see SOURCE_MANIFEST.json for original SHA-256.
import math

import torch

from torch import nn

from torch.nn import functional as F

B1 = 0.9

RHO = 0.999

EPS2 = 1e-16

class C50(torch.optim.Optimizer):
    """Row/column gains with K^2 = (a_i^2+b_j^2)/2 so the weighted residual budget is exact:
    L = (mean(R a^2)+mean(C b^2))/2, and the update is (K*B)/sqrt(D+L+eps^2) with a per-tensor
    scalar (RMS(U)^2 = D/(D+L) <= 1 by construction). 1D params keep a per-coordinate residual EMA."""
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
                        L=0.;K=None
                    else:
                        Bm=(m.reshape(p.shape[0],-1)/(1-beta**(t-1)))
                        s=Bm.square().mean(1);z=Bm.square().mean(0)
                        R=st['r']/(1-rho**(t-1));C=st['c']/(1-rho**(t-1))
                        a=(s+eps2)/(s+R+eps2);b=(z+eps2)/(z+C+eps2)
                        K=torch.sqrt(0.5*(a.square()[:,None]+b.square()[None,:]))
                        L=float((R*a.square()).mean()+(C*b.square()).mean())/2.
                    E2=G2-m.reshape(p.shape[0],-1)
                    m.mul_(beta).add_(G,alpha=1-beta)
                    B=m.reshape(p.shape[0],-1)/(1-beta**t)
                    KB=B if K is None else K*B
                    D=float(KB.square().mean())
                    U=KB/(D+L+eps2)**0.5
                    p.add_(U.reshape(p.shape),alpha=-lr)
                    st['r'].mul_(rho).add_(E2.square().mean(1),alpha=(1-rho)*beta)
                    st['c'].mul_(rho).add_(E2.square().mean(0),alpha=(1-rho)*beta)
                else:
                    if t==1:
                        L=0.;K=None
                    else:
                        Bm=m/(1-beta**(t-1));n=st['n']/(1-rho**(t-1))
                        K=(Bm.square()+eps2)/(Bm.square()+n+eps2)
                        L=float((n*K.square()).mean())
                    E=G-m
                    m.mul_(beta).add_(G,alpha=1-beta)
                    B=m/(1-beta**t)
                    KB=B if K is None else K*B
                    D=float(KB.square().mean())
                    U=KB/(D+L+eps2)**0.5
                    p.add_(U,alpha=-lr)
                    st['n'].mul_(rho).add_(E.square(),alpha=(1-rho)*beta)
        return loss
