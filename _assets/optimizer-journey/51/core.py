"""Candidate 51 (as run in the previous round) + the ablation variants the reviewer asked for + decoupled
weight decay.

  B  = M/(1-beta^{t-1}); R = r/(1-rho^{t-1}); C = c/(1-rho^{t-1}); muR = mean(R)
  N~ = R_i C_j / muR
  P  = B / sqrt(B^2 + N~ + eps^2)                          (|P| <= 1)          history direction
  vR = rowmean(B^2)+R ; vC = colmean(B^2)+C ; v0 = mean(vR)
  K^2 = 0.5/(v0+eps^2) + 0.25/(vR_i+eps^2) + 0.25/(vC_j+eps^2)                 innovation gain (with floor)
  omega = (1-beta)/(1-beta^t)
  U  = P + clip(omega*K*(G-B), -2, 2)                                          (|U| <= 3)
  t = 1: U = G / sqrt(max(G^2, mean(G^2)) + eps^2)
  W  <- (1 - lr*lam) W - lr*U                                                  decoupled decay

modes:
  'full'   : the candidate as measured (5.4718 mean val NLL at 4.12M tokens)
  'p48gain': the innovation gain K is replaced by 48's per-coordinate A48 = 1/sqrt(B^2+N~)
             (isolates what 50's square-additive gain contributes)
  'noclip' : no +-2 clip on the innovation
  'nofloor': the global-scale share is removed and the row/col weights are redistributed
             K^2 = 0.5/(vR+eps^2) + 0.5/(vC+eps^2)
"""
import torch
from optzoo import B1,RHO,EPS2

MODES=('full','p48gain','noclip','nofloor')

class C51(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,beta=B1,rho=RHO,eps2=EPS2,mode='full',wd=0.):
        super().__init__(params,dict(lr=lr,beta=beta,rho=rho,eps2=eps2,weight_decay=wd))
        self.mode=mode
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
            lr=grp['lr'];beta=grp['beta'];rho=grp['rho'];eps2=grp['eps2'];lam=grp['weight_decay']
            for p in grp['params']:
                if p.grad is None:continue
                st=self.state[p];st['t']+=1;t=st['t']
                G=p.grad;m=st['m']
                if p.ndim>=2:
                    G2=G.reshape(p.shape[0],-1);m2=m.reshape(p.shape[0],-1)
                    if t==1:
                        U=G2/torch.sqrt(torch.maximum(G2.square(),G2.square().mean())+eps2)
                    else:
                        B=m2/(1-beta**(t-1));R=st['r']/(1-rho**(t-1));C=st['c']/(1-rho**(t-1))
                        muR=R.mean()
                        N=R[:,None]*C[None,:]/muR.clamp_min(1e-30)
                        P=B/torch.sqrt(B.square()+N+eps2)
                        vR=B.square().mean(1)+R;vC=B.square().mean(0)+C;v0=vR.mean()
                        if self.mode=='nofloor':
                            K=torch.sqrt(0.5/(vR+eps2)[:,None]+0.5/(vC+eps2)[None,:])
                        elif self.mode=='p48gain':
                            K=torch.rsqrt(B.square()+N+eps2)
                        else:
                            K=torch.sqrt(0.5/(v0+eps2)+0.25/(vR+eps2)[:,None]+0.25/(vC+eps2)[None,:])
                        om=(1-beta)/(1-beta**t)
                        J=om*K*(G2-B)
                        U=P+J if self.mode=='noclip' else P+torch.clamp(J,-2.,2.)
                    E2=G2-m2
                    m.mul_(beta).add_(G,alpha=1-beta)
                    if lam>0:p.mul_(1-lr*lam)
                    p.add_(U.reshape(p.shape),alpha=-lr)
                    st['r'].mul_(rho).add_(E2.square().mean(1),alpha=(1-rho)*beta)
                    st['c'].mul_(rho).add_(E2.square().mean(0),alpha=(1-rho)*beta)
                else:
                    if t==1:
                        U=G/torch.sqrt(torch.maximum(G.square(),G.square().mean())+eps2)
                    else:
                        B=m/(1-beta**(t-1));R=st['n']/(1-rho**(t-1))
                        P=B/torch.sqrt(B.square()+R+eps2)
                        v=B.square()+R
                        if self.mode=='nofloor':K=torch.rsqrt(v+eps2)
                        elif self.mode=='p48gain':K=torch.rsqrt(B.square()+R+eps2)
                        else:K=torch.sqrt(0.5/(v.mean()+eps2)+0.5/(v+eps2))
                        om=(1-beta)/(1-beta**t)
                        J=om*K*(G-B)
                        U=P+J if self.mode=='noclip' else P+torch.clamp(J,-2.,2.)
                    E=G-m
                    m.mul_(beta).add_(G,alpha=1-beta)
                    if lam>0:p.mul_(1-lr*lam)
                    p.add_(U,alpha=-lr)
                    st['n'].mul_(rho).add_(E.square(),alpha=(1-rho)*beta)
        return loss
