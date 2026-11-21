# Optimizer-only excerpt; see SOURCE_MANIFEST.json for original SHA-256.
import math

import torch

from torch import nn

from torch.nn import functional as F

B1 = 0.9

BLOCK = 1024

GROUP = 8

class C47(torch.optim.Optimizer):
    def __init__(self,params,lr=0.1,beta=B1,tau=1e-3):
        super().__init__(params,dict(lr=lr,beta=beta,tau=tau))
        for grp in self.param_groups:
            for p in grp['params']:
                st=self.state[p];st['m']=torch.zeros_like(p);st['t']=0
                if p.ndim>=2:
                    nrows=p.shape[0];ncols=p.numel()//nrows
                else:
                    nrows,ncols=1,p.numel()
                nb=(ncols+BLOCK-1)//BLOCK;pad=nb*BLOCK-ncols
                st['meta']=(nrows,ncols,nb,pad)
                dev=p.device
                mask=torch.zeros(nb*BLOCK,device=dev)
                gok=torch.zeros(nb*BLOCK,device=dev)
                bsz=torch.zeros(nb,device=dev)
                for b in range(nb):
                    L=min(BLOCK,ncols-b*BLOCK);bsz[b]=L
                    mask[b*BLOCK:b*BLOCK+L]=1.
                    full=(L//GROUP)*GROUP
                    gok[b*BLOCK:b*BLOCK+full]=1.
                st['mask']=mask
                st['gok']=gok.view(1,nb,BLOCK//GROUP,GROUP)
                st['gden']=st['gok'].sum(-1,keepdim=True).clamp_min(1.)
                st['q']=torch.zeros(nrows,nb,device=dev)
                st['mask2']=mask.view(1,nb,BLOCK)
                st['bnorm']=(1./bsz).view(1,nb)
    @torch.no_grad()
    def step(self,closure=None):
        loss=closure() if closure is not None else None
        for grp in self.param_groups:
            lr=grp['lr'];beta=grp['beta'];tau=grp['tau']
            for p in grp['params']:
                if p.grad is None:continue
                st=self.state[p];st['t']+=1
                nrows,ncols,nb,pad=st['meta']
                g=p.grad.reshape(nrows,ncols)
                m=st['m'].reshape(nrows,ncols)
                gp=F.pad(g,(0,pad)) if pad else g
                if p.ndim>=2:
                    h=gp.view(nrows,nb,BLOCK//GROUP,GROUP)
                    mu=(h*st['gok']).sum(-1,keepdim=True)/st['gden']
                    u=(h-mu*st['gok']).reshape(nrows,nb*BLOCK)[:,:ncols]
                else:
                    u=gp[:,:ncols]
                m_old=m.clone()
                m.mul_(beta).add_(u,alpha=1-beta)
                r=(u-m_old).abs()
                rp=F.pad(r,(0,pad)) if pad else r
                over=(rp>tau).float()
                cnt=(over.view(nrows,nb,BLOCK)*st['mask2']).sum(-1)*st['bnorm']
                st['q']+=cnt
                hb=torch.rsqrt(1.+st['q'])
                hbar=torch.repeat_interleave(hb,BLOCK,dim=1)[:,:ncols].reshape(p.shape)
                p.addcmul_(hbar,m.reshape(p.shape),value=-lr)
        return loss
