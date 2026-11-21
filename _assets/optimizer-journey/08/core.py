"""Exploratory coupled shrinkage/lookahead; no changes to Adam's denominator.

One closure/backward per step. Gradients are sampled at y, but the final step
and decoupled decay start from x. Local quadratic theory is in theory/08.
"""
import torch
from torch.optim import Optimizer


class PredictiveShrinkAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(.9,.999), eps=1e-8,
                 weight_decay=.01, shrink=.5, predictive=None):
        if not 0<=shrink<1: raise ValueError("shrink must be in [0,1)")
        if predictive is None: predictive=shrink
        if predictive<0: raise ValueError("predictive must be nonnegative")
        if lr<0 or eps<0 or weight_decay<0 or not all(0<=b<1 for b in betas):
            raise ValueError("invalid Adam parameters")
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
                                     shrink=shrink,predictive=predictive))

    @torch.no_grad()
    def step(self,closure):
        if closure is None: raise ValueError("A closure computing ONE fresh gradient is required")
        originals=[]
        try:
            for group in self.param_groups:
                p1,p2=group['betas']
                for p in group['params']:
                    if not p.requires_grad: continue
                    st=self.state[p]
                    if not st:
                        dtype=torch.float64 if p.dtype==torch.float64 else torch.float32
                        st.update(step=0,m=torch.zeros_like(p,dtype=dtype),v=torch.zeros_like(p,dtype=dtype))
                    t=st['step']+1
                    if t>1 and group['predictive']>0:
                        originals.append((p,p.detach().clone()))
                        h=st['m']/(1-p1**(t-1))
                        den=(st['v']/(1-p2**(t-1))).sqrt()+group['eps']
                        a=(1-p1)/(1-p1**t)
                        shift=group['lr']*group['predictive']/a*h/den
                        p.sub_(shift.to(p.dtype))
            with torch.enable_grad(): loss=closure()
        finally:
            for p,original in originals: p.copy_(original)
        for group in self.param_groups:
            p1,p2=group['betas']
            for p in group['params']:
                if p.grad is None: continue
                if p.grad.is_sparse or p.is_complex(): raise RuntimeError("dense real gradients required")
                st=self.state[p];t=st['step']+1
                g=p.grad.to(st['m'].dtype)
                h=st['m']/(1-p1**(t-1)) if t>1 else torch.zeros_like(g)
                q=(1-group['shrink'])*g.square()+group['shrink']*h.square()
                st['m'].mul_(p1).add_(g,alpha=1-p1)
                st['v'].mul_(p2).add_(q,alpha=1-p2)
                p.mul_(1-group['lr']*group['weight_decay'])
                step=(st['m']/(1-p1**t))/((st['v']/(1-p2**t)).sqrt()+group['eps'])
                p.add_(step.to(p.dtype),alpha=-group['lr'])
                st['step']=t
        return loss
