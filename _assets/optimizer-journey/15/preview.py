import torch
class Preview(torch.optim.Optimizer):
 def __init__(self,params,lr=.01,name='A-mix'):
  super().__init__(params,dict(lr=lr,name=name))
 @torch.no_grad()
 def step(self):
  for group in self.param_groups:
   for p in group['params']:
    if p.grad is None:continue
    q=p.grad;z=self.state[p];name=group['name']
    if not z:z.update(t=0,m=torch.zeros_like(p),v=torch.zeros_like(p),u=torch.zeros_like(p),delta=torch.zeros_like(p))
    z['t']+=1;t=z['t'];m=z['m'];v=z['v']
    a=1024/(t+1023) if name=='H-mix' else 2/(t+1)
    if name in ['EMA-mix','Preview-only']:
     m.lerp_(q,.1);v.mul_(.999).addcmul_(q,q,value=.001)
     mm=m/(1-.9**t);vv=v/(1-.999**t)
    else:
     m.lerp_(q,a);v.mul_(1-a).addcmul_(q,q,value=a);mm=m;vv=v
    u=mm/(vv.sqrt()+1e-8)
    if name in ['A-mix-lever','A-mix-diff','A-mix-rawdiff']:
     u=u*(.5+.5*mm.square()/(vv+1e-16))
    if name=='A-mix-diff':d=group['lr']*(u-(1-1/t)*z['u'])
    elif name=='A-mix-rawdiff':d=group['lr']/t*(u-z['u'])
    else:d=group['lr']*u
    p.sub_(d);z['delta'].copy_(d);z['u'].copy_(u)
