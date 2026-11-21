import torch
class Preview(torch.optim.Optimizer):
 def __init__(self,params,lr=.01,name='AWA-constant'):
  super().__init__(params,dict(lr=lr,name=name))
 @torch.no_grad()
 def step(self):
  for group in self.param_groups:
   for p in group['params']:
    if p.grad is None:continue
    z=self.state[p];g=p.grad
    if not z:z.update(t=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
    z['t']+=1;a=2/(z['t']+1)
    if group['name'].startswith('Joint'):a=max(a,.1*z['t']**(-1/3))
    z['m'].lerp_(g,a);z['v'].mul_(1-a).addcmul_(g,g,value=a)
    p.addcdiv_(z['m'],z['v'].sqrt().add_(1e-8),value=-group['lr'])
