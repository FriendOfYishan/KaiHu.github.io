import math
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(P,G,M,V,D,N:tl.constexpr,A,B,LR,C1,C2,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);mask=i<N
    p=tl.load(P+i,mask,0);g=tl.load(G+i,mask,0)
    m=tl.load(M+i,mask,0);v=tl.load(V+i,mask,0)
    m=(1.-A)*m+A*g;v=(1.-B)*v+B*g*g
    decay=tl.load(D+i,mask,0).to(tl.float32)*.1
    p=p*(1.-LR*decay)-LR*(m/C1)/(tl.sqrt(v/C2)+1.e-8)
    tl.store(P+i,p,mask);tl.store(M+i,m,mask);tl.store(V+i,v,mask)

class FlatOptimizer:
    def __init__(self,model,method,eta,horizon):
        self.params=list(model.parameters());self.method=method;self.eta=eta;self.horizon=horizon;self.t=0
        self.p=torch.cat([p.detach().flatten() for p in self.params]);self.g=torch.zeros_like(self.p)
        self.m=torch.zeros_like(self.p);self.v=torch.zeros_like(self.p);self.decay=torch.empty_like(self.p,dtype=torch.uint8)
        offset=0
        for p in self.params:
            n=p.numel();p.data=self.p[offset:offset+n].view_as(p);p.grad=self.g[offset:offset+n].view_as(p)
            self.decay[offset:offset+n].fill_(int(p.ndim>=2));offset+=n
    def zero_grad(self):self.g.zero_()
    def lr(self,t):
        warm=math.ceil(.02*self.horizon)
        if t<=warm:return self.eta*t/warm
        return self.eta*(.1+.45*(1+math.cos(math.pi*(t-warm)/(self.horizon-warm))))
    def coefficients(self,t):
        method='AA' if self.method=='J' else self.method
        w=max(2/(t+1),.1*t**(-1/3))
        return (w if method[0]=='A' else .1,w if method[1]=='A' else .001,
                1. if method[0]=='A' else 1-.9**t,1. if method[1]=='A' else 1-.999**t)
    @torch.no_grad()
    def step(self):
        self.t+=1
        torch.nn.utils.clip_grad_norm_(self.params,1.,foreach=True)
        a,b,c1,c2=self.coefficients(self.t)
        kernel[(triton.cdiv(self.p.numel(),1024),)](self.p,self.g,self.m,self.v,self.decay,self.p.numel(),a,b,self.lr(self.t),c1,c2,BLOCK=1024,enable_fp_fusion=False)
    def state_dict(self):return dict(t=self.t,m=self.m,v=self.v,method=self.method,eta=self.eta,horizon=self.horizon)
    def load_state_dict(self,s):
        assert (s['method'],s['eta'],s['horizon'])==(self.method,self.eta,self.horizon)
        self.t=s['t'];self.m.copy_(s['m']);self.v.copy_(s['v'])

def check():
    out={}
    for method in ['EE','AE','EA','AA']:
        torch.manual_seed(981);model=torch.nn.Linear(64,32).cuda();opt=FlatOptimizer(model,method,.001,4096)
        p=opt.p.double().clone();m=torch.zeros_like(p);v=torch.zeros_like(p);worst=0.
        for t in range(1,257):
            g=torch.randn_like(opt.g)*.1;opt.g.copy_(g);gd=g.double();gd*=min(1.,1/(gd.norm().item()+1e-6))
            a,b,c1,c2=opt.coefficients(t);m=(1-a)*m+a*gd;v=(1-b)*v+b*gd.square()
            p=p*(1-opt.lr(t)*.1*opt.decay.double())-opt.lr(t)*(m/c1)/((v/c2).sqrt()+1e-8)
            opt.step();worst=max(worst,(p-opt.p.double()).abs().max().item())
        assert worst<3e-6,(method,worst)
        out[method]=dict(max_parameter_error_fp64=worst,steps=256)
    return out
