"""CUDA-only synthetic packed-optimizer entry; no historical training-table reproduction."""
import torch
from lab45 import Packed
model=torch.nn.Sequential(torch.nn.Linear(7,5,bias=False),torch.nn.LayerNorm(5)).cuda()
packed=Packed(model,'lm')
from core import Optim46
opt=Optim46(packed,'46a',.001,0.)
for t in range(8):
    packed.g.copy_(torch.randn_like(packed.g)*.01)
    opt.step(.001)
torch.cuda.synchronize()
assert torch.isfinite(packed.p).all()
print("8 finite packed GPU updates")
