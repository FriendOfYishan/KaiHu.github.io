"""Synthetic optimizer smoke run; does NOT reproduce historical training-table metrics."""
import torch
from core import C48
torch.manual_seed(51)
p=torch.nn.Parameter(torch.randn(5,7))
opt=C48([p],lr=0.001)
for t in range(32):
    p.grad=p.detach().clone()+torch.randn_like(p)*0.01
    opt.step()
    assert torch.isfinite(p).all()
print("32 finite synthetic updates:",type(opt).__name__,tuple(p.shape))
