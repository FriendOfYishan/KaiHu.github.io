"""CUDA-only synthetic kernel entry for pair-clipped 46."""
import torch
from core import step46
p=torch.zeros(1024,device='cuda');g=torch.randn_like(p)*.01;m=torch.zeros_like(p);q=torch.zeros(1,device='cuda');counts=torch.tensor([1024],device='cuda',dtype=torch.int32)
for t in range(8):step46[(1,)](p,g,m,q,counts,.01,.1,N=1024,W=1024)
torch.cuda.synchronize();assert torch.isfinite(p).all()
print("8 finite kernel updates",float(q[0]))
