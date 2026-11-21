# Optimizer-only excerpt; see SOURCE_MANIFEST.json for original SHA-256.
import math

import numpy as np

import torch

import triton

import triton.language as tl

@triton.jit
def step46(P,G,M,Q,COUNTS,LR,EPS,N:tl.constexpr,W:tl.constexpr):
 b=tl.program_id(0); k=tl.arange(0,W); i=b*W+k
 count=tl.load(COUNTS+b); valid=k<count
 old=tl.load(M+i,valid,0); g=tl.load(G+i,valid,0); p=tl.load(P+i,valid,0)
 pair=k ^ 1; other=tl.load(G+b*W+pair,(pair<count),0)-tl.load(M+b*W+pair,(pair<count),0)
 a=tl.abs(g-old); u=tl.where(pair<count,tl.minimum(a,2*tl.abs(other)),tl.minimum(a,EPS))
 q=tl.load(Q+b); m=.9*old+.1*g
 tl.store(P+i,p-LR*m*tl.rsqrt(1+q/EPS),valid); tl.store(M+i,m,valid)
 tl.store(Q+b,q+tl.sum(tl.where(valid,u,0),0)/count)
