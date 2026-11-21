# Optimizer-only excerpt; see SOURCE_MANIFEST.json for original SHA-256.
import math

import numpy as np

import torch

base = run.base

OUT = Path('/root/autodl-tmp/layer37b')

def train(method,growth,lr,seed,phase):
 p=OUT/f'{phase}_{method}_{growth}_{lr}_{seed}.json'
 if p.exists():return json.loads(p.read_text())
 gc.collect();torch.cuda.empty_cache();e=base.Engine(seed);o=run.Optim(e,method,lr,.999)
 rng=np.random.default_rng(seed*1000003);positions=rng.integers(0,len(e.train)-base.SEQ-1,size=32000)
 used=t=0;tt=ot=0.;rec=[];next_eval=3200
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 while used<32000:
  t+=1;n=min(32000-used,math.ceil(32*t**.1) if growth else 32)
  e.g.zero_();torch.cuda.synchronize();start=time.perf_counter();loss_total=0.
  for k in range(0,n,32):
   count=min(32,n-k);x,y=e.batch(e.train,positions[used+k:used+k+count])
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=e.model(x,y)
   (loss*(count/n)).backward();loss_total+=loss.detach().item()*count/n
  torch.nn.utils.clip_grad_norm_(e.params,1.)
  # Common token-based practical schedule; not the constant-step theorem.
  frac=used/32000;eta=lr*min(1,(used+n)/640)*(.1+.9*.5*(1+math.cos(math.pi*frac)))
  torch.cuda.synchronize();mid=time.perf_counter();o.step(eta);torch.cuda.synchronize();end=time.perf_counter()
  used+=n;tt+=end-start;ot+=end-mid
  if used>=next_eval or used==32000:
   val=e.eval('val');rec.append(dict(step=t,tokens=used*128,val_loss=val,train_loss=loss_total,train_s=tt,opt_s=ot));next_eval+=3200
   print(phase,method,growth,lr,seed,used,val,flush=True)
   if not math.isfinite(val):break
 result=dict(method=method,growth=growth,lr=lr,seed=seed,phase=phase,records=rec,state_bytes=o.state_bytes(),peak_bytes=torch.cuda.max_memory_allocated(),test_loss=e.eval('test') if phase=='confirm' else None)
 p.write_text(json.dumps(result,indent=2));del e,o;gc.collect();return result
