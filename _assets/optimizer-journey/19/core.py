import os
os.environ['OMP_NUM_THREADS']='1'
import torch,numpy as np,json,csv,time
from pathlib import Path
try:
 import torch_musa
 device='musa' if torch.musa.is_available() else 'cpu'
except ImportError:device='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(1)
P=Path(__file__).parent;cfg=json.loads((P/'protocol.json').read_text())
def sync():
 if device=='musa':torch.musa.synchronize()
 elif device=='cuda':torch.cuda.synchronize()
cfg['device']=device;cfg['device_name']=torch.musa.get_device_name(0) if device=='musa' else (torch.cuda.get_device_name(0) if device=='cuda' else 'CPU');cfg['torch']=torch.__version__
(P/'executed_config.json').write_text(json.dumps(cfg,indent=2))
# Exact expectation, exogenous monotone drift. No Monte Carlo error.
oracle=[]
for r in cfg['r']:
 for p in cfg['p']:
  bias=var=mu=0.
  for t in range(1,16385):
   if t>1:mu+=(t-1)**(-r)
   a=max(2/(t+1),.1*t**(-p))
   bias=(1-a)*(bias-((t-1)**(-r) if t>1 else 0));var=(1-a)**2*var+a*a
   if t in [256,1024,4096,16384]:oracle.append(dict(r=r,p=p,step=t,bias2=bias*bias,variance=var,mse=bias*bias+var))
with (P/'oracle.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=oracle[0]);w.writeheader();w.writerows(oracle)
data=np.load(P/'data.npz');total=0
fields=['r','p','seed','step','train_loss','val_loss','test_loss','test_accuracy','tracking_mse','gradient_norm2','relative_tracking','seconds','diagnostic_seconds','train_samples','probe_samples','alpha','eta','batch_noise2','probe_gradient_drift2']
f=(P/'curves.csv').open('w',newline='');writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
xt=torch.tensor(data['x_train'],device=device);yt=torch.tensor(data['y_train'],device=device,dtype=torch.long)
xv=torch.tensor(data['x_val'],device=device);yv=torch.tensor(data['y_val'],device=device,dtype=torch.long)
xe=torch.tensor(data['x_test'],device=device);ye=torch.tensor(data['y_test'],device=device,dtype=torch.long)
N=len(yt)
for r in cfg['r']:
 for p in cfg['p']+['EMA']:
  for seed in cfg['seeds']:
   torch.manual_seed(seed);rng=np.random.default_rng(seed+100)
   model=torch.nn.Sequential(torch.nn.Linear(30,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).to(device)
   pars=list(model.parameters());m=[torch.zeros_like(z) for z in pars];v=[torch.zeros_like(z) for z in pars]
   seconds=diagnostic=0.;probes=0;last_fg=None
   for t in range(1,cfg['steps']+1):
    ids=torch.tensor(rng.integers(0,N,size=32),device=device);sync();start=time.perf_counter()
    loss=torch.nn.functional.cross_entropy(model(xt[ids]),yt[ids]);gs=torch.autograd.grad(loss,pars)
    a=(1. if t==1 else .1) if p=='EMA' else ((1. if t==1 else .1*t**(-p)) if cfg.get('schedule')=='pure' else max(2/(t+1),.1*t**(-p)));eta=cfg['lr']*t**(-r)
    with torch.no_grad():
     for mi,vi,g in zip(m,v,gs):mi.mul_(1-a).add_(g,alpha=a);vi.mul_(1-a).addcmul_(g,g,value=a)
    # Measure e_t BEFORE moving x: m_t and full gradient at exactly the same x_t.
    probe=t%32==0 or t==1
    sync();seconds+=time.perf_counter()-start
    if probe:
     sync();start=time.perf_counter();full_loss=torch.nn.functional.cross_entropy(model(xt),yt);fg=torch.autograd.grad(full_loss,pars)
     err=sum((mi-g).square().sum() for mi,g in zip(m,fg)).item();norm=sum(g.square().sum() for g in fg).item()
     noise=sum((g-h).square().sum() for g,h in zip(gs,fg)).item()
     drift=sum((g-h).square().sum() for g,h in zip(fg,last_fg)).item() if last_fg is not None else float('nan');last_fg=[g.detach().clone() for g in fg]
     with torch.no_grad():
      vl=torch.nn.functional.cross_entropy(model(xv),yv).item()
      # Sealed test: queried only at the predetermined terminal update.
      if t==cfg['steps']:
       z=model(xe);tl=torch.nn.functional.cross_entropy(z,ye).item();acc=(z.argmax(1)==ye).float().mean().item()
      else:tl=acc=float('nan')
     sync();diagnostic+=time.perf_counter()-start;probes+=N
    sync();start=time.perf_counter()
    with torch.no_grad():
     for z,mi,vi in zip(pars,m,v):z.addcdiv_(mi,vi.sqrt().add_(1e-8),value=-eta)
    sync();seconds+=time.perf_counter()-start
    if probe:writer.writerow(dict(r=r,p=p,seed=seed,step=t,train_loss=full_loss.item(),val_loss=vl,test_loss=tl,test_accuracy=acc,tracking_mse=err,gradient_norm2=norm,relative_tracking=err/(norm+1e-12),seconds=seconds,diagnostic_seconds=diagnostic,train_samples=t*32,probe_samples=probes,alpha=a,eta=eta,batch_noise2=noise,probe_gradient_drift2=drift));f.flush()
   total+=1;print('DONE',total,r,p,seed,flush=True)
f.close();(P/'DONE.json').write_text(json.dumps({'runs':total}));print('ALL_DONE',flush=True)
