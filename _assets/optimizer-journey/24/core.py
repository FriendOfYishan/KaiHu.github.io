"""Paired exact conditional comparisons on reachable Joint-AWA states."""
import csv
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--range-refinement',action='store_true')
args=parser.parse_args()
OUT=Path(__file__).parent/('range_refinement' if args.range_refinement else '')
SNAPS=[1,8,32,128,512]
CONFIG=dict(tasks=['quadratic','nonconvex','logistic'],noise_levels=[0,.1,1,3],
            eta0=[.1,.5],epsilon=[1e-3,1e-8],seeds=list(range(8)),
            snapshots=SNAPS,steps=512,dimension=2,
            development_seeds=list(range(4)),confirmation_seeds=list(range(4,8)),
            selection='No tuning or algorithm changes; all settings reported.',
            oracle='Four equiprobable outcomes; exact conditional enumeration. Logistic uses four fixed minibatches; other tasks use independent Rademacher noise.',
            timing='CPU diagnostic experiment, not GPU optimizer throughput.')
if args.range_refinement:
 CONFIG.update(seeds=list(range(8,16)),development_seeds=list(range(8,12)),
               confirmation_seeds=list(range(12,16)),
               refinement='Previously proved range bound min(H*variance,sqrt(variance)); fresh seeds, unchanged task/grid.')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'protocol.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')

def alpha(t): return max(2/(t+1),.1/t**(1/3))
def q(t,eta0): return 0. if t==0 else eta0/np.sqrt(t)*(1-alpha(t))/(2*alpha(t))

class Task:
    def __init__(self,name,noise):
        self.name,self.noise=name,noise
        self.signs=np.array(list(itertools.product([-1.,1.],repeat=2)))
        if name=='logistic':
            rng=np.random.default_rng(9041)
            self.X=rng.normal(size=(32,2))
            self.y=(self.X@np.array([1.,-.6])+rng.normal(size=32)*.5>0).astype(float)
            self.L=float(np.linalg.eigvalsh(self.X.T@self.X/32).max()/4+.05)
        else: self.L=4. if name=='quadratic' else 1.1
    def F(self,x):
        if self.name=='quadratic': return float(.5*np.sum(np.array([1.,4.])*x*x))
        if self.name=='nonconvex': return float(np.sum(.05*x*x+1-np.cos(x)))
        z=self.X@x
        return float(np.mean(np.logaddexp(0,z)-self.y*z)+.025*np.sum(x*x))
    def gradients(self,x):
        if self.name=='quadratic': mu=np.array([1.,4.])*x
        elif self.name=='nonconvex': mu=.1*x+np.sin(x)
        else:
            z=np.clip(self.X@x,-700,700)
            r=1/(1+np.exp(-z))-self.y
            return np.array([self.X[k:k+8].T@r[k:k+8]/8+.05*x for k in range(0,32,8)])
        return mu+self.noise*self.signs

rows=[];bank=[]; start=time.perf_counter()
for name in CONFIG['tasks']:
 for noise in ([None] if name=='logistic' else CONFIG['noise_levels']):
  task=Task(name,noise)
  for eta0 in CONFIG['eta0']:
   for eps in CONFIG['epsilon']:
    for seed in CONFIG['seeds']:
     rng=np.random.default_rng(seed+12001)
     x=np.array([1.5,-1.])+rng.normal(size=2)*.2
     b=np.zeros(2);v=np.zeros(2)
     for t in range(1,513):
      a=alpha(t);eta=eta0/np.sqrt(t)
      gs=task.gradients(x);mu=gs.mean(axis=0);nu=np.mean((gs-mu)**2,axis=0)
      if t in SNAPS:
       ms=(1-a)*b+a*gs;vs=(1-a)*v+a*gs*gs
       Ds=1/(np.sqrt(vs)+eps);us=Ds*ms;es=ms-mu
       M=np.sum(ms*ms*Ds,axis=1);U=np.sum(us*us,axis=1)
       A=np.sum(es*us,axis=1);J=np.sum((ms-gs)*us,axis=1)
       E2=np.sum(es*es,axis=1).mean();UU=U.mean()
       H=a/(np.sqrt((1-a)*v)+eps)+np.sqrt(a)*(1-a)*np.abs(b)/(np.sqrt((1-a)*v)+eps)**2
       bounds=dict(young22=.5/eps*E2+.5*eps*UU,
                   cauchy_oracle=np.sqrt(max(0,E2*UU)),
                   sensitivity_known=float(J.mean()+np.sum(H*nu)),
                   sensitivity_innovation=float(J.mean()+np.sum(H*np.mean((gs-b)**2,axis=0))))
       if args.range_refinement:
        bounds['sensitivity_range']=float(J.mean()+np.sum(np.minimum(H*nu,np.sqrt(nu))))
       actualF=np.mean([task.F(x-eta*u)-task.F(x) for u in us])
       scale=np.sqrt(max(0,E2*UU))+1e-12
       Dprev=1/(np.sqrt(v)+eps);qt=q(t,eta0);qp=q(t-1,eta0)
       I=np.sum((ms-b)**2*Ds,axis=1)
       V=np.sum(b*b*(qt*Ds-qp*Dprev),axis=1)
       psi_delta=np.mean([task.F(x-eta*u)-task.F(x) for u in us])+qt*M.mean()-qp*np.sum(b*b*Dprev)
       # Old alignment bound transferred to the SAME energy potential for fair comparison.
       energy_offset=qt*M.mean()-qp*np.sum(b*b*Dprev)
       for method,B in bounds.items():
        Fbound=-eta*M.mean()+eta*B+task.L*eta*eta*UU/2
        Psibound=Fbound+energy_offset
        assert B>=A.mean()-1e-8*(1+abs(B)+abs(A.mean()))
        assert Fbound>=actualF-1e-8*(1+abs(Fbound)+abs(actualF))
        if method=='sensitivity_known':
         direct=(-eta*M-qt*I+task.L*eta*eta*U/2+V).mean()+eta*np.sum(H*nu)
         assert np.isclose(direct,Psibound,rtol=1e-7,atol=1e-9)
        rows.append(dict(task=name,noise=noise,eta0=eta0,epsilon=eps,seed=seed,t=t,
                         split='development' if seed in CONFIG['development_seeds'] else 'confirmation',method=method,
                         true_alignment=float(A.mean()),bound=float(B),slack=float(B-A.mean()),
                         normalized_slack=float((B-A.mean())/scale),
                         true_delta_F=float(actualF),F_bound=float(Fbound),
                         true_delta_psi=float(psi_delta),psi_bound=float(Psibound),
                         V=float(V.mean()),noise_penalty=float(np.sum(H*nu)),
                         actual_descent=bool(actualF< -1e-12),certified_descent=bool(Fbound< -1e-12)))
       bank.append(dict(task=name,noise=noise,eta0=eta0,epsilon=eps,seed=seed,t=t,
                        x=x.tolist(),m_previous=b.tolist(),v_previous=v.tolist()))
      g=gs[rng.integers(4)]
      b=(1-a)*b+a*g;v=(1-a)*v+a*g*g
      x=x-eta*b/(np.sqrt(v)+eps)

with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'snapshots.json').write_text(json.dumps(bank),encoding='utf-8')

def summary(sub):
 result={}
 methods=list(dict.fromkeys(r['method'] for r in sub))
 by={m:[r for r in sub if r['method']==m] for m in methods}
 for method,rr in by.items():
  result[method]=dict(n=len(rr),median_normalized_slack=float(np.median([r['normalized_slack'] for r in rr])),
        mean_slack=float(np.mean([r['slack'] for r in rr])),
        actual_descent_count=sum(r['actual_descent'] for r in rr),
        certified_descent_count=sum(r['certified_descent'] for r in rr),
        false_descent_certificates=sum(r['certified_descent'] and not r['actual_descent'] for r in rr))
 for method in [m for m in methods if m.startswith('sensitivity')]:
  for ref in ['young22','cauchy_oracle']:
   pairs=list(zip(by[method],by[ref]))
   result[method]['tighter_than_'+ref]=sum(a['bound']<b['bound']-1e-10 for a,b in pairs)/len(pairs)
 return result

res=dict(protocol=CONFIG,snapshots=len(bank),elapsed_seconds=time.perf_counter()-start,
         all=summary(rows),confirmation=summary([r for r in rows if r['split']=='confirmation']),
         by_task={name:summary([r for r in rows if r['task']==name]) for name in CONFIG['tasks']},
         by_step={str(t):summary([r for r in rows if r['t']==t]) for t in SNAPS},
         by_epsilon={str(e):summary([r for r in rows if r['epsilon']==e]) for e in CONFIG['epsilon']})
(OUT/'results.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
colors=['gray','orange','blue','green','purple']
for (method,stats),color in zip(res['all'].items(),colors):
 med=[res['by_step'][str(t)][method]['median_normalized_slack'] for t in SNAPS]
 axs[0].plot(SNAPS,np.maximum(med,1e-12),marker='o',label=method,color=color)
 counts=[res['by_step'][str(t)][method]['certified_descent_count']/res['by_step'][str(t)][method]['actual_descent_count'] for t in SNAPS]
 axs[1].plot(SNAPS,counts,marker='o',label=method,color=color)
axs[0].set(xscale='log',yscale='log',xlabel='Joint-AWA step',ylabel='Median alignment-bound slack / Cauchy scale')
axs[1].set(xscale='log',xlabel='Joint-AWA step',ylabel='Certified / actual expected descent')
for ax in axs: ax.grid(alpha=.25)
axs[0].legend(fontsize=8)
fig.suptitle('Exact conditional evaluation on 1,440 reachable frozen states')
fig.tight_layout();fig.savefig(OUT/'bound_comparison.png',dpi=180)
print(json.dumps(dict(snapshots=res['snapshots'],elapsed_seconds=res['elapsed_seconds'],all=res['all'],confirmation=res['confirmation']),indent=2))
