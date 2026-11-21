"""Energy-absorption identity and fixed-snapshot diagnostic; no new optimizer."""
import ast
import csv
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).parent
PRIOR=OUT.parent/'20261110_alignment_comparison'
OUT.mkdir(parents=True,exist_ok=True)
protocol=dict(scope='Exact identities and conditional energy-descent certificates; original Joint-AWA unchanged.',
              snapshot_banks=['snapshots.json','range_refinement/snapshots.json'],
              exclude_steps=[1,2],random_states=5000,random_seed=261110,
              scalar_schedule_scan=[3,1000000],
              comparisons='Direct energy bound 23, absorbed bound with exact theta, conservative 1/3 bound; same potential, same noise correction.',
              cost='CPU diagnostics, not training speed evidence.')
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
def alpha(t):return max(2/(t+1),.1/t**(1/3))
def eta(t,e0=1.):return e0/math.sqrt(t)
def q(t,e0=1.):return eta(t,e0)*(1-alpha(t))/(2*alpha(t))
def coefficients(t,e0=1.):
    a=alpha(t);qt=q(t,e0);qp=q(t-1,e0);k=qp*math.sqrt(1-a)
    theta=eta(t,e0)+qt-qt*qt/k
    return qt,qp,k,theta
def leq(x,y):assert x<=y+2e-9*(1+abs(x)+abs(y)),(x,y)

start=time.perf_counter();rng=np.random.default_rng(261110)
errors=[]
for _ in range(5000):
    t=int(rng.integers(3,100000));a=alpha(t);e0=10**rng.uniform(-2,1)
    qt,qp,k,theta=coefficients(t,e0);et=eta(t,e0)
    b=rng.normal(size=7);v=b*b+rng.uniform(0,3,size=7);g=rng.normal(size=7)*10**rng.uniform(-2,1)
    eps=10**rng.uniform(-8,0)
    m=(1-a)*b+a*g;vn=(1-a)*v+a*g*g
    D=1/(np.sqrt(vn)+eps);Dp=1/(np.sqrt(v)+eps)
    M=float(np.sum(m*m*D));I=float(np.sum((m-b)**2*D));V=float(np.sum(b*b*(qt*D-qp*Dp)))
    R=float(k*np.sum((qt/k*m-b)**2*D))
    W=float(qp*np.sum(b*b*(Dp-math.sqrt(1-a)*D)))
    lhs=-et*M-qt*I+V;rhs=-theta*M-R-W
    err=abs(lhs-rhs)/(1+abs(lhs)+abs(rhs));errors.append(err)
    assert err<1e-9
    leq(0,W);leq(0,R);leq(et/3,theta);leq(lhs,-theta*M)

ts=np.arange(3,1000001,dtype=float)
aa=np.maximum(2/(ts+1),.1/np.cbrt(ts));ap=np.maximum(2/ts,.1/np.cbrt(ts-1))
ee=1/np.sqrt(ts);qq=ee*(1-aa)/(2*aa);qp=1/np.sqrt(ts-1)*(1-ap)/(2*ap)
ratios=(ee+qq-qq*qq/(qp*np.sqrt(1-aa)))/ee
linear=(ts+3)/4-(ts-1)**2/(4*(ts-2))*np.sqrt(1+1/ts)
assert ratios.min()>=1/3-1e-9
assert np.all(ratios>=linear-1e-7)

tree=ast.parse((PRIOR/'run.py').read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
env={'np':np,'itertools':itertools}
exec(compile(ast.Module(body=[node],type_ignores=[]),'prior_tasks','exec'),env)
Task=env['Task'];rows=[]
for bank in protocol['snapshot_banks']:
 for s in json.loads((PRIOR/bank).read_text(encoding='utf-8')):
    t=s['t']
    if t<3:continue
    task=Task(s['task'],s['noise']);x=np.array(s['x']);b=np.array(s['m_previous']);v=np.array(s['v_previous'])
    a=alpha(t);et=eta(t,s['eta0']);qt,qp,k,theta=coefficients(t,s['eta0']);eps=s['epsilon']
    gs=task.gradients(x);mu=gs.mean(0);nu=((gs-mu)**2).mean(0)
    m=(1-a)*b+a*gs;vn=(1-a)*v+a*gs*gs
    D=1/(np.sqrt(vn)+eps);Dp=1/(np.sqrt(v)+eps);u=D*m
    M=np.sum(m*m*D,axis=1);I=np.sum((m-b)**2*D,axis=1);U=np.sum(u*u,axis=1)
    V=np.sum(b*b*(qt*D-qp*Dp),axis=1)
    R=k*np.sum((qt/k*m-b)**2*D,axis=1)
    W=qp*np.sum(b*b*(Dp-np.sqrt(1-a)*D),axis=1)
    assert np.allclose(-et*M-qt*I+V,-theta*M-R-W,rtol=1e-7,atol=1e-8)
    H=a/(np.sqrt((1-a)*v)+eps)+np.sqrt(a)*(1-a)*np.abs(b)/(np.sqrt((1-a)*v)+eps)**2
    penalty=et*float(np.minimum(H*nu,np.sqrt(nu)).sum())
    actual=np.mean([task.F(x-et*ui)-task.F(x) for ui in u])+qt*M.mean()-qp*np.sum(b*b*Dp)
    curvature=task.L*et*et*U.mean()/2
    direct=float((-et*M-qt*I+V).mean()+curvature+penalty)
    absorbed=float(-theta*M.mean()+curvature+penalty)
    universal=float(-et/3*M.mean()+curvature+penalty)
    global_curvature=float(-(et/3-task.L*et*et/(2*eps))*M.mean()+penalty)
    for bound in [direct,absorbed,universal,global_curvature]:leq(actual,bound)
    leq(direct,absorbed);leq(absorbed,universal);leq(universal,global_curvature)
    rows.append(dict(bank=bank,task=s['task'],noise=s['noise'],seed=s['seed'],t=t,epsilon=eps,
                     eta0=s['eta0'],theta_ratio=theta/et,actual=float(actual),direct=direct,
                     absorbed=absorbed,universal=universal,global_curvature=global_curvature,
                     V=float(V.mean()),residual_R=float(R.mean()),residual_W=float(W.mean()),
                     noise_penalty=penalty,global_safe_step=et<=eps/(3*task.L)))
with (OUT/'snapshots.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(rr):
    return dict(n=len(rr),actual_energy_descent=sum(r['actual']< -1e-12 for r in rr),
                positive_V=sum(r['V']>1e-12 for r in rr),global_safe_steps=sum(r['global_safe_step'] for r in rr),
                certificates={key:dict(certified=sum(r[key]< -1e-12 for r in rr),
                    false_certificates=sum(r[key]< -1e-12 and r['actual']>=-1e-12 for r in rr),
                    mean_slack=float(np.mean([r[key]-r['actual'] for r in rr])))
                    for key in ['direct','absorbed','universal','global_curvature']})
res=dict(protocol=protocol,random_identity_max_relative_error=max(errors),
         schedule_min=dict(t=int(ts[ratios.argmin()]),theta_ratio=float(ratios.min())),
         schedule_examples=[dict(t=t,joint=coefficients(t)[3]/eta(t),
            linear=(t+3)/4-(t-1)**2/(4*(t-2))*math.sqrt(1+1/t)) for t in [3,4,8,32,128,512,1024,1000000]],
         all=summarize(rows),noiseless=summarize([r for r in rows if r['noise']==0]),
         noisy=summarize([r for r in rows if r['noise']!=0]),elapsed_seconds=time.perf_counter()-start)
(OUT/'results.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(11,4.3))
idx=np.unique(np.geomspace(1,len(ts),500).astype(int)-1)
axs[0].plot(ts[idx],ratios[idx],label='Joint-AWA');axs[0].plot(ts[idx],linear[idx],label='Linear AWA')
axs[0].axhline(1/3,color='gray',ls='--',label='Proved lower bound: 1/3')
axs[0].set(xscale='log',xlabel='Step',ylabel='Effective energy coefficient / learning rate');axs[0].legend(fontsize=8)
keys=['direct','absorbed','universal','global_curvature']
axs[1].bar(range(4),[res['all']['certificates'][k]['certified'] for k in keys])
axs[1].axhline(res['all']['actual_energy_descent'],color='gray',ls='--',label='Actual expected energy descent')
axs[1].set_xticks(range(4),['Direct 23','Absorbed','1/3 bound','Global curvature'],rotation=15)
axs[1].set(ylabel='Certified energy-descent snapshots');axs[1].legend(fontsize=8)
for ax in axs:ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'energy_absorption.png',dpi=180)
print(json.dumps(res,indent=2))
