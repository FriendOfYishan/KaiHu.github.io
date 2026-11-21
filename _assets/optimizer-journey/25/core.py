"""Offline identification audit. Joint-AWA updates are unchanged."""
import ast
import csv
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
PROTOCOL=dict(
    previous_snapshot_banks=['snapshots.json','range_refinement/snapshots.json'],
    development_tasks=['quadratic','nonconvex','logistic'],
    new_tasks=['rotated_quadratic8','coupled_nonconvex8','huber_regression8'],
    development_seeds=list(range(200,208)),new_seeds=list(range(300,308)),
    eta0=[.1,.5],epsilon=.001,additive_noise=[.1,1.,3.],steps=1024,
    record='t>=33 and t%16==0; all features end at t-1, before the current gradient',
    label='historical conditional bias squared > current oracle noise variance trace',
    warning='This is bias dominance, not proof that bias is caused by deterministic drift.',
    features=['vector_corr','projection1_corr','projection4_corr','energy_baseline','time_baseline'],
    projection_seed=1705,smoothing='existing alpha_t, no fitted smoothing coefficient',
    threshold='Choose balanced-accuracy threshold using development tasks only; correlation orientation fixed positive; baseline orientation selected on development.',
    deployment_gate='Each new task: projection1 AUC>=.75, balanced accuracy>=.65, and AUC at least .05 above the best scalar baseline; each class at least 20 cases. Even passing requires cost review.',
    scope='CPU exact-oracle offline diagnostics. Previous innovation vectors are stored only by diagnostics, not added to the optimizer.')
(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')

# Reuse exactly the old task definitions, without executing their experiment.
tree=ast.parse((PRIOR/'run.py').read_text(encoding='utf-8'))
task_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
import itertools
env={'np':np,'itertools':itertools}
exec(compile(ast.Module(body=[task_node],type_ignores=[]),'prior_task_definition','exec'),env)
OldTask=env['Task']
def alpha(t): return max(2/(t+1),.1/t**(1/3))
def writecsv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def audit_snapshots():
    rows=[]
    for bankname in PROTOCOL['previous_snapshot_banks']:
      for item in json.loads((PRIOR/bankname).read_text(encoding='utf-8')):
        task=OldTask(item['task'],item['noise'])
        x=np.array(item['x']);b=np.array(item['m_previous']);v=np.array(item['v_previous'])
        a=alpha(item['t']);eps=item['epsilon']
        gs=task.gradients(x);mu=gs.mean(0);nu=((gs-mu)**2).mean(0)
        h=b-mu;B=float(h@h);N=float(nu.sum());sigma=math.sqrt(N)
        ms=(1-a)*b+a*gs;us=ms/(np.sqrt((1-a)*v+a*gs*gs)+eps)
        H=a/(np.sqrt((1-a)*v)+eps)+np.sqrt(a)*(1-a)*np.abs(b)/(np.sqrt((1-a)*v)+eps)**2
        J=float(np.sum((ms-gs)*us,axis=1).mean())
        E2=float(np.sum((ms-mu)**2,axis=1).mean());U=float(np.sum(us*us,axis=1).mean())
        BC=math.sqrt(max(0,E2*U));BH=J+float(H@nu);BI=J+float(H@(nu+h*h))
        BR=J+float(np.minimum(H*nu,np.sqrt(nu)).sum())
        u0=((1-a)*b+a*mu)/(np.sqrt((1-a)*v+a*mu*mu)+eps)
        Hmax=float(H.max());z=float(np.linalg.norm(u0));bias=math.sqrt(B)
        lower=math.sqrt((1-a)**2*B+a*a*N)*max(0,z-Hmax*sigma)
        upper=(1-a)*bias*(z+Hmax*sigma)+(2-a)*Hmax*N+Hmax*B
        margin=lower-upper
        assert BC-BI>=margin-1e-8*(1+abs(BC)+abs(BI)+abs(margin))
        share=B/(B+N+1e-30)
        snr=sigma/(float(np.linalg.norm(mu))+1e-30)
        rows.append(dict(bank=bankname,task=item['task'],noise=item['noise'],seed=item['seed'],
                         t=item['t'],eta0=item['eta0'],epsilon=eps,bias_squared=B,noise_variance=N,
                         bias_share=share,noise_signal_ratio=snr,matched_history=bool(B<1e-20),
                         generalized_margin=margin,sufficient=bool(margin>1e-10),
                         H_tighter=bool(BH<BC-1e-10),I_tighter=bool(BI<BC-1e-10),
                         R_tighter=bool(BR<BC-1e-10)))
    writecsv(OUT/'snapshot_conditions.csv',rows)
    bins=[]
    for lo,hi in zip([0,.01,.1,.5,.9],[.01,.1,.5,.9,1.000001]):
        rr=[r for r in rows if lo<=r['bias_share']<hi and r['noise_variance']>1e-20]
        bins.append(dict(bias_share_interval=[lo,hi],n=len(rr),**{
            k:sum(r[k] for r in rr)/len(rr) if rr else None for k in ['H_tighter','I_tighter','R_tighter']}))
    return dict(n=len(rows),matched_history=sum(r['matched_history'] for r in rows),
                generalized_sufficient=sum(r['sufficient'] for r in rows),
                sufficient_failures=sum(r['sufficient'] and not r['I_tighter'] for r in rows),
                noisy_bias_bins=bins)

class NewTask:
    def __init__(self,name,noise):
        self.name,self.noise=name,noise
        self.dim=8
        rng=np.random.default_rng(5501)
        Z=rng.normal(size=(4,8));Z-=Z.mean(0)
        self.Z=Z/np.sqrt(np.mean(Z*Z))
        Q=np.linalg.qr(rng.normal(size=(8,8)))[0]
        self.Q=Q@np.diag(np.geomspace(1,20,8))@Q.T
        self.C=rng.normal(size=(8,8))/np.sqrt(8)
        self.X=rng.normal(size=(64,8))
        truth=rng.normal(size=8)
        self.y=self.X@truth+rng.normal(size=64)*.2
        self.y[::8]+=5.0
    def gradients(self,x):
        if self.name=='rotated_quadratic8': mu=self.Q@x
        elif self.name=='coupled_nonconvex8': mu=.05*x+np.sin(x)+.2*self.C.T@(self.C@x)
        else:
            residual=self.X@x-self.y
            psi=np.clip(residual,-1,1)
            return np.array([self.X[k:k+16].T@psi[k:k+16]/16+.05*x for k in range(0,64,16)])
        return mu+self.noise*self.Z

def generate():
    rows=[]
    for split,names,seeds in [('development',PROTOCOL['development_tasks'],PROTOCOL['development_seeds']),
                              ('new_task',PROTOCOL['new_tasks'],PROTOCOL['new_seeds'])]:
      for name in names:
       finite=name in ('logistic','huber_regression8')
       for noise in ([None] if finite else PROTOCOL['additive_noise']):
        task=OldTask(name,noise) if split=='development' else NewTask(name,noise)
        dim=2 if split=='development' else 8
        proj=np.random.default_rng(1705).choice([-1.,1.],size=(4,dim))/math.sqrt(dim)
        for eta0 in PROTOCOL['eta0']:
         for seed in seeds:
          rng=np.random.default_rng(seed)
          x=np.resize(np.array([1.5,-1.]),dim)+rng.normal(size=dim)*.2
          m=np.zeros(dim);v=np.zeros(dim);prev=None;mu_prev=None
          cross=np.zeros(3);energy=np.zeros(3);innovation_energy=0.
          for t in range(1,PROTOCOL['steps']+1):
           a=alpha(t);eta=eta0/math.sqrt(t)
           gs=task.gradients(x);mu=gs.mean(0);nu=((gs-mu)**2).mean(0)
           B=float(np.sum((mu-m)**2));N=float(nu.sum())
           if t>=33 and t%16==0:
            scores=cross/(energy+1e-30)
            movement=0. if mu_prev is None else float(np.sum((mu-mu_prev)**2))
            rows.append(dict(split=split,task=name,noise=noise,seed=seed,eta0=eta0,t=t,
                             label=int(B>N),bias_share=B/(B+N+1e-30),bias_squared=B,noise_variance=N,
                             gradient_motion_squared=movement,
                             vector_corr=float(scores[0]),projection1_corr=float(scores[1]),
                             projection4_corr=float(scores[2]),
                             energy_baseline=float(np.log((innovation_energy+1e-30)/(v.sum()+1e-30))),
                             time_baseline=-math.log(t)))
           g=gs[rng.integers(len(gs))];innov=g-m
           if prev is not None:
            p=proj@innov;pp=proj@prev
            cc=np.array([innov@prev,p[0]*pp[0],np.mean(p*pp)])
            ee=np.array([.5*(innov@innov+prev@prev),.5*(p[0]**2+pp[0]**2),.5*np.mean(p*p+pp*pp)])
            cross=(1-a)*cross+a*cc;energy=(1-a)*energy+a*ee
           innovation_energy=(1-a)*innovation_energy+a*float(innov@innov)
           m=(1-a)*m+a*g;v=(1-a)*v+a*g*g
           x=x-eta*m/(np.sqrt(v)+PROTOCOL['epsilon'])
           prev=innov;mu_prev=mu
    writecsv(OUT/'identification.csv',rows)
    return rows

def auc(y,s):
    y=np.asarray(y,dtype=bool);s=np.asarray(s)
    n1=y.sum();n0=len(y)-n1
    if n1==0 or n0==0:return None
    order=np.argsort(s,kind='stable');sorted_s=s[order];ranks=np.empty(len(s))
    lo=0
    while lo<len(s):
        hi=lo+1
        while hi<len(s) and sorted_s[hi]==sorted_s[lo]: hi+=1
        ranks[order[lo:hi]]=(lo+1+hi)/2
        lo=hi
    return float((ranks[y].sum()-n1*(n1+1)/2)/(n1*n0))

def metrics(rr,feat,orientation=1,threshold=0.):
    y=np.array([r['label'] for r in rr],dtype=bool);s=np.array([r[feat] for r in rr])*orientation
    pred=s>threshold;n1=int(y.sum());n0=len(y)-n1
    recall=float(np.mean(pred[y])) if n1 else None
    specificity=float(np.mean(~pred[~y])) if n0 else None
    return dict(n=len(y),positive=n1,negative=n0,auc=auc(y,s),
                balanced_accuracy=(recall+specificity)/2 if n1 and n0 else None,
                precision=float(np.mean(y[pred])) if pred.any() else None,
                recall=recall,specificity=specificity)

def evaluate(rows):
    dev=[r for r in rows if r['split']=='development']
    chosen={}
    for feat in PROTOCOL['features']:
        best=None
        for orient in ([1,-1] if feat.endswith('baseline') else [1]):
            scores=np.array([r[feat]*orient for r in dev])
            for thresh in np.unique(np.quantile(scores,np.linspace(0,1,101))):
                met=metrics(dev,feat,orient,float(thresh));value=met['balanced_accuracy']
                if best is None or value>best['development']['balanced_accuracy']:
                    best=dict(orientation=orient,threshold=float(thresh),development=met)
        chosen[feat]=best
    result=dict(selection=chosen,new_tasks={})
    for name in PROTOCOL['new_tasks']:
        rr=[r for r in rows if r['task']==name]
        result['new_tasks'][name]={feat:metrics(rr,feat,c['orientation'],c['threshold']) for feat,c in chosen.items()}
    gates={}
    for name,methods in result['new_tasks'].items():
        p=methods['projection1_corr'];baseline=max(methods[b]['auc'] for b in ['energy_baseline','time_baseline'])
        gates[name]=bool(p['positive']>=20 and p['negative']>=20 and p['auc']>=.75 and p['balanced_accuracy']>=.65 and p['auc']>=baseline+.05)
    result['gate_by_task']=gates;result['gate_passed']=all(gates.values())
    # Per-task/seed summaries retain the dependence structure for later audit.
    seedrows=[]
    for name in PROTOCOL['new_tasks']:
      for seed in PROTOCOL['new_seeds']:
       rr=[r for r in rows if r['task']==name and r['seed']==seed]
       for feat,c in chosen.items():
        seedrows.append(dict(task=name,seed=seed,feature=feat,**metrics(rr,feat,c['orientation'],c['threshold'])))
    writecsv(OUT/'metrics_by_seed.csv',seedrows)
    return result

def stationary_null():
    rng=np.random.default_rng(851)
    result=[]
    for a in [.01,.1,.5]:
        m=0.;prev=None;cross=energy=0.;count=0
        for t in range(101000):
            g=rng.normal();delta=g-m
            if t>=1000:
                cross+=delta*prev;energy+=.5*(delta*delta+prev*prev);count+=1
            m=(1-a)*m+a*g;prev=delta
        result.append(dict(alpha=a,predicted_correlation=-a/2,observed_correlation=cross/energy,n=count))
    return result

start=time.perf_counter()
snap=audit_snapshots()
rows=generate()
res=evaluate(rows)
res.update(snapshot_audit=snap,stationary_null=stationary_null(),elapsed_seconds=time.perf_counter()-start)
(OUT/'results.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
names=PROTOCOL['new_tasks'];xx=np.arange(len(names))
for k,feat in enumerate(PROTOCOL['features']):
    label=['Full vector','1 projection','4 projections','Energy','Time'][k]
    axs[0].bar(xx+(k-2)*.15,[res['new_tasks'][n][feat]['auc'] for n in names],width=.15,label=label)
axs[0].axhline(.5,color='gray',ls='--');axs[0].axhline(.75,color='black',ls=':')
axs[0].set_xticks(xx,['Rotated quadratic','Coupled nonconvex','Huber regression'])
axs[0].set(ylabel='AUC: next-state historical bias dominance',ylim=(0,1));axs[0].legend(fontsize=8,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01))
bins=snap['noisy_bias_bins']
for key in ['H_tighter','I_tighter','R_tighter']:
    axs[1].plot(range(len(bins)),[b[key] for b in bins],marker='o',label=key)
axs[1].set_xticks(range(len(bins)),['0-.01','.01-.1','.1-.5','.5-.9','.9-1'])
axs[1].set(xlabel='True historical bias share (noisy saved states)',ylabel='Fraction tighter than Cauchy',ylim=(0,1))
axs[1].legend(fontsize=8)
for ax in axs:ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'identification.png',dpi=180)
print(json.dumps({k:v for k,v in res.items() if k!='selection'},indent=2))
