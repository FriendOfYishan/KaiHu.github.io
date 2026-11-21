"""Algebra/inequality checks; the plot is a known-noise tracking illustration."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def grad(x):return np.sin(x)+.1*x
def obj(x):return np.sum(1-np.cos(x)+.05*x*x)

def main():
    rng=np.random.default_rng(634)
    a=.1;beta=1-a;kappa=beta/a;gamma=(1+a)/(1-a)
    L=1.1;eta=.005;lo=.5;hi=2.;delta=.05
    maxima=dict(risk_identity=0.,virtual_identity=0.,lower_bound_violation=0.,
                release_risk_increase=0.,descent_certificate_violation=0.,
                virtual_descent_violation=0.)
    triggers=0;cases=0
    for dim in [1,7,64]:
        for _ in range(200):
            x=rng.normal(size=dim);m=rng.normal(size=dim)
            h=rng.uniform(lo,hi,size=dim);hp=rng.uniform(lo,hi,size=dim)
            release=(rng.random(dim)<.2).astype(float)
            alpha=a+beta*release;mu=grad(x)
            v=rng.normal(size=dim)*.1
            noises=[2*v,-v];prob=[1/3,2/3];nu=2*np.dot(v,v)
            z=x-eta*kappa*hp*m;w=grad(z)
            e=h*(mu-w)+kappa*release*h*(mu-m)+kappa*(h-hp)*m
            expected_virtual=0.
            for noise,p in zip(noises,prob):
                g=mu+noise;mn=(1-alpha)*m+alpha*g;xn=x-eta*h*mn
                zn=xn-eta*kappa*h*mn
                rhs=z-eta*h*g-eta*kappa*release*h*(g-m)+eta*kappa*(hp-h)*m
                maxima['virtual_identity']=max(maxima['virtual_identity'],float(np.max(np.abs(zn-rhs))))
                expected_virtual+=p*obj(zn)
            virtual_bound=obj(z)-eta*lo/2*np.dot(w,w)+2*eta/lo*np.dot(e,e)+L*eta**2*hi**2*nu/(2*a*a)
            maxima['virtual_descent_violation']=max(maxima['virtual_descent_violation'],expected_virtual-virtual_bound)

            # One whole vector is a block for the release-risk check.
            current_a=a if rng.random()<.8 else 1.
            for noise,p in zip(noises,prob):
                g=mu+noise;mn=(1-current_a)*m+current_a*g;xn=x-eta*h*mn
                mun=grad(xn);q=np.sqrt(nu/delta);move=L*np.linalg.norm(xn-x)
                lower=max(0.,np.linalg.norm(g-mn)-q-move)**2
                bias=np.dot(mun-mn,mun-mn)
                maxima['lower_bound_violation']=max(maxima['lower_bound_violation'],lower-bias)
                risks=[]
                for weight in [a,1.]:
                    empirical=sum(pp*np.dot((1-weight)*mn+weight*(mun+eps)-mun,
                                             (1-weight)*mn+weight*(mun+eps)-mun)
                                  for eps,pp in zip(noises,prob))
                    exact=(1-weight)**2*bias+weight**2*nu
                    maxima['risk_identity']=max(maxima['risk_identity'],abs(empirical-exact))
                    risks.append(empirical)
                fire=lower>gamma*nu
                if fire:
                    triggers+=1
                    maxima['release_risk_increase']=max(maxima['release_risk_increase'],risks[1]-risks[0])
                # The next scalar preconditioner is history measurable.
                hs=1.3;weight=1. if fire else a
                actual=sum(pp*obj(xn-eta*hs*((1-weight)*mn+weight*(mun+eps))) for eps,pp in zip(noises,prob))
                risk=(1-weight)**2*bias+weight**2*nu
                bound=obj(xn)-eta*hs/2*np.dot(mun,mun)+eta*hs/2*risk
                maxima['descent_certificate_violation']=max(maxima['descent_certificate_violation'],actual-bound)
                cases+=1
    assert maxima['risk_identity']<1e-10
    assert maxima['virtual_identity']<1e-10
    for key in ['lower_bound_violation','release_risk_increase','descent_certificate_violation','virtual_descent_violation']:
        assert maxima[key]<1e-10,(key,maxima[key])

    # Independently check the existing-microbatch scalar noise estimator.
    K=4;N=100000;dimension=8
    micro=rng.normal(0,.3,size=(N,K,dimension));g=micro.mean(axis=1)
    estimated=(np.mean(np.sum(micro*micro,axis=2),axis=1)-np.sum(g*g,axis=1))/(K-1)
    target=dimension*.3**2/K
    se=estimated.std(ddof=1)/np.sqrt(N)
    assert abs(estimated.mean()-target)<5*se

    # Illustration only: exogenous scalar mean jumps, valid known sigma, zero motion fee.
    N=10000;sigma=.05;steps=7
    noise=rng.normal(0,sigma,(steps,N));true=np.full(steps,3.);true[0]=1.
    ema=np.ones(N);flash=np.ones(N);r=np.zeros(N,dtype=bool)
    em=[ema.copy()];fl=[flash.copy()];rates=[]
    for t in range(1,steps):
        g=true[t]+noise[t]
        ema=beta*ema+a*g
        at=np.where(r,1.,a);flash=(1-at)*flash+at*g
        lower=np.maximum(0,np.abs(g-flash)-sigma/np.sqrt(delta))**2
        r=lower>gamma*sigma*sigma
        em.append(ema.copy());fl.append(flash.copy());rates.append(float(r.mean()))
    em=np.array(em);fl=np.array(fl)
    fig,ax=plt.subplots(figsize=(8,4.3))
    ax.plot(true,'o-',label='True gradient mean',color='#4e97ff')
    ax.plot(em.mean(axis=1),'o-',label='EMA, a=0.1',color='#50b969')
    ax.plot(fl.mean(axis=1),'o-',label='Flash, evidence controls next sample',color='#f4833c')
    ax.fill_between(range(steps),fl.mean(axis=1)-fl.std(axis=1),fl.mean(axis=1)+fl.std(axis=1),color='#f4833c',alpha=.15)
    ax.set(xlabel='Observations (change at 1)',ylabel='Momentum estimate',
           title='Mechanism illustration: known noise, exogenous step; not Transformer training')
    ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout()
    fig.savefig(ROOT/'36_FLASH_STEP_ILLUSTRATION.png',dpi=170);plt.close(fig)
    result=dict(description='Exact finite-distribution checks plus a labeled tracking illustration.',
                finite_distribution_cases=cases,certified_releases=triggers,maxima=maxima,
                microbatch_variance=dict(expected=target,observed=float(estimated.mean()),standard_error=float(se)),
                step_illustration=dict(ema_mean=em.mean(axis=1).tolist(),flash_mean=fl.mean(axis=1).tolist(),
                                      next_release_fraction=rates,known_sigma=sigma,paths=N))
    (ROOT/'36_FLASH_CHECKS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
