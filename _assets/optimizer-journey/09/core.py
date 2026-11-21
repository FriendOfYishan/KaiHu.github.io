"""Mathematical audit and small synthetic checks, not a neural-network benchmark."""
from pathlib import Path
import json
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent

def kernel(gradients, shrink=.5, transport=.5, beta1=.9, beta2=.999, eps=1e-8):
    """All gradients belong to CURRENT x. Two states: transported m and raw v.

    Prescribed gradients here isolate identities. They do not establish training
    performance. transport=shrink is the one-coefficient research candidate.
    """
    m=np.zeros_like(gradients[0],dtype=float);v=m.copy();old_s=m+eps
    records=[]
    for t,g in enumerate(gradients,1):
        a=(1-beta1)/(1-beta1**t);b=1-a
        q=(1-shrink)*g*g+shrink*m*m
        v=beta2*v+(1-beta2)*q
        s=np.sqrt(v/(1-beta2**t))+eps
        m=b*(s/old_s)**transport*m+a*g
        u=m/s
        records.append((m.copy(),v.copy(),s.copy(),u.copy(),q.copy()))
        old_s=s
    return records

def main():
    b,l,vp,z,dt,dp=sp.symbols('b l vp z dt dp')
    vt=b*vp+(1-b)*(z-l*dt)
    wt=vt+l*dt;wp=vp+l*dp
    correct=b*wp+(1-b)*z+b*l*(dt-dp)
    assert sp.simplify(wt-correct)==0
    ratio=sp.expand((1-l)**2*(1-b*b)+(l+b*(1-l))**2)
    assert sp.simplify(ratio-(1-2*l*(1-l)*(1-b)))==0
    a,k=sp.symbols('a k')
    assert sp.simplify((b+k)/(1+k)+(1-b)/(1+k)-1)==0
    mm,V,g=sp.symbols('mm V g')
    new_m=(1-a)*mm+a*g
    new_V=(1-a)*V+a*g*g-l*a*(1-a)*(g-mm)**2
    coherent_rhs=(1-a)*(V-mm*mm)+(1-l)*a*(1-a)*(g-mm)**2
    assert sp.simplify(new_V-new_m*new_m-coherent_rhs)==0
    # The shifted variable need not stay positive, even when the original v does.
    shifted_negative=.9*.05+.1*.5-.5
    assert shifted_negative<0
    # Explicit scalar counterexample to the claimed cumulative tilde-v recursion.
    v=0.;last_g=0.;last_w=0.;wrong=[]
    for g in [1.,2.,3.]:
        delta=g*g-last_g*last_g
        v=.9*v+.1*(g*g-.5*delta)
        w=v+.5*delta
        wrong.append(dict(g=g,v=v,tilde_v=w,claimed=last_w+.1*g*g))
        last_w=w;last_g=g
    assert not np.isclose(wrong[1]['tilde_v'],wrong[1]['claimed'])

    # Independent iid input to the second-moment filter. Var(z)=1 is sufficient.
    # Four concurrent filters, common innovations; only validates the exact filter law.
    rng=np.random.default_rng(20261108)
    chains=12000;steps=3000
    vv=np.zeros((2,chains));baseline=np.zeros_like(vv);old=np.zeros(chains)
    bs=np.array([.9,.999])[:,None];lam=.5
    for _ in range(steps):
        z=rng.normal(size=chains)
        baseline=bs*baseline+(1-bs)*z
        vv=bs*vv+(1-bs)*((1-lam)*z+lam*old)
        old=z
    obs=vv.var(axis=1,ddof=1)/baseline.var(axis=1,ddof=1)
    expected=1-2*lam*(1-lam)*(1-bs[:,0])
    # For beta=.999 the finite initialization is negligible at this scale.
    assert np.all(abs(obs-expected)<.007)

    # Geometric transport equivalence, zero-correction Adam limit and x recurrence.
    gradients=rng.normal(size=(140,7)) * np.exp(np.sin(np.arange(140)/9))[:,None]
    records=kernel(gradients)
    p=np.zeros(7);x=np.ones(7);previous_x=x.copy();last_u=np.zeros(7)
    previous_s=np.full(7,1e-8);previous_eta=None
    err_split=err_x=0.
    for t,(g,rec) in enumerate(zip(gradients,records),1):
        m,v,s,u,q=rec;a=.1/(1-.9**t);bt=1-a
        p=bt*p+a*g/s**.5
        err_split=max(err_split,float(np.max(abs(p/s**.5-u))))
        eta=.003/(1+.002*t);wd=.01
        direct=(1-eta*wd)*x-eta*u
        if t>1:
            displacement=x-(1-previous_eta*wd)*previous_x
            explicit=(1-eta*wd)*x+bt*eta/previous_eta*(previous_s/s)**.5*displacement-eta*a*g/s
            err_x=max(err_x,float(np.max(abs(explicit-direct))))
        previous_x,x=x,direct
        previous_s=s;previous_eta=eta;last_u=u
    assert err_split<1e-12 and err_x<1e-12
    neutral=kernel(gradients,shrink=0,transport=0)
    raw_m=np.zeros(7);raw_v=np.zeros(7);neutral_error=0.
    for t,(g,rec) in enumerate(zip(gradients,neutral),1):
        raw_m=.9*raw_m+.1*g;raw_v=.999*raw_v+.001*g*g
        u=(raw_m/(1-.9**t))/(np.sqrt(raw_v/(1-.999**t))+1e-8)
        neutral_error=max(neutral_error,float(np.max(abs(u-rec[3]))))
    assert neutral_error<1e-12

    # Shared prescribed denominator path isolates historical reweighting only.
    # Linear interpolation can pump an old velocity around a closed scale cycle.
    beta=.9; lam=.5;R=4.
    cycle_arithmetic=beta**2*(lam+(1-lam)*R)*(lam+(1-lam)/R)
    cycle_geometric=beta**2*R**(1-lam)*(1/R)**(1-lam)
    assert cycle_arithmetic>1 and np.isclose(cycle_geometric,beta**2)
    scales=np.exp(rng.normal(size=(100,7)))
    gains=(scales[:-1]/scales[1:])**(1-lam)
    assert np.allclose(np.prod(gains,axis=0),(scales[0]/scales[-1])**(1-lam))

    # Sufficient finite-time amplitude bound for the actual endogenous q recursion.
    b1=.9;b2=.7;lam=.5;rho=.5
    recs=kernel(gradients,lam,rho,b1,b2)
    kappa=b1*b2**(-(1-rho)/2)
    assert kappa<1
    ratios=[]
    for t,rec in enumerate(recs,1):
        js=np.arange(t)
        bound=(1-b1)/(1-b1**t)/np.sqrt((1-b2)*(1-lam))*(1-b2**t)**((1-rho)/2)
        bound*=np.sum(kappa**js*(1-b2**(t-js))**(rho/2))
        ratios.append(float(np.max(abs(rec[3]))/bound))
    assert max(ratios)<=1+1e-12

    # An externally frozen denominator makes transport disappear exactly.
    constant_scale=2.;u0=0.;u1=0.
    for g in gradients[:,0]:
        u0=.9*u0+.1*g/constant_scale
        u1=.9*(constant_scale/constant_scale)**.5*u1+.1*g/constant_scale
        assert u0==u1

    result=dict(
        scope='Mathematical identities and synthetic mechanisms only; no new neural-network/GPU runs.',
        tilde_identity=True,cumulative_counterexample=wrong,
        shifted_v_negative_counterexample=shifted_negative,coherent_moment_identity=True,
        lagged_square_filter=dict(beta=[.9,.999],lambda_=lam,exact_variance_ratio=expected.tolist(),
            simulated_variance_ratio=obs.tolist(),chains=chains,steps=steps,
            note='iid centered scalar inputs isolate the linear filter, not a changing neural-network gradient'),
        geometric_candidate=dict(split_normalization_max_error=err_split,x_recurrence_max_error=err_x,
            neutral_adam_max_error=neutral_error,cycle_arithmetic_gain=cycle_arithmetic,
            cycle_geometric_gain=cycle_geometric,telescope=True,
            amplitude_bound_max_ratio=max(ratios),sufficient_gain=kappa,
            frozen_denominator_equals_ordinary_momentum=True),
        claims_not_established=['global novelty','faster convergence','smaller stationary parameter variance',
                               'lower wall-clock cost','ICML acceptance'])
    (HERE/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
