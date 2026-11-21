"""Algebra and adversarial checks of the stated, limited mathematical claims."""
from pathlib import Path
import json
import numpy as np
import sympy as sp

HERE=Path(__file__).resolve().parent

def main():
    t,g,h,C,lam=sp.symbols('t g h C lam',real=True)
    s0=(t-1)*h
    c1=C+(t-1)/t*(g-h)**2
    v0=s0**2/(t-1)+(1-lam)*C
    v1=(s0+g)**2/t+(1-lam)*c1
    assert sp.simplify(v1-v0-g*g+lam*(t-1)/t*(g-h)**2)==0

    rng=np.random.default_rng(1111)
    stats=[]
    for shrink in [0.,.5,.9]:
        # Independent streams also test arbitrary signs and many V decreases.
        dim,steps=128,1024
        grads=rng.uniform(-1,1,size=(steps,dim))
        grads[1::3]=0
        S=np.zeros(dim); Cn=S.copy(); Q=S.copy(); x=S.copy()
        delta=1-shrink; eps2=delta; eta=.3; radius=2.
        D=np.full(dim,np.sqrt(eps2)); D0=D.copy()
        positive=S.copy(); negative=S.copy(); smooth=S.copy()
        loss=S.copy(); uold=S.copy(); xd=np.ones(dim)
        weighted=np.zeros(dim); product=1.
        max_wd_error=0.
        for k,grad in enumerate(grads,1):
            loss+=grad*x
            smooth+=eta*.5*grad**2/D
            hprev=S/(k-1) if k>1 else np.zeros(dim)
            Cn+=(k-1)/k*(grad-hprev)**2
            S+=grad; Q+=grad**2
            V=S*S/k+delta*Cn
            newD=np.sqrt(V+eps2)
            positive+=np.maximum(newD-D,0)
            negative+=np.maximum(D-newD,0)
            u=S/newD
            x=np.clip(-eta*u,-radius,radius)
            tau=1/(np.sqrt(k)+np.sqrt(k-1))
            decay=1-eta*.1*tau
            xd=decay*xd-eta*(u-uold)
            # Unrolled recurrence: weighted is sum of discounted delta-U.
            weighted=decay*weighted+(u-uold)
            product*=decay
            max_wd_error=max(max_wd_error,float(np.max(abs(xd-(product-eta*weighted)))))
            uold=u; D=newD
        comparator=-radius*np.sign(S)
        regret=loss-S*comparator
        bound=D*comparator**2/(2*eta)+smooth+radius**2/(2*eta)*negative
        assert np.all(regret <= bound+1e-9)
        assert np.all(positive <= 2*np.sqrt(Q/delta)+1e-9)
        assert np.all(negative <= positive+1e-9)
        stats.append(dict(shrink=shrink,minimum_regret_bound_slack=float(np.min(bound-regret)),
            max_positive_variation_ratio=float(np.max(positive/(2*np.sqrt(Q/delta)))),
            weight_decay_unroll_error=max_wd_error))

    # Closed form constant signal and matched decoupled decay.
    n=256; mu=2.; eta=.2; wd=.3
    x=1.; potential=0.; maximum_signal_error=0.
    ref=1.
    for k in range(1,n+1):
        newu=np.sqrt(k)*np.sign(mu)
        tau=np.sqrt(k)-np.sqrt(k-1)
        x=(1-eta*wd*tau)*x-eta*(newu-potential)
        ref=(1-eta*wd*tau)*ref-eta*tau*np.sign(mu)
        maximum_signal_error=max(maximum_signal_error,abs(x-ref))
        potential=newu
    # Shared-scalar-scale nonconvex path inequality, two-dimensional objective.
    nonconvex=[]
    for shrink in [0.,.5,.9]:
        delta=1-shrink; eta=.3; eps=.1; L=1.2
        initial=np.array([2.,-1.]); x=initial.copy()
        s=np.zeros(2); center=0.; q=0.; lhs=0.; energy=0.; tv=0.
        oldD=eps; maxB=0.; maxM=0.; maxG=0.
        f=lambda z:float(np.sum(.1*z*z+1-np.cos(z)))
        for k in range(1,1025):
            grad=.2*x+np.sin(x)
            maxB=max(maxB,float(np.linalg.norm(x-initial)))
            maxM=max(maxM,f(x)); maxG=max(maxG,float(np.linalg.norm(grad)))
            gvec=grad+rng.normal(size=2)
            lhs+=float(grad@gvec)
            prevmean=s/(k-1) if k>1 else np.zeros(2)
            center+=(k-1)/k*float(np.sum((gvec-prevmean)**2))
            s+=gvec; q+=float(gvec@gvec)
            D=np.sqrt(float(s@s)/k+delta*center+eps*eps)
            nxt=initial-eta*s/D
            np.testing.assert_allclose(gvec,-D/eta*(nxt-x)-(D-oldD)/eta*(x-initial),atol=1e-11)
            energy+=D/eta*float(np.sum((nxt-x)**2))
            tv+=abs(D-oldD)
            x=nxt; oldD=D
        maxB=max(maxB,float(np.linalg.norm(x-initial))); maxM=max(maxM,f(x))
        kd=max(1.,1/np.sqrt(delta)-1)
        a0=eps*f(initial)/eta
        a1=(4*(maxM+maxG*maxB)/eta+2*L*eta+4*L*maxB**2*kd/eta)/np.sqrt(delta)
        energy_bound=(4*eta+8*maxB**2*kd/eta)*np.sqrt(q/delta)
        assert lhs<=a0+a1*np.sqrt(q)+1e-9
        assert energy<=energy_bound+1e-9
        assert tv<=4*np.sqrt(q/delta)+1e-9
        nonconvex.append(dict(shrink=shrink,path_descent_slack=a0+a1*np.sqrt(q)-lhs,
            energy_ratio=energy/energy_bound, observed_radius=maxB,
            note='Observed radius checks a finite path, not a uniform-in-time boundedness proof.'))
    result=dict(symbolic_identity=True, variation_regret_checks=stats,
                scalar_nonconvex_path_checks=nonconvex,
                constant_signal_matched_decay_error=maximum_signal_error,
                note='Finite checks corroborate proofs, not proofs of general NN convergence.')
    (HERE/'theorem_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
