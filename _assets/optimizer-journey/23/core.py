"""Fixed-state algebra and exact finite-oracle checks; no training changes."""
import json
import math
import random
from pathlib import Path

rng = random.Random(231110)
checks = dict(states=0, pathwise_identities=0, covariance_bounds=0,
              expectation_certificates=0, gaussian_integrals=0)
max_identity_error = 0.0
max_gaussian_error = 0.0

def close(x, y, tol=1e-9):
    global max_identity_error
    err = abs(x-y)
    max_identity_error = max(max_identity_error, err)
    assert err <= tol*(1+abs(x)+abs(y)), (x,y)

def leq(x, y):
    assert x <= y + 1e-9*(1+abs(x)+abs(y)), (x,y)

def setup(a, b, v, eps):
    r0 = math.sqrt((1-a)*v)
    H = a/(r0+eps) + math.sqrt(a)*(1-a)*abs(b)/(r0+eps)**2
    def phi(g):
        return ((1-a)*b+a*g)/(math.sqrt((1-a)*v+a*g*g)+eps)
    def deriv(g):
        r = math.sqrt((1-a)*v+a*g*g)
        if r == 0:
            return a/eps
        return a*((1-a)*v+eps*r-(1-a)*b*g)/(r*(r+eps)**2)
    return phi, deriv, H

for case in range(2000):
    a = 1.0 if case % 20 == 0 else rng.uniform(.01,.95)
    b = rng.uniform(-2,2)
    v = b*b+rng.uniform(0,3)
    if a == 1:
        b,v = 0.0,0.0
    eps = rng.uniform(.1,1)
    L = rng.uniform(.1,3)
    mu = rng.uniform(-3,3)
    x = mu/L
    eta = rng.uniform(.001,.5)
    q = eta*(1-a)/(2*a)
    qprev = 0.0 if a == 1 else rng.uniform(0,1)
    Dprev = 1/(math.sqrt(v)+eps)
    phi, deriv, H = setup(a,b,v,eps)
    weights = [rng.uniform(.1,1) for _ in range(4)]
    total = sum(weights)
    probs = [p/total for p in weights]
    raw = [rng.uniform(-2,2) for _ in range(4)]
    center = sum(p*z for p,z in zip(probs,raw))
    noises = [z-center for z in raw]
    nu = sum(p*z*z for p,z in zip(probs,noises))
    Ecov=EA=EJ=Eupper=Edpsi=Erhs=Einnovrhs=0.0
    for p,noise in zip(probs,noises):
        g=mu+noise
        m=(1-a)*b+a*g
        vt=(1-a)*v+a*g*g
        D=1/(math.sqrt(vt)+eps)
        u=phi(g)
        M=m*m*D
        I=(m-b)**2*D
        V=b*b*(q*D-qprev*Dprev)
        J=(m-g)*u
        A=(m-mu)*u
        close(A,J+noise*u)
        close(J,-(1-a)/(2*a)*(M-b*b*D+I))
        leq(abs(deriv(g)),H)
        step=1e-5*(1+abs(g))
        close(deriv(g),(phi(g+step)-phi(g-step))/(2*step),tol=1e-6)
        xn=x-eta*u
        dpsi=L*(xn*xn-x*x)/2+q*M-qprev*b*b*Dprev
        core=-eta*M-q*I+L*eta*eta*u*u/2+V
        close(dpsi,core+eta*noise*u)
        checks['pathwise_identities']+=3
        Ecov+=p*noise*u
        EA+=p*A
        EJ+=p*J
        Eupper+=p*(J+H*(g-b)**2)
        Edpsi+=p*dpsi
        Erhs+=p*core
        Einnovrhs+=p*(core+eta*H*(g-b)**2)
    leq(abs(Ecov),H*nu)
    leq(EA,EJ+H*nu)
    leq(EA,Eupper)
    leq(Edpsi,Erhs+eta*H*nu)
    leq(Edpsi,Einnovrhs)
    checks['covariance_bounds']+=1
    checks['expectation_certificates']+=4
    checks['states']+=1

# Numerical integration of both sides of Stein's identity. No sampling error.
for case in range(30):
    a=rng.uniform(.03,.9)
    b=rng.uniform(-2,2)
    v=b*b+rng.uniform(.2,2)
    eps=rng.uniform(.2,1)
    mu=rng.uniform(-2,2)
    sigma=rng.uniform(.2,1.5)
    phi,deriv,H=setup(a,b,v,eps)
    N=12000
    h=20/N
    lhs=rhs=0.0
    for j in range(N+1):
        z=-10+j*h
        weight=1 if j in (0,N) else (4 if j%2 else 2)
        density=math.exp(-z*z/2)/math.sqrt(2*math.pi)
        g=mu+sigma*z
        lhs+=weight*density*(sigma*z)*phi(g)
        rhs+=weight*density*sigma*sigma*deriv(g)
    lhs*=h/3
    rhs*=h/3
    max_gaussian_error=max(max_gaussian_error,abs(lhs-rhs))
    assert abs(lhs-rhs)<1e-8,(lhs,rhs)
    checks['gaussian_integrals']+=1

# Two indistinguishable first-step observations, opposite true alignments.
eps=.5
g=m=v=1.0
u=m/(math.sqrt(v)+eps)
assert (m-2)*u < 0 < (m-0)*u
result=dict(status='passed',seed=231110,**checks,
            max_algebra_and_derivative_check_error=max_identity_error,
            max_gaussian_integral_error=max_gaussian_error,
            opposite_alignment_counterexample=dict(observed_g=g,observed_m=m,
                observed_v=v,world_A_alignment=(m-2)*u,world_B_alignment=m*u),
            scope='Finite-support conditional expectations enumerated exactly; Gaussian identities checked by numerical quadrature. Not GPU training, not a speed or convergence benchmark.')
Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
