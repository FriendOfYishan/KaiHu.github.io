from pathlib import Path
import numpy as np,json
rng=np.random.default_rng(2101);worst=0.
def solve(mu,b,P,H,S):
 d=mu-b;C=P@H@P;u0=mu@P@b;u1=mu@P@d;q0=b@C@b;q1=b@C@d;q2=d@C@d+np.trace(C@S)
 def score(a):return np.maximum(u0+u1*a,0)**2/(2*(q0+2*q1*a+q2*a*a))
 cand=[0.,1.];den=u0*q2-u1*q1
 if abs(den)>1e-14:
  a=(u1*q0-u0*q1)/den
  if 0<a<1:cand.append(a)
 a=max(cand,key=score);eta=max(u0+u1*a,0)/(q0+2*q1*a+q2*a*a)
 return a,eta,score,(u0,u1,q0,q1,q2)
for _ in range(300):
 mu=rng.normal(size=3);b=rng.normal(size=3);P=np.diag(np.exp(rng.normal(size=3)));H=np.diag(np.exp(rng.normal(size=3)));S=np.diag(np.exp(rng.normal(size=3)))
 a,eta,score,coef=solve(mu,b,P,H,S);grid=np.linspace(0,1,10001);gap=float(np.max(score(grid))-score(a));worst=max(worst,gap);assert gap<1e-10
H=np.diag([1.,4.]);P=np.diag([1.,.5]);x=np.array([1.,.3]);mu=H@x;b=np.array([.8,-.2]);S=np.diag([.2,.5]);W=P
D=(b-mu)@W@(b-mu);V=np.trace(W@S);astar=D/(D+V)
a,eta,score,co=solve(mu,b,P,H,S)
N=200000;A=rng.multivariate_normal(mu,S,N);B=rng.multivariate_normal(mu,S,N);C=rng.multivariate_normal(mu,S,N)
Dh=np.einsum('ni,ij,nj->n',b-A,W,b-B);Vh=.5*np.einsum('ni,ij,nj->n',A-B,W,A-B)
ah=np.clip(np.maximum(Dh,0)/(np.maximum(Dh,0)+Vh+1e-12),0,1)
def risk(alpha):return (1-alpha)**2*D+alpha**2*V
assert abs(risk(.3)-risk(astar)-(D+V)*(.3-astar)**2)<1e-12
M=(1-ah[:,None])*b+ah[:,None]*C
emp=np.mean(np.einsum('ni,ij,nj->n',M-mu,W,M-mu));pred=float(np.mean(risk(ah)))
mc=(1-a)*b+a*C;xp=x-eta*(mc@P);loss=.5*np.einsum('ni,ij,nj->n',xp,H,xp);expected=.5*x@H@x-score(a)
assert abs(emp-pred)<.02 and abs(loss.mean()-expected)<.02
out=dict(random_quadratic_joint_checks=300,max_grid_advantage=worst,example=dict(tracking_optimal_alpha=astar,joint_optimal_alpha=a,joint_eta=eta,expected_next_loss=expected,monte_carlo_next_loss=float(loss.mean()),expected_plugin_risk=pred,monte_carlo_plugin_risk=float(emp),true_D=D,estimated_D_mean=float(Dh.mean()),true_V=V,estimated_V_mean=float(Vh.mean()),plugin_excess_risk=float(np.mean(risk(ah))-risk(astar)),fixed_EMA_01_excess_risk=float(risk(.1)-risk(astar))))
Path(__file__).with_name('verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
