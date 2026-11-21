from pathlib import Path
import json
import math
import sympy as sp

theta,k,r,n=sp.symbols('theta k r n',positive=True)
risk=(1-theta*k)**2*r+theta**2*n
opt=k*r/(k*k*r+n)
assert sp.simplify(sp.diff(risk,theta).subs(theta,opt))==0
assert sp.simplify(risk.subs(theta,opt)-n*r/(k*k*r+n))==0
assert sp.simplify(sp.diff(risk,theta,2)-2*(k*k*r+n))==0
out=dict(
    symbolic_optimal_step_verified=True,
    symbolic_conditional_risk_verified=True,
    exp_counterexample=dict(a=1,b=1,exp_sum=math.exp(2),sum_exp=2*math.e),
    kurtosis_counterexample=dict(curvatures=[1,100],x=0,
        mean_gradient=1,noise_variance=1,gradient_distribution='N(1,1) in both models',
        standardized_central_kurtosis=3),
    gaussian_exponential_bias=[dict(variance=v,multiplicative_bias=math.exp(v/2)) for v in [0,1,4]],
    scope='Symbolic checks and exact counterexamples; no claimed novel optimizer or training run')
dest=Path(__file__).resolve().parent/'results.json'
dest.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
