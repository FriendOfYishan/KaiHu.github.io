"""New standalone energy-identity audit using original schedule functions."""
import ast,math,numpy as np
from pathlib import Path
tree=ast.parse(Path('core.py').read_text(encoding='utf-8-sig'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'alpha','eta','q','coefficients'}]
env={"math":math};exec(compile(ast.Module(body=nodes,type_ignores=[]),'schedules','exec'),env)
rng=np.random.default_rng(26)
for t in range(3,103):
    a=env['alpha'](t);qt,qp,k,theta=env['coefficients'](t,1.);et=env['eta'](t,1.)
    b=rng.normal(size=7);v=b*b+1.;g=rng.normal(size=7);m=(1-a)*b+a*g
    D=1/(np.sqrt((1-a)*v+a*g*g)+.1);Dp=1/(np.sqrt(v)+.1)
    M=np.sum(m*m*D);I=np.sum((m-b)**2*D);V=np.sum(b*b*(qt*D-qp*Dp))
    R=k*np.sum((qt/k*m-b)**2*D);W=qp*np.sum(b*b*(Dp-math.sqrt(1-a)*D))
    assert abs((-et*M-qt*I+V)-(-theta*M-R-W))<1e-8
print("100 energy identities checked; no training results generated")
