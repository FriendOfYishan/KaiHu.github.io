"""New fixed-state exact-enumeration audit using the original Task definition."""
import ast,itertools,numpy as np
from pathlib import Path
tree=ast.parse(Path('core.py').read_text(encoding='utf-8-sig'))
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
env={"np":np,"itertools":itertools}
exec(compile(ast.Module(body=[node],type_ignores=[]),'Task','exec'),env)
task=env['Task']('quadratic',.2);x=np.array([1.,-1.]);gs=task.gradients(x);mu=gs.mean(0)
b=np.zeros(2);a=.2;m=(1-a)*b+a*gs;u=m/(np.sqrt(a*gs*gs)+.1);err=m-mu
actual=float(np.sum(err*u,axis=1).mean());bound=float(np.sqrt(np.sum(err*err,axis=1).mean()*np.sum(u*u,axis=1).mean()))
assert actual<=bound+1e-12
print({"signed_alignment":actual,"Cauchy_bound":bound})
