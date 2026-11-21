"""New audit of the historical AUC helper; does not reproduce the snapshot coverage experiment."""
import ast,numpy as np
from pathlib import Path
tree=ast.parse(Path('core.py').read_text(encoding='utf-8-sig'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='auc')
env={"np":np};exec(compile(ast.Module(body=[node],type_ignores=[]),'auc','exec'),env)
y=np.array([0,0,1,1]);s=np.array([0.,1.,2.,3.]);value=env['auc'](y,s)
assert abs(value-1)<1e-12
print({"perfect_ranking_auc":value,"scope":"helper check only; report contains the actual negative result"})
