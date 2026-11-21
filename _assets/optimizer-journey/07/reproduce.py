from core import certified_gate
gate,upper=certified_gate(alignment_lower=1.,baseline_energy_upper=1.,increment_energy_upper=1.,curvature_upper=1.,lr=.1)
assert 0<=gate<=1 and upper<=0
print({"gate":gate,"certified_upper":upper})
