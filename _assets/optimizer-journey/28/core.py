"""New audit aid: exact deterministic transport on a scalar quadratic."""
x_old,x_new,k=2.,1.,3.
m_old=k*x_old
correction=k*(x_new-x_old)
assert m_old+correction == k*x_new
print("Exact transport requires curvature/movement information, not just a new name.")
