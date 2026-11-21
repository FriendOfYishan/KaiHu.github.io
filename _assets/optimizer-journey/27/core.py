"""New audit aid: exact tracking can coexist with no optimization."""
x=1.0
grad=x
m=grad
eta=0.0
x_next=x-eta*m
assert (m-grad)**2 == 0 and x_next*x_next/2 == .5
print("tracking error = 0; objective gap = 0.5")
