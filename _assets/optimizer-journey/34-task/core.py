"""New audit aid for stable 34: reciprocal-residual identity on its positive domain."""
from fractions import Fraction as Q
c=Q(3);g=[Q(-1),Q(1),Q(2)];p=[Q(1,3)]*3
a=sum(w*x for w,x in zip(p,g));h=1/sum(w/(c+x) for w,x in zip(p,g))
r=sum(w*x*x/(c+x) for w,x in zip(p,g))
assert r == a-c+c*c/h
assert sum(w*(c+x) for w,x in zip(p,g)) == c+a
assert all(c+x>0 for x in g)
print("Domain and positive shifted arithmetic/harmonic statistics checked; no task advantage proved.")
