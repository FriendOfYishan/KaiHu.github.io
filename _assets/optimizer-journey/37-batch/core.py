"""New minimal audit of the historical effective-batch schedule; no training claims."""
import math
def budget(grow):
    used=t=0
    while used<32000:
        t+=1
        n=min(32000-used,math.ceil(32*t**.1) if grow else 32)
        used+=n
    return t,used*128
fixed,growing=budget(False),budget(True)
assert fixed[1]==growing[1]==4096000 and growing[0]<fixed[0]
print({"fixed":fixed,"growing":growing,"columns":["updates","tokens"]})
