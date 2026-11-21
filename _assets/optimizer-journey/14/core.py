"""AWA research variants. See theory/14_AWA_SHRINK_AND_RATE_ZH.md.

No EMA, no bias correction. `innovation` is preconditioned SGD whose
preconditioner uses AWA statistics; it is NOT an AWA momentum direction.
"""
import torch


class AWAShrink(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, mode="tdiff", shrink=0.5,
                 lever=True, eps=1e-8, bounds=(0.1, 10.0), weight_decay=0.):
        if mode not in ("original_safe", "tdiff", "innovation"):
            raise ValueError(mode)
        if not 0 <= shrink < 1 or lr <= 0 or eps <= 0:
            raise ValueError("invalid lr/shrink/eps")
        if not 0 < bounds[0] <= bounds[1]:
            raise ValueError("invalid preconditioner bounds")
        super().__init__(params, dict(lr=lr, mode=mode, shrink=shrink,
            lever=lever, eps=eps, bounds=bounds, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise ValueError("dense gradients required")
                st = self.state[p]
                if not st:
                    st.update(t=0, m=torch.zeros_like(p), s=torch.zeros_like(p))
                    if group["mode"] != "innovation":
                        st["P"] = torch.zeros_like(p)
                st["t"] += 1
                t, m, s = st["t"], st["m"], st["s"]
                w, lam, eps = 2/(t+1), group["shrink"], group["eps"]
                mode = group["mode"]
                if mode == "innovation":
                    # OLD statistics: measurable before drawing this gradient.
                    if t == 1:
                        b = torch.ones_like(p).clamp_(*group["bounds"])
                    else:
                        v = (1-lam)*s + lam*m.square()
                        if group["lever"]:
                            b = (v+eps**2).sqrt()/(s+eps**2)
                        else:
                            b = 1/(v+eps**2).sqrt()
                        b.clamp_(*group["bounds"])
                    step = group["lr"]/(t+1)
                    p.mul_(1-step*group["weight_decay"]).addcmul_(b,g,value=-step)
                    m.lerp_(g,w)
                    s.mul_(1-w).addcmul_(g,g,value=w)
                    continue
                if mode == "original_safe":
                    # Literal nested w, except undefined tilde correction removed.
                    q = g.square()-lam*w*(1-w)*(g-m).square()
                    s.mul_(1-w).add_(q,alpha=w)
                    m.lerp_(g,w)
                    P = m/(s.clamp_min(0).sqrt()+eps)
                else:
                    m.lerp_(g,w)
                    s.mul_(1-w).addcmul_(g,g,value=w)
                    v = (1-lam)*s + lam*m.square()
                    if group["lever"]:
                        P = t*m*(v+eps**2).sqrt()/(s+eps**2)
                    else:
                        P = t*m/(v+eps**2).sqrt()
                step = group["lr"]/t
                p.mul_(1-step*group["weight_decay"]).add_(P-st["P"],alpha=-step)
                st["P"].copy_(P)
        return loss
