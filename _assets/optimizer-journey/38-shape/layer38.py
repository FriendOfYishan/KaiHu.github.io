"""Layer38 reference optimizers (single precision, unfused, CPU/GPU agnostic).

The family is indexed by the *shape bit budget* b: how many bits per parameter
are spent on storing a per-coordinate inverse scale (the "shape" P).

    name        shape P                          persistent bytes/param
    adamw       EMA(g^2), fp32 second moment     8   (m + v)       reference
    q8          log2-quantised EMA(g^2), 8 bit   5   (m + 1 byte)
    q4          log2-quantised EMA(g^2), 4 bit   4.5 (m + 0.5 byte)
    sign        instantaneous |m|  (u = sign m)  4   (m only)
    layer       none               (u = m)       4   (m only)   <- Layer37 family

Everything except `adamw*` shares one scale law: a per-group trust ratio h_g
refreshed every K steps that normalises the *update* RMS to tau, with a
multiplicative total-variation cap |h_new/h_old - 1| <= min(tv/j, 1).  h has
units of "1 / direction RMS", so the global lr is a *relative* step: every
group moves by about lr per coordinate per step.  lr is therefore dimensionless
and directly transferable across arms, which the Layer37 gradient-RMS law did
not allow (its grid had to be 0.03 ... 10 while AdamW's was 1e-4 ... 3e-2).

Only `m` (4 bytes/param) and, for b>0, the quantised shape are persistent.
The exact second moment `self.v` kept below is a convenience of the reference
code; a fused kernel carries the EMA in registers and writes back only the
quantised levels (see workspace_bytes()).
"""

import math
import torch

MODES = ("adamw", "adamw_cache", "adam_sparse", "layer", "sign", "q4", "q8", "q32")


# --------------------------------------------------------------------------- #
def quant_log2(v, bits, span):
    """Quantise log2(v) with `bits` levels covering `span` octaves.

    Returns (levels, centre, octave-per-level).  The levels are centred on the
    group mean so the integer range is used on the data that exists.
    """
    lg = torch.log2(v.clamp_min(1e-38))
    step = span / (2 ** bits)
    centre = float(lg.mean())
    lev = torch.round((lg - centre) / step)
    half = 2 ** (bits - 1)
    clipped = float((lev.abs() > half).float().mean())
    lev = lev.clamp(-half, half - 1)
    dt = torch.int8 if bits <= 8 else torch.int16
    return lev.to(dt), centre, step, clipped


def dequant_log2(lev, centre, step):
    return torch.pow(2.0, centre + step * lev.to(torch.float32))


# --------------------------------------------------------------------------- #
class L38:
    def __init__(self, params, groups, mode="q4", beta=0.9, rho=0.95, K=8, Kq=1,
                 wd=0.0, eps=1e-8, tv=4.0, tau=1.0, span=40.0,
                 betas=(0.9, 0.95)):
        assert mode in MODES, mode
        self.params = [p for p in params if p.requires_grad]
        self.groups = list(groups)[:len(self.params)]
        assert len(self.groups) == len(self.params)
        self.mode = mode
        self.beta, self.rho, self.K, self.Kq = beta, rho, K, Kq
        self.wd, self.eps, self.tv, self.tau, self.span = wd, eps, tv, tau, span
        self.bits = {"q4": 4, "q8": 8, "q32": 32}.get(mode, 0)
        self.ng = max(self.groups) + 1
        self.t = 0
        self.j = 0
        self.hits = torch.zeros(self.ng)
        self.clip_frac = 0.0
        self.nq = 0

        self.m = [torch.zeros_like(p) for p in self.params]
        if self.bits:
            dt = torch.int8 if self.bits <= 8 else torch.int16
            self.lev = [torch.zeros(p.numel(), dtype=dt) for p in self.params]
            self.qc = [0.0] * len(self.params)
            self.qs = [1.0] * len(self.params)
            self.v = [torch.zeros_like(p) for p in self.params]
        if mode.startswith("adam"):
            self.m1 = [torch.zeros_like(p) for p in self.params]
            self.v = [torch.zeros_like(p) for p in self.params]
            if mode != "adamw":
                self.inv = [torch.ones_like(p) for p in self.params]
        else:
            self.h = torch.ones(self.ng)
        self.betas = betas

    # ---- accounting ------------------------------------------------------
    def persistent_bytes(self):
        n = sum(p.numel() for p in self.params)
        if self.mode.startswith("adam"):
            return 8 * n + (4 * n if self.mode != "adamw" else 0)
        return 4 * n + math.ceil(self.bits / 8 * n)

    def workspace_bytes(self):
        """Transient buffers a fused kernel would not need."""
        n = sum(p.numel() for p in self.params)
        if self.mode == "adamw_cache" or self.mode == "adam_sparse":
            return 4 * n
        if self.bits in (4, 8):
            return 4 * n
        return 0

    # ---- main step -------------------------------------------------------
    def step(self, lr):
        self.t += 1
        with torch.no_grad():
            do_ref = (self.t - 1) % self.K == 0            # refresh at t = 1, 1+K, ...
            if do_ref:
                self.j += 1
            if self.mode.startswith("adam"):
                self._step_adam(lr, do_ref)
            else:
                self._step_l38(lr, do_ref)

    # -- layer / sign / q4 / q8 / q32 -------------------------------------
    def _refresh_shape(self, i, g):
        self.v[i].mul_(self.rho).addcmul_(g, g, value=1.0 - self.rho)
        vs = self.v[i].reshape(-1) / (1.0 - self.rho ** self.t)
        lev, c, s, clip = quant_log2(vs, self.bits, self.span)
        self.lev[i], self.qc[i], self.qs[i] = lev, c, s
        self.clip_frac += clip
        self.nq += 1

    def _step_l38(self, lr, do_ref):
        us, touched = [None] * len(self.params), [None] * len(self.params)
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue
            bm = self.m[i].mul_(self.beta).add_(g, alpha=1.0 - self.beta)
            mhat = bm / (1.0 - self.beta ** self.t)
            if self.bits and (do_ref or self.Kq == 1):
                self._refresh_shape(i, g)
            if self.mode == "layer":
                u = mhat
            elif self.mode == "sign":
                u = torch.sign(mhat)
            else:
                if self.bits == 32:
                    vq = self.v[i] / (1.0 - self.rho ** self.t)
                else:
                    vq = dequant_log2(self.lev[i], self.qc[i], self.qs[i]).view_as(p)
                u = mhat / (torch.sqrt(vq) + self.eps)
            us[i] = u
            touched[i] = True
            if self.wd:
                p.mul_(1.0 - lr * self.wd)
            p.add_(u, alpha=-lr * float(self.h[self.groups[i]]))

        if not do_ref:
            return
        for b in range(self.ng):
            idx = [i for i, gi in enumerate(self.groups) if gi == b and touched[i]]
            if not idx:
                continue
            num = sum(float(us[i].pow(2).sum()) for i in idx)
            den = sum(self.params[i].numel() for i in idx)
            rms = math.sqrt(num / den) + 1e-30
            h_old = float(self.h[b])
            # trust ratio: pick h so that the *applied* update has RMS tau.
            # prop = tau / rms(u) is independent of h_old; the cap below limits
            # how fast h may travel, so the two must not be conflated.
            prop = self.tau / rms
            c = min(self.tv / self.j, 1.0) if self.tv else 1.0
            lo, hi = h_old * (1.0 - c), h_old * (1.0 + c)
            h_new = min(max(prop, lo), hi)
            if abs(h_new - prop) > 1e-12 * max(1.0, abs(prop)):
                self.hits[b] += 1
            self.h[b] = min(max(h_new, 1e-12), 1e12)

    # -- adamw family ------------------------------------------------------
    def _step_adam(self, lr, do_ref):
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue
            self.m1[i].mul_(b1).add_(g, alpha=1.0 - b1)
            if self.mode == "adam_sparse":
                if do_ref:
                    self.v[i].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    self.inv[i] = 1.0 / (torch.sqrt(self.v[i] / (1.0 - b2 ** self.j)) + self.eps)
                inv = self.inv[i]
            elif self.mode == "adamw_cache":
                self.v[i].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                if do_ref:
                    self.inv[i] = 1.0 / (torch.sqrt(self.v[i] / (1.0 - b2 ** self.t)) + self.eps)
                inv = self.inv[i]
            else:
                self.v[i].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                inv = 1.0 / (torch.sqrt(self.v[i] / (1.0 - b2 ** self.t)) + self.eps)
            mhat = self.m1[i] / (1.0 - b1 ** self.t)
            p.mul_(1.0 - lr * self.wd)
            p.addcmul_(mhat, inv, value=-lr)


def make_opt(name, params, groups, **kw):
    return L38(params, groups, mode=name, **kw)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    P = [torch.nn.Parameter(torch.randn(7, 5)) for _ in range(3)]
    Q = [torch.nn.Parameter(p.detach().clone()) for p in P]
    o1 = L38(P, [0, 0, 1], mode="adamw", wd=0.01, betas=(0.9, 0.95))
    o2 = torch.optim.AdamW(Q, lr=1.0, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8)
    for _ in range(9):
        for p, q in zip(P, Q):
            g = torch.randn_like(p)
            p.grad, q.grad = g.clone(), g.clone()
        o1.step(0.01)
        o2.param_groups[0]["lr"] = 0.01
        o2.step()
    err = max(float((a - b).detach().abs().max()) for a, b in zip(P, Q))
    assert err < 1e-6, err
    print("adamw equivalence              max|diff| = %.2e" % err)

    for mode in ("layer", "sign", "q4", "q8", "q32"):
        P = [torch.nn.Parameter(torch.randn(256)) for _ in range(2)]
        o = L38(P, [0, 1], mode=mode, K=4, tau=1.0, tv=None)
        rms = []
        for t in range(41):
            for p in P:
                p.grad = torch.randn_like(p)
            before = [p.detach().clone() for p in P]
            o.step(1e-3)
            rms += [float(((p.detach() - b) / 1e-3).pow(2).mean().sqrt()) for p, b in zip(P, before)]
        print("trust ratio %-5s stationary RMS after 40 steps = %.3f (target 1)"
              % (mode, sum(rms[-4:]) / 4))

    v = torch.exp(torch.randn(200000) * 1.5) * 1e-6
    for bits, span in ((4, 12.0), (8, 12.0), (4, 16.0)):
        lev, c, s, clip = quant_log2(v, bits, span)
        rec = dequant_log2(lev, c, s)
        rel = ((1 / rec.sqrt()) - (1 / v.sqrt())).abs() * v.sqrt()
        ok = (lev > lev.min()) & (lev < lev.max())
        print("quant b=%d span=%.0f  max rel err 1/sqrt(v): in-range %.4f  median %.4f  "
              "predicted in-range %.4f  clipped %.4f"
              % (bits, span, float(rel[ok].max()), float(rel.median()),
                 2 ** (s / 4) - 1, clip))
    print("OK")
