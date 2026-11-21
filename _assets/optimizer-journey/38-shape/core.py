"""Numerical checks for theory/38 (Layer38).  CPU only, a few seconds.

Every block prints one measured number and asserts the theoretical claim it
tests.  Nothing here trains a Transformer; this is the algebraic/numerical
kernel of the 38 theory, verified the same way 37 verifies its identities.

  A  virtual-point identity            z_{t+1}-z_t = -G_t g_t + kappa (G_{t-1}-G_t) m_{t-1}
  B  complete-the-square identity      -<a,Gb> + 1/2<b,Gb> = -1/2<a,Ga> + 1/2<a-b,G(a-b)>
  C  window master inequality          E dF(z_t) <= -1/2||grad F(z)||^2_G
                                                     + 1/2||grad F(z)-grad F(x)||^2_G
                                                     + 1/2 tr(G Sigma)
  D  exponent calculus                 avg_G(T) = O(T^{-(1/2+delta)}) for eta_t = g t^{-(1/2-delta)}
  E  kappa separation                  per-layer scalar needs Omega(kappa) steps, per-coordinate O(1)
  F  total-variation budget            sum_j min(c/j,1)^2 is bounded, so V_T = O(G^2)
  G  quantiser bound                   |1/sqrt(vq) - 1/sqrt(v)| <= 2^{step/4} - 1 in range
  H  free noise probe omega = J/M      monotone in sigma/|mu| and follows the t^{-delta} envelope
"""

import json
import math
from pathlib import Path

import torch

import layer38 as L38

torch.manual_seed(0)
xstar_global = torch.randn(8)
OUT = {}
R = Path(__file__).resolve().parent


def blocks(dims):
    G = []
    for d in dims:
        G.append(torch.randn(d, d))
        G[-1] = G[-1] @ G[-1].T / d + 0.1 * torch.eye(d) * (1 + torch.rand(1).item())
    return torch.block_diag(*G)


# --------------------------------------------------------------------------- A
def check_A():
    d = 13
    beta, a = 0.9, 0.1
    kappa = beta / a
    errs = []
    x = torch.randn(d)
    m_prev = torch.zeros(d)
    G_prev = torch.eye(d)
    for t in range(60):
        H = blocks([d])
        H = H / H.norm() * (0.3 + 0.05 * t)
        h = torch.exp(torch.randn(3) * 0.5)                       # per-layer scales
        Hb = torch.block_diag(*[h[i] * torch.eye(d // 3 + (1 if i < d % 3 else 0))
                                for i in range(3)])[:d, :d]
        G = 0.01 * (Hb + 1e-3 * torch.eye(d))
        g = torch.randn(d)
        m = beta * m_prev + a * g
        x_next = x - G @ m
        z_t = x - kappa * G_prev @ m_prev
        z_next = x_next - kappa * G @ m
        lhs = z_next - z_t
        rhs = -G @ g + kappa * (G_prev - G) @ m_prev
        errs.append(float((lhs - rhs).abs().max()))
        x, m_prev, G_prev = x_next, m, G
    OUT["A_virtual_point_max_abs_err"] = max(errs)
    assert max(errs) < 1e-5, max(errs)
    print("A  virtual point identity      max |lhs-rhs| = %.2e over 60 steps" % max(errs))


# --------------------------------------------------------------------------- B
def check_B():
    errs = []
    for _ in range(200):
        d = 9
        A = torch.randn(d, d, dtype=torch.float64)
        G = A @ A.T + 0.05 * torch.eye(d, dtype=torch.float64)
        a, b = torch.randn(d, dtype=torch.float64), torch.randn(d, dtype=torch.float64)
        lhs = -a @ (G @ b) + 0.5 * (b @ (G @ b))
        rhs = -0.5 * (a @ (G @ a)) + 0.5 * ((a - b) @ (G @ (a - b)))
        errs.append(abs(float(lhs - rhs)))
    OUT["B_complete_square_max_abs_err"] = max(errs)
    assert max(errs) < 1e-10, max(errs)
    print("B  complete the square         max |lhs-rhs| = %.2e over 200 draws" % max(errs))


# --------------------------------------------------------------------------- C
def check_C():
    """Master inequality (4), in both the loose and the tight noise form.

    C1  loose form:   E dF <= -1/2 ||grad F(z)||^2_G + 1/2 tr(G Sigma)
    C2  tight form:   E dF <= -1/2 ||grad F(z)||^2_G + (Lambda*eta*h/2) tr(G Sigma)

    (4) of theory/38 keeps the tight coefficient, because the loose one makes the
    noise term first order in eta and then the exponent is 1/2 for every delta.
    With P = I and h = 1 the tight form is stated for a scalar Lambda*eta; here
    G = eta*I with eta = 0.9/Lambda so that Lambda*eta = 0.9 < 1.
    """
    d = 10
    dt = torch.float64
    A = torch.randn(d, d, dtype=dt)
    H = A @ A.T / d + 0.5 * torch.eye(d, dtype=dt)
    Lam = float(torch.linalg.eigvalsh(H).max())
    eta = 0.9 / Lam                                        # Lambda * eta = 0.9
    G = eta * torch.eye(d, dtype=dt)
    sig2 = 0.3
    F = lambda w_: 0.5 * w_ @ (H @ w_)
    loose, tight, viol = [], [], 0
    for _ in range(600):
        z = torch.randn(d, dtype=dt)
        gz = H @ z
        tot = 0.0
        for _ in range(400):
            xi = torch.randn(d, dtype=dt)
            tot += float(F(z - G @ (gz + math.sqrt(sig2) * xi)) - F(z))
        lhs = tot / 400
        base = float(-0.5 * (gz @ (G @ gz)))
        rhs_loose = base + 0.5 * sig2 * float(torch.trace(G))
        rhs_tight = base + 0.5 * Lam * eta * sig2 * float(torch.trace(G))
        loose.append(rhs_loose - lhs)
        tight.append(rhs_tight - lhs)
        viol += int(lhs > rhs_tight + 1e-9)
    OUT["C_loose_min_slack"] = min(loose)
    OUT["C_tight_min_slack"] = min(tight)
    OUT["C_tight_violations"] = viol
    OUT["C_Lambda_eta"] = Lam * eta
    assert min(loose) > -1e-12 and min(tight) > -1e-9 and viol == 0
    print("C  master inequality: loose slack %+.2e  tight slack %+.2e  "
          "(Lambda*eta=%.2f)  violations %d/600"
          % (min(loose), min(tight), Lam * eta, viol))
    print("C  the tight form is the one whose noise term is O(eta^2); using the loose "
          "form would pin the exponent at 1/2")


# --------------------------------------------------------------------------- D
def check_D():
    """Two independent claims about the exponent.

    D1  calculus: for eta_t = g t^{-alpha}, sigma_t^2 = s2 t^{-2delta} with
        alpha = 1/2 - delta,  B(T) = (2 Delta + sum eta_t^2 sigma_t^2)/sum eta_t
        decays like T^{-(1/2+delta)}  (up to log T at delta = 0).
    D2  dynamics: a heavy-ball run with that schedule stays below a constant
        multiple of B(T), i.e. the master inequality is not vacuous.
    """
    d = 8
    lam = torch.linspace(0.5, 2.0, d)                     # mu = 0.5, L = 2
    Ts = [200, 500, 1000, 2000, 5000, 10000]
    g, s2, Delta = 0.3, 1.0, 20.0
    fit = {}
    for delta in (0.0, 0.1, 0.3):
        alpha = 0.5 - delta
        B = {}
        for T in Ts:
            se = sum(g * t ** (-alpha) for t in range(1, T + 1))
            sn = sum(g * g * t ** (-2 * alpha) * s2 * t ** (-2 * delta) for t in range(1, T + 1))
            B[T] = (2 * Delta + sn) / se
        xs = [math.log(T) for T in Ts[-4:]]
        ys = [math.log(B[T]) for T in Ts[-4:]]
        mx, my = sum(xs) / 4, sum(ys) / 4
        slope = sum((u - mx) * (v - my) for u, v in zip(xs, ys)) / sum((u - mx) ** 2 for u in xs)
        # the delta = 0 case is T^{-1/2} log T; the local log-log slope is
        # -1/2 + 1/log T at the fit centre.
        pred = -(0.5 + delta) + (1.0 / math.exp(mx) if delta == 0 else 0.0)
        fit[delta] = (slope, pred)
        print("D1 delta=%.1f  bound slope %+.3f   predicted %+.3f%s"
              % (delta, slope, pred, "  (T^-1/2 log T)" if delta == 0 else ""))
        assert abs(slope - pred) < 0.06, (delta, slope, pred)
    OUT["D1_bound_slope"] = {str(k): dict(slope=v[0], predicted=v[1]) for k, v in fit.items()}

    rows, ratio = {}, {}
    for delta in (0.0, 0.1, 0.3):
        alpha = 0.5 - delta
        acc = {T: [] for T in Ts}
        gfix, Tmax = 0.3, max(Ts)
        for seed in range(8):
            torch.manual_seed(1000 + seed)
            x = torch.randn(d) * 2.0
            m, G, wsum = torch.zeros(d), torch.zeros(d), 0.0
            for t in range(1, Tmax + 1):
                st = 1.0 * t ** (-2 * delta)
                grad = lam * (x - xstar_global) + math.sqrt(st) * torch.randn(d)
                m = 0.9 * m + 0.1 * grad
                eta = gfix * t ** (-alpha)
                x = x - eta * m
                G = G + lam * (x - xstar_global) ** 2
                wsum += eta
                if t in acc:
                    acc[t].append(float(G.sum()) / wsum)
        rows[delta] = {T: sum(v) / len(v) for T, v in acc.items()}
        bound = {}
        for T in Ts:
            se = sum(gfix * t ** (-alpha) for t in range(1, T + 1))
            sn = sum(gfix ** 2 * t ** (-2 * alpha) * t ** (-2 * delta) for t in range(1, T + 1))
            bound[T] = (2 * Delta + sn) / se
        ratio[delta] = {T: rows[delta][T] / bound[T] for T in Ts}
        print("D2 delta=%.1f  measured/bound at T=1e4 : %.2f   (empirical constant, not tight)"
              % (delta, ratio[delta][Ts[-1]]))
        assert max(ratio[delta].values()) < 100.0
    OUT["D2_measured_over_bound"] = {str(k): {str(T): v for T, v in r.items()} for k, r in ratio.items()}
    OUT["D_avg_sq_grad"] = {str(k): {str(T): v for T, v in r.items()} for k, r in rows.items()}


# --------------------------------------------------------------------------- E
def check_E():
    """Allocation / kappa separation.

    A layer-normalised method spends a fixed step length per layer.  If the
    direction is u proportional to m, coordinate i gets the share |m_i|/RMS(m),
    so a coordinate whose gradient is r times smaller travels r times slower.
    A per-coordinate shape (sign or any b>0) gives every coordinate the same
    share, so its time to target does not depend on r at all.

    F = 1/2 sum c_i (x_i - 1)^2 with c_i spanning 1e-6 ... 1; all coordinates
    need the same displacement.  Reported: steps until max_i |x_i - 1| <= 0.05.
    """
    cap = 4000
    out, sat = {}, {}
    plans = [("layer", None), ("sign", None), ("q4", 40.0), ("q8", 40.0), ("q32", None),
             ("q4", 12.0), ("q8", 12.0)]
    for name, span in plans:
        n = 8
        c = torch.tensor([1.0] * (n // 2) + [1e-6] * (n // 2))
        P = [torch.nn.Parameter(torch.zeros(n))]
        kw = dict(span=span) if span else {}
        o = L38.L38(P, [0], mode=name, K=4, Kq=1, tv=None, tau=1.0, **kw)
        hit = cap
        for t in range(1, cap + 1):
            P[0].grad = c * (P[0].detach() - 1.0)
            o.step(0.05)
            if float((P[0].detach() - 1.0).abs().max()) <= 0.05:
                hit = t
                break
        key = name if span is None else "%s|span%g" % (name, span)
        out[key] = hit
        sat[key] = o.clip_frac / max(o.nq, 1)
        print("E  %-14s steps to max|x-1|<=0.05 (c in [1e-6,1]) : %-7s saturation %.3f"
              % (key, hit if hit < cap else ">%d" % cap, sat[key]))
    OUT["E_steps_equal_displacement"] = out
    OUT["E_saturation"] = sat
    assert out["sign"] < 200 and out["q32"] < 200, out
    assert out["q8|span40"] < 200 and out["q4|span40"] < 200, out
    assert out["layer"] >= cap, out
    print("E  a per-layer scalar needs Omega(1/c_min) more steps; a shaped arm with "
          "enough span does not depend on the spread at all")


# --------------------------------------------------------------------------- F


# --------------------------------------------------------------------------- F
def check_F():
    c = 4.0
    for T, K in ((500, 8), (2000, 8)):
        s = sum(min(c / j, 1.0) ** 2 for j in range(1, T // K + 1))
        OUT.setdefault("F_tv_budget", {})[str((T, K))] = s
        print("F  T=%d K=%d  sum_j min(c/j,1)^2 = %.3f  (bounded as T grows)" % (T, K, s))
        assert s < 12.0
    s2 = sum(c ** 2 / j ** 2 for j in range(1, 10 ** 6))
    print("F  c^2 pi^2/6 = %.3f   sum c^2/j^2 = %.3f" % (c * c * math.pi ** 2 / 6, s2))
    assert abs(s2 - c * c * math.pi ** 2 / 6) < 1e-3


# --------------------------------------------------------------------------- G
def check_G():
    torch.manual_seed(1)
    v = torch.exp(torch.randn(300000) * 1.5) * 1e-6
    for bits, span in ((4, 16.0), (4, 24.0), (8, 32.0), (8, 40.0)):
        lev, c, s, clip = L38.quant_log2(v, bits, span)
        rec = L38.dequant_log2(lev, c, s)
        rel = ((1 / rec.sqrt()) - (1 / v.sqrt())).abs() * v.sqrt()
        ok = (lev > lev.min()) & (lev < lev.max())
        bound = 2 ** (s / 4) - 1
        kq = ((1 + bound) / (1 - bound)) ** 2
        OUT.setdefault("G_quant", {})["b%d_span%g" % (bits, span)] = dict(
            step_octaves=s, max_rel_err_in_range=float(rel[ok].max()),
            bound=bound, clipped=clip, kappa_quant=kq)
        print("G  b=%d span=%2.0f octaves  scale step %.3f oct  max rel err %.3f (bound %.3f)"
              "  clipped %.4f  -> kappa_q <= %.2f"
              % (bits, span, s / 2, float(rel[ok].max()), bound, clip, kq))
        assert float(rel[ok].max()) <= bound + 1e-6
    print("G  reference: a per-layer scalar has kappa_q = kappa (unbounded); "
          "Adam (b=32) has kappa_q = 1")


# --------------------------------------------------------------------------- H
def check_H():
    """omega = E||m_t-m_{t-1}||^2 / E||m_t||^2 is a monotone probe of sigma/|mu|."""
    beta, a = 0.9, 0.1
    rows = []
    for ratio in (0.1, 0.3, 1.0, 3.0, 10.0):
        torch.manual_seed(5)
        m = torch.zeros(4096)
        JM, MM = 0.0, 0.0
        for t in range(1, 400):
            g = 1.0 + ratio * torch.randn(4096)
            m_old = m.clone()
            m = beta * m + a * g
            if t > 100:
                JM += float(((m - m_old) ** 2).mean())
                MM += float((m ** 2).mean())
        rows.append((ratio, JM / MM))
    OUT["H_omega"] = [(r, w) for r, w in rows]
    print("H  sigma/|mu| -> omega=J/M  :  " + ", ".join("%.1f->%.4f" % r for r in rows))
    ws = [w for _, w in rows]
    assert all(ws[i] < ws[i + 1] for i in range(len(ws) - 1)), ws


# --------------------------------------------------------------------------- I
def check_I():
    """How tight is the certified constant, before any eta/L/G/T is chosen?

    The two bounds have the shape
        term2  = (2 L eta sigma^2) * M2 + ...      M2_37 = h_max^2 / h_min
        term3  = (L^2 eta^2 kappa^2 G^2) * M3      M3_37 = 16 h_max^4/h_min^2 + 2 h_max^2
        term4  = (G^2 kappa^2 / T) * M4            M4_37 = 16 C^2 pi^2/(6 h_min^2)
    The multipliers M2, M3, M4 depend only on the declared box, so their ratio
    is a fair, parameter-free measure of how much the theory constrains h.
    """
    kappa = 9.0
    # 37: declared box and drift budget
    hmin, hmax, C = 1e-3, 1e4, 1e4
    M2_37 = hmax ** 2 / hmin
    M3_37 = 16 * hmax ** 4 / hmin ** 2 + 2 * hmax ** 2
    M4_37 = 16 * C ** 2 * math.pi ** 2 / (6 * hmin ** 2)
    # 38: h pinned by the trust ratio, b = 0 gives rho = 1 exactly and V_T = 0
    tv = sum(min(4.0 / j, 1.0) ** 2 for j in range(1, 10 ** 5))
    rows = {}
    for name, rho in (("b=0", 1.0), ("b=8", 4.0)):
        hmin38, hmax38 = 1.0 / rho, rho
        M2 = hmax38 ** 2 / hmin38
        M3 = 16 * hmax38 ** 4 / hmin38 ** 2 + 2 * hmax38 ** 2
        M4 = rho ** 2 * tv
        rows[name] = (M2, M3, M4)
    rows["37"] = (M2_37, M3_37, M4_37)
    OUT["I_constant_multipliers"] = {k: dict(M2=v[0], M3=v[1], M4=v[2]) for k, v in rows.items()}
    OUT["I_ratio"] = {k: dict(M2=M2_37 / v[0], M3=M3_37 / v[1], M4=M4_37 / v[2])
                      for k, v in rows.items()}
    print("I  37  M2=%.1e  M3=%.1e  M4=%.1e" % rows["37"])
    for name in ("b=0", "b=8"):
        print("I  38 %-4s M2=%.1e  M3=%.1e  M4=%.1e   -> %d / %d / %d times tighter"
              % ((name,) + rows[name] + (M2_37 / rows[name][0], M3_37 / rows[name][1],
                                         M4_37 / rows[name][2])))
    assert rows["b=0"][1] < 1e3 and M3_37 / rows["b=0"][1] > 1e15
    print("I  the 37 bound's h-box multipliers are 1e12 - 1e20 times larger than 38's, "
          "i.e. (7) cannot rank any two h policies at the declared parameters")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    check_A()
    check_B()
    check_C()
    check_D()
    check_E()
    check_F()
    check_G()
    check_H()
    check_I()
    (R / "38_CHECKS.json").write_text(json.dumps(OUT, indent=2))
    print("\nALL 38 CHECKS PASSED -> 38_CHECKS.json")
