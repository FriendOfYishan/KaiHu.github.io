"""Candidate 51 for contiguous FP32 CUDA tensors. Matrix axes are (shape[0], -1).

Matrix passes share temporary storage on one stream. Each partial tile stores
two sets of axis sums (history squared and raw prediction error squared).
Meta holds [R, vR, C, vC, mean(R), mean(vR)] from *before* the EMA update.
Only m, r, c (or m, n for vectors) and the step belong to a checkpoint.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partials(G, M, Rows, Cols, h, w, bc, first: tl.constexpr,
              X: tl.constexpr, Y: tl.constexpr):
    a, b = tl.program_id(0), tl.program_id(1)
    i = a * X + tl.arange(0, X)
    j = b * Y + tl.arange(0, Y)
    valid = (i[:, None] < h) & (j[None, :] < w)
    off = i[:, None] * w + j[None, :]
    g = tl.load(G + off, valid, 0)
    m = tl.load(M + off, valid, 0)
    e = g - m
    if first:
        signal = g
    else:
        signal = m / bc
    tile = a * tl.cdiv(w, Y) + b
    tl.store(Rows + tile * (2*X) + tl.arange(0,X), tl.sum(signal*signal,1))
    tl.store(Rows + tile * (2*X) + X + tl.arange(0,X), tl.sum(e*e,1))
    tl.store(Cols + tile * (2*Y) + tl.arange(0,Y), tl.sum(signal*signal,0))
    tl.store(Cols + tile * (2*Y) + Y + tl.arange(0,Y), tl.sum(e*e,0))


@triton.jit
def _marginals(Rows, Cols, R, C, Meta, h, w, rc, rho, payment,
               X:tl.constexpr, Y:tl.constexpr, RR:tl.constexpr, CC:tl.constexpr,
               L:tl.constexpr):
    pid = tl.program_id(0)
    nr, nc = tl.cdiv(h,X), tl.cdiv(w,Y)
    lanes = tl.arange(0,L)
    if pid < tl.cdiv(h,L):
        i = pid*L + lanes
        k = tl.arange(0,CC)
        addr = ((i[:,None]//X)*nc+k[None,:])*(2*X)+i[:,None]%X
        valid = (i[:,None]<h)&(k[None,:]<nc)
        old = tl.load(R+i,i<h,0)
        sig = tl.sum(tl.load(Rows+addr,valid,0),1)/w
        err = tl.sum(tl.load(Rows+addr+X,valid,0),1)/w
        corrected = old/rc
        tl.store(Meta+i,corrected,i<h)
        tl.store(Meta+h+i,corrected+sig,i<h)
        tl.store(R+i,rho*old+payment*err,i<h)
    else:
        j = (pid-tl.cdiv(h,L))*L + lanes
        kr = tl.arange(0,RR)
        col_addr = (kr[:,None]*nc+j[None,:]//Y)*(2*Y)+j[None,:]%Y
        col_valid = (kr[:,None]<nr)&(j[None,:]<w)
        old = tl.load(C+j,j<w,0)
        sig = tl.sum(tl.load(Cols+col_addr,col_valid,0),0)/h
        err = tl.sum(tl.load(Cols+col_addr+Y,col_valid,0),0)/h
        corrected = old/rc
        tl.store(Meta+2*h+j,corrected,j<w)
        tl.store(Meta+2*h+w+j,corrected+sig,j<w)
        tl.store(C+j,rho*old+payment*err,j<w)


@triton.jit
def _global(Meta, h, w, N:tl.constexpr):
    i = tl.arange(0,N)
    end = Meta+2*(h+w)
    tl.store(end,tl.sum(tl.load(Meta+i,i<h,0),0)/h)
    tl.store(end+1,tl.sum(tl.load(Meta+h+i,i<h,0),0)/h)


@triton.jit
def _apply(W,G,M,Meta,h,w,bc,omega,beta,lr,decay,alpha,tau,eps,
           first:tl.constexpr,X:tl.constexpr,Y:tl.constexpr, Sums=None,
           cache_next:tl.constexpr=False):
    i = tl.program_id(0)*X+tl.arange(0,X)
    j = tl.program_id(1)*Y+tl.arange(0,Y)
    valid = (i[:,None]<h)&(j[None,:]<w)
    off = i[:,None]*w+j[None,:]
    g = tl.load(G+off,valid,0)
    m = tl.load(M+off,valid,0)
    v0 = tl.load(Meta+2*(h+w)+1)
    if first:
        u = g*tl.rsqrt(tl.maximum(g*g,v0)+eps)
    else:
        mean_r = tl.maximum(tl.load(Meta+2*(h+w)),1.e-30)
        r = tl.load(Meta+i,i<h,0)
        c = tl.load(Meta+2*h+j,j<w,0)
        b = m/bc
        retained = b*tl.rsqrt(b*b+r[:,None]*c[None,:]/mean_r+eps)
        share = (1.-alpha)*0.5
        row = share/(tl.load(Meta+h+i,i<h,0)+eps)
        col = share/(tl.load(Meta+2*h+w+j,j<w,0)+eps)
        correction = omega*(g-b)*tl.sqrt(alpha/(v0+eps)+row[:,None]+col[None,:])
        u = retained+tl.maximum(-tau,tl.minimum(tau,correction))
    weight = tl.load(W+off,valid,0)
    tl.store(W+off,weight*decay-lr*u,valid)
    new_m = beta*m+(1-beta)*g
    tl.store(M+off,new_m,valid)
    if cache_next:
        new_sq = new_m*new_m
        err = g-m
        err_sq = err*err
        tl.atomic_add(Sums+i,tl.sum(new_sq,1),i<h,sem="relaxed")
        tl.atomic_add(Sums+h+i,tl.sum(err_sq,1),i<h,sem="relaxed")
        tl.atomic_add(Sums+2*h+j,tl.sum(new_sq,0),j<w,sem="relaxed")
        tl.atomic_add(Sums+2*h+w+j,tl.sum(err_sq,0),j<w,sem="relaxed")


@triton.jit
def _cache_next(Sums, R, C, Meta, h, w, bc_sq, rc, rho, payment,
                BLOCK:tl.constexpr):
    j = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row = j<h
    col = j<w
    row_m = tl.load(Sums+j,row,0)/w
    row_e = tl.load(Sums+h+j,row,0)/w
    col_m = tl.load(Sums+2*h+j,col,0)/h
    col_e = tl.load(Sums+2*h+w+j,col,0)/h
    r = rho*tl.load(R+j,row,0)+payment*row_e
    c = rho*tl.load(C+j,col,0)+payment*col_e
    tl.store(R+j,r,row)
    tl.store(C+j,c,col)
    tl.store(Meta+j,r/rc,row)
    tl.store(Meta+h+j,row_m/bc_sq+r/rc,row)
    tl.store(Meta+2*h+j,c/rc,col)
    tl.store(Meta+2*h+w+j,col_m/bc_sq+c/rc,col)
    tl.store(Sums+j,0.,row)
    tl.store(Sums+h+j,0.,row)
    tl.store(Sums+2*h+j,0.,col)
    tl.store(Sums+2*h+w+j,0.,col)


@triton.jit
def _vector(W,G,M,Nstate,n,bc,rc,omega,beta,rho,payment,lr,decay,alpha,tau,eps,
            first:tl.constexpr,BLOCK:tl.constexpr):
    i = tl.arange(0,BLOCK)
    g = tl.load(G+i,i<n,0)
    m = tl.load(M+i,i<n,0)
    old = tl.load(Nstate+i,i<n,0)
    if first:
        u = g*tl.rsqrt(tl.maximum(g*g,tl.sum(g*g,0)/n)+eps)
    else:
        b = m/bc
        v = b*b+old/rc
        inv = 1./(v+eps)
        k = tl.sqrt(alpha/(tl.sum(v,0)/n+eps)+(1.-alpha)*inv)
        u = b*tl.rsqrt(v+eps)+tl.maximum(-tau,tl.minimum(tau,omega*k*(g-b)))
    tl.store(W+i,tl.load(W+i,i<n,0)*decay-lr*u,i<n)
    tl.store(Nstate+i,rho*old+payment*(g-m)*(g-m),i<n)
    tl.store(M+i,beta*m+(1-beta)*g,i<n)


class C51Fused(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=.9, rho=.999, eps2=1e-16,
                 wd=0., alpha=.5, tau=2., single_pass=False):
        super().__init__(params, dict(lr=lr, beta=beta, rho=rho, eps2=eps2,
                                      weight_decay=wd, alpha=alpha, tau=tau))
        self._scratch = {}
        self.single_pass = single_pass
        self._next_meta = {}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._next_meta.clear()
        for group in self.param_groups:
            for key in ('alpha', 'tau'):
                group.setdefault(key, self.defaults[key])

    def _workspace(self, p, size):
        stream = torch.cuda.current_stream(p.device)
        key = (p.device, stream.cuda_stream)
        buf = self._scratch.get(key)
        if buf is None or buf.numel() < size:
            allocate = torch.zeros if self.single_pass else torch.empty
            buf = allocate(size, dtype=torch.float32, device=p.device)
            self._scratch[key] = buf
        return buf

    def _cached_matrix(self, p, g, st, h, w, bc, rc, omega, beta, rho,
                       payment, lr, decay, alpha, tau, eps, t):
        meta = self._next_meta.get(p)
        if meta is None:
            meta = p.new_empty(2*(h+w)+2)
            self._next_meta[p] = meta
            if t == 1:
                # Only the original first-step scale needs an extra gradient pass.
                meta[-1:].copy_(torch.linalg.vector_norm(g).square()/g.numel())
            else:
                b_sq = (st['m'].view(h, w)/bc).square()
                r, c = st['r']/rc, st['c']/rc
                meta[:h].copy_(r)
                meta[h:2*h].copy_(b_sq.mean(1)+r)
                meta[2*h:2*h+w].copy_(c)
                meta[2*h+w:2*(h+w)].copy_(b_sq.mean(0)+c)
                _global[(1,)](meta, h, w, triton.next_power_of_2(h))
        span = 2*(h+w)
        sums = self._workspace(p, span)[:span]
        # The previous post-update kernel clears its prefix for reuse.
        x, y = 16, 64
        _apply[(triton.cdiv(h,x), triton.cdiv(w,y))](
            p, g, st['m'], meta, h, w, bc, omega, beta, lr, decay,
            alpha, tau, eps, t == 1, x, y, sums, True)
        _cache_next[(triton.cdiv(max(h,w), 128),)](
            sums, st['r'], st['c'], meta, h, w, (1-beta**t)**2, 1-rho**t,
            rho, payment, 128)
        _global[(1,)](meta, h, w, triton.next_power_of_2(h))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta, rho, eps, lr, wd, alpha, tau = (
                group[k] for k in ('beta', 'rho', 'eps2', 'lr', 'weight_decay', 'alpha', 'tau')
            )
            if not (0 <= alpha <= 1 and tau > 0 and 0 <= beta < 1 and 0 <= rho < 1 and eps > 0):
                raise ValueError('invalid C51 gain, threshold, decay or epsilon')
            for p in group['params']:
                g = p.grad
                if g is None or not p.numel():
                    continue
                if not (p.is_cuda and p.is_contiguous() and p.dtype == torch.float32
                        and g.layout == torch.strided and g.is_contiguous() and g.dtype == p.dtype):
                    raise ValueError('C51Fused requires contiguous CUDA FP32 weights and gradients')
                vector = p.ndim < 2
                h = 1 if vector else p.shape[0]
                w = p.numel() // h
                if max(h, w) > 65536 or p.numel() > 2**31-1:
                    raise ValueError('C51Fused requires axes <= 65536 and 32-bit matrix offsets')
                st = self.state[p]
                if not st:
                    st.update(m=torch.zeros_like(p), t=0)
                    if vector:
                        st['n'] = torch.zeros_like(p)
                    else:
                        st.update(r=p.new_zeros(h), c=p.new_zeros(w))
                t = st['t'] + 1
                bc = 1 - beta**(t-1) if t > 1 else 1.
                rc = 1 - rho**(t-1) if t > 1 else 1.
                omega = (1-beta) / (1-beta**t)
                payment = (1-rho) * beta
                decay = 1 - lr*wd
                with torch.cuda.device(p.device):
                    if vector:
                        _vector[(1,)](p, g, st['m'], st['n'], w, bc, rc, omega, beta, rho,
                                      payment, lr, decay, alpha, tau, eps, t == 1, triton.next_power_of_2(w))
                    elif self.single_pass:
                        self._cached_matrix(p, g, st, h, w, bc, rc, omega, beta, rho,
                                            payment, lr, decay, alpha, tau, eps, t)
                    else:
                        x, y = 16, 64
                        nr, nc = triton.cdiv(h, x), triton.cdiv(w, y)
                        row_end = nr * nc * 2*x
                        col_end = row_end + nr * nc * 2*y
                        buf = self._workspace(p, col_end + 2*(h+w) + 2)
                        rows, cols, meta = buf[:row_end], buf[row_end:col_end], buf[col_end:]
                        _partials[(nr, nc)](g, st['m'], rows, cols, h, w, bc, t == 1, x, y)
                        lanes = 8
                        groups = triton.cdiv(h, lanes) + triton.cdiv(w, lanes)
                        _marginals[(groups,)](
                            rows, cols, st['r'], st['c'], meta, h, w, rc, rho, payment, x, y,
                            triton.next_power_of_2(nr), triton.next_power_of_2(nc), lanes)
                        _global[(1,)](meta, h, w, triton.next_power_of_2(h))
                        _apply[(nr, nc)](p, g, st['m'], meta, h, w, bc, omega, beta, lr, decay,
                                         alpha, tau, eps, t == 1, x, y)
                st['t'] = t
        return loss
