"""Readable candidate 51 reference. Matrix axis statistics are FP32."""
import torch


class C51Slim(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=.9, rho=.999, eps2=1e-16,
                 wd=0., alpha=.5, tau=2.):
        super().__init__(params, dict(lr=lr, beta=beta, rho=rho, eps2=eps2,
                                      weight_decay=wd, alpha=alpha, tau=tau))

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for key in ('alpha', 'tau'):
                group.setdefault(key, self.defaults[key])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd, beta, rho, eps, alpha, tau = (
                group[key] for key in ('lr', 'weight_decay', 'beta', 'rho', 'eps2', 'alpha', 'tau')
            )
            if not (0 <= alpha <= 1 and tau > 0 and 0 <= beta < 1 and 0 <= rho < 1 and eps > 0):
                raise ValueError('Invalid C51 hyperparameters')
            for weight in group['params']:
                if weight.grad is None or not weight.numel():
                    continue
                if weight.dtype != torch.float32 or weight.grad.layout != torch.strided:
                    raise ValueError('C51Slim expects dense gradients and FP32 parameters')
                matrix = weight.ndim >= 2
                g = weight.grad.reshape(weight.shape[0], -1) if matrix else weight.grad
                st = self.state[weight]
                if not st:
                    st.update(t=0, m=torch.zeros_like(weight, memory_format=torch.contiguous_format))
                    if matrix:
                        st.update(r=g.new_zeros(g.size(0)), c=g.new_zeros(g.size(1)))
                    else:
                        st['n'] = torch.zeros_like(weight)
                m = st['m'].view_as(g)
                t = st['t'] + 1
                if t == 1:
                    square = g.square()
                    update = g * (square.maximum(square.mean()) + eps).rsqrt()
                else:
                    history = m / (1 - beta**(t-1))
                    correction = 1 - rho**(t-1)
                    if matrix:
                        row, col = st['r'] / correction, st['c'] / correction
                        signal = history.square()
                        vr = signal.mean(1) + row
                        vc = signal.mean(0) + col
                        variance = signal + row[:, None] * col[None, :] / row.mean().clamp_min(1e-30)
                        local_share = (1-alpha) / 2
                        row_gain = alpha / (vr.mean()+eps) + local_share / (vr+eps)
                        gain_sq = row_gain[:, None] + (local_share / (vc+eps))[None, :]
                    else:
                        variance = history.square() + st['n'] / correction
                        gain_sq = alpha / (variance.mean()+eps) + (1-alpha) / (variance+eps)
                    omega = (1-beta) / (1-beta**t)
                    innovation = (g-history) * (omega * gain_sq.sqrt())
                    update = history * (variance+eps).rsqrt() + innovation.clamp(-tau, tau)
                error_sq = (g-m).square()
                m.mul_(beta).add_(g, alpha=1-beta)
                payment = beta * (1-rho)
                if matrix:
                    st['r'].mul_(rho).add_(error_sq.mean(1), alpha=payment)
                    st['c'].mul_(rho).add_(error_sq.mean(0), alpha=payment)
                else:
                    st['n'].mul_(rho).add_(error_sq, alpha=payment)
                weight.mul_(1-lr*wd).add_(update.reshape_as(weight), alpha=-lr)
                st['t'] = t
        return loss
