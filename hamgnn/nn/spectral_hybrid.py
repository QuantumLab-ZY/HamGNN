"""Spectral-hybrid optimizer for HamGNN.

Matrix parameters take a spectral update: a streaming rank-r basis of the
momentum is refreshed with one power iteration per step, and every
``period`` steps each singular direction of the update is reweighted by how
consistently the current gradient and a lagged gradient estimate (an
exponential moving average) point along it. Persistent directions grow and
noisy ones are damped; when reweighting gains nothing the update reduces to
an orthogonalized (Muon-like) step. All other parameters take an AdamW step.

The optimizer only uses the gradients exposed by PyTorch Lightning automatic
optimization; at HamGNN's usual batch size of one the gradient noise is
estimated from the temporal fluctuations of the gradient.
"""

import math
from typing import Dict, Iterable, Tuple

import torch
from torch.optim import Optimizer


DEFAULT_TEMPERATURE_GRID = (
    0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0,
)


def _rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.float().pow(2).mean().sqrt().to(dtype=x.dtype).clamp_min(eps)


def _normalize_columns(y: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    s = torch.linalg.vector_norm(y, dim=0).clamp_min(eps)
    return y / s.unsqueeze(0), s


def _standardize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shift to zero mean and scale to unit root-mean-square."""
    x = x.float()
    xc = x - x.mean()
    scale = xc.pow(2).mean().sqrt().clamp_min(eps)
    return (xc / scale).to(dtype=x.dtype)


def _normalize_temperature(t: torch.Tensor, tmin: float, tmax: float, eps: float) -> torch.Tensor:
    """Normalize to unit geometric mean and clip to [tmin, tmax] (twice)."""
    t = torch.nan_to_num(t.float(), nan=1.0, posinf=tmax, neginf=tmin).clamp_min(eps)
    t = (t / t.log().mean().exp()).clamp(tmin, tmax)
    t = (t / t.log().mean().exp()).clamp(tmin, tmax)
    return t


def _tube_project(w: torch.Tensor, delta: float, target_l2: float, eps: float, n_iter: int = 4) -> torch.Tensor:
    """Clamp weights to [1 - delta, 1 + delta] and rescale to ||w||_2 = target_l2."""
    lo = max(0.0, 1.0 - float(delta)) if math.isfinite(float(delta)) else 0.0
    hi = 1.0 + float(delta) if math.isfinite(float(delta)) else float("inf")
    w = w.float()
    for _ in range(max(1, n_iter)):
        w = w.clamp(lo, hi)
        w = w * (float(target_l2) / w.norm().clamp_min(eps))
    w = w.clamp(lo, hi)
    return w * (float(target_l2) / w.norm().clamp_min(eps))


def _weights_from_advantage(a: torch.Tensor, temperature: float, delta: float, eps: float) -> torch.Tensor:
    """Softmax weights over the standardized per-direction advantage."""
    r = a.numel()
    if r == 0 or temperature <= 0.0:
        return torch.ones_like(a)
    z = _standardize(a, eps)
    prob = torch.softmax(float(temperature) * z, dim=0)
    w = torch.sqrt(prob * float(r))
    return _tube_project(w, delta=delta, target_l2=math.sqrt(float(r)), eps=eps).to(dtype=a.dtype)


class SpectralHybrid(Optimizer):
    """Lightning-compatible Spectral-hybrid optimizer.

    Parameter groups with ``use_spectral=True`` (the default) take the
    spectral update; tensors in such a group with fewer than two dimensions
    (or a matrix side shorter than two) fall back to AdamW. Groups with
    ``use_spectral=False`` take AdamW. Tensors with more than two dimensions
    are flattened to ``(shape[0], -1)``.

    Parameters
    ----------
    params : iterable
        Parameters or parameter groups.
    lr : float
        Default learning rate.
    momentum : float
        Momentum coefficient of the spectral update.
    grad_ema : float
        Coefficient of the lagged gradient average used as the replica.
    rank : int
        Rank of the streaming basis; ``0`` uses the full rank.
    power_iters : int
        Power iterations per step for the streaming basis.
    period : int
        The spectral reweighting is applied every ``period`` steps; the other
        steps use unit weights.
    metric : bool
        Use the metric variant, which rescales rows and columns by
        temperatures derived from the signal-to-noise ratio of the gradient.
    betas, eps, weight_decay
        AdamW settings (``eps`` also regularizes the spectral update).
        Weight decay is decoupled and applied on both routes.
    lambda_noise, lambda_consistency : float
        Weights of the noise and sign-consistency terms of the curvature
        proxy ``B``.
    tube_delta : float
        Half-width of the tube around unit weight.
    temperature_grid : tuple of float
        Candidate softmax temperatures; the best one under the cross-fitted
        surrogate is selected each refresh, and ``0`` means unit weights.
    lambda_diff, lambda_uncert, lambda_tube : float
        Penalties of the cross-fitted surrogate (diffusion, uncertainty and
        distance from unit weight).
    soft_lr_threshold : float
        At learning rates at or above this value the selected temperature is
        shrunk smoothly instead of accepted or rejected (hard selection).
    hard_threshold, soft_threshold, soft_tau : float
        Acceptance threshold of the hard selection, and centre and width of
        the sigmoid shrinkage of the soft selection.
    metric_alpha_row, metric_alpha_col, metric_lambda_noise, metric_tmin, metric_tmax
        Settings of the metric variant.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        grad_ema: float = 0.95,
        rank: int = 32,
        power_iters: int = 1,
        period: int = 4,
        metric: bool = False,
        betas: Tuple[float, float] = (0.99, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lambda_noise: float = 2.0,
        lambda_consistency: float = 0.25,
        tube_delta: float = 0.35,
        temperature_grid: Tuple[float, ...] = DEFAULT_TEMPERATURE_GRID,
        lambda_diff: float = 0.5,
        lambda_uncert: float = 0.5,
        lambda_tube: float = 0.05,
        soft_lr_threshold: float = 0.05,
        hard_threshold: float = 0.0,
        soft_threshold: float = 0.0,
        soft_tau: float = 0.01,
        metric_alpha_row: float = 0.25,
        metric_alpha_col: float = 0.25,
        metric_lambda_noise: float = 1.0,
        metric_tmin: float = 0.5,
        metric_tmax: float = 2.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, use_spectral=True)
        super().__init__(params, defaults)
        self.momentum = float(momentum)
        self.grad_ema = float(grad_ema)
        self.rank = int(rank)
        self.power_iters = int(power_iters)
        self.period = int(period)
        self.metric = bool(metric)
        self.eps = float(eps)
        self.lambda_noise = float(lambda_noise)
        self.lambda_consistency = float(lambda_consistency)
        self.tube_delta = float(tube_delta)
        self.temperature_grid = tuple(float(t) for t in temperature_grid)
        self.lambda_diff = float(lambda_diff)
        self.lambda_uncert = float(lambda_uncert)
        self.lambda_tube = float(lambda_tube)
        self.soft_lr_threshold = float(soft_lr_threshold)
        self.hard_threshold = float(hard_threshold)
        self.soft_threshold = float(soft_threshold)
        self.soft_tau = float(soft_tau)
        self.metric_alpha_row = float(metric_alpha_row)
        self.metric_alpha_col = float(metric_alpha_col)
        self.metric_lambda_noise = float(metric_lambda_noise)
        self.metric_tmin = float(metric_tmin)
        self.metric_tmax = float(metric_tmax)
        self._global_step = 0

    def state_dict(self):
        d = super().state_dict()
        d['_global_step'] = self._global_step
        return d

    def load_state_dict(self, state_dict):
        self._global_step = state_dict.get('_global_step', 0)
        optimizer_state = {
            key: value for key, value in state_dict.items() if key != '_global_step'
        }
        super().load_state_dict(optimizer_state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1
        for group in self.param_groups:
            lr = float(group["lr"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("SpectralHybrid does not support sparse gradients.")
                if not group.get("use_spectral", True):
                    self._adamw_step(p, g, group)
                    continue
                if not torch.is_floating_point(g):
                    continue
                if p.ndim < 2 or min(p.shape[0], int(p.numel() // p.shape[0])) < 2:
                    self._adamw_step(p, g, group)
                    continue
                update = self._spectral_update(p, g, lr)
                weight_decay = group["weight_decay"]
                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update, alpha=-lr)
        return loss

    def _adamw_step(self, p: torch.Tensor, g: torch.Tensor, group: Dict) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        state = self.state[p]
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] = 0
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        state["step"] += 1
        exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
        if weight_decay:
            p.mul_(1.0 - lr * weight_decay)
        p.addcdiv_(exp_avg, denom, value=-step_size)

    def _spectral_update(self, p: torch.Tensor, g: torch.Tensor, lr: float) -> torch.Tensor:
        eps = self.eps
        state = self.state[p]
        n = int(p.shape[0])
        m = int(p.numel() // n)
        gmat = g.reshape(n, m)
        if "momentum" not in state or state["momentum"].shape != gmat.shape:
            state["momentum"] = torch.zeros_like(gmat)
            state["grad_ema"] = gmat.detach().clone()
            replica = gmat.detach().clone()
        else:
            replica = state["grad_ema"].detach().clone()
        momentum = state["momentum"]
        momentum.mul_(self.momentum).add_(gmat, alpha=1.0 - self.momentum)

        # The lagged gradient average is the replica of the current gradient;
        # its difference from the gradient (the innovation) estimates the noise.
        innovation = gmat - replica
        state["grad_ema"].mul_(self.grad_ema).add_(gmat, alpha=1.0 - self.grad_ema)

        if self.metric:
            sqrt_tr, sqrt_tc = self._metric_temperatures(state, gmat, innovation)
            m_work = sqrt_tr[:, None] * momentum * sqrt_tc[None, :]
            g_work = sqrt_tr[:, None] * gmat * sqrt_tc[None, :]
            r_work = sqrt_tr[:, None] * replica * sqrt_tc[None, :]
        else:
            m_work, g_work, r_work = momentum, gmat, replica

        U, sigma, V = self._streaming_basis(state, m_work)
        if self.period <= 1 or self._global_step % self.period == 0:
            w = self._spectral_weights(U, V, g_work, r_work, lr)
        else:
            w = torch.ones_like(sigma)

        update = (U * w.unsqueeze(0)).matmul(V.t())
        if self.metric:
            update = sqrt_tr[:, None] * update * sqrt_tc[None, :]
        target_rms = math.sqrt(float(min(n, m)) / float(n * m))
        update = update * (target_rms / float(_rms(update, eps).item()))
        return update.reshape(p.shape)

    def _metric_temperatures(self, state: Dict, g: torch.Tensor, innovation: torch.Tensor):
        eps = self.eps
        n, m = g.shape
        if "row_signal" not in state or state["row_signal"].numel() != n:
            state["row_signal"] = torch.zeros(n, device=g.device, dtype=torch.float32)
            state["row_noise"] = torch.zeros(n, device=g.device, dtype=torch.float32)
        if "col_signal" not in state or state["col_signal"].numel() != m:
            state["col_signal"] = torch.zeros(m, device=g.device, dtype=torch.float32)
            state["col_noise"] = torch.zeros(m, device=g.device, dtype=torch.float32)
        for key in ("row_signal", "row_noise", "col_signal", "col_noise"):
            # load_state_dict casts state to the parameter dtype; keep float32.
            if state[key].dtype != torch.float32:
                state[key] = state[key].float()
        b = self.grad_ema
        state["row_signal"].mul_(b).add_(g.float().pow(2).sum(dim=1), alpha=1 - b)
        state["col_signal"].mul_(b).add_(g.float().pow(2).sum(dim=0), alpha=1 - b)
        state["row_noise"].mul_(b).add_(innovation.float().pow(2).sum(dim=1), alpha=1 - b)
        state["col_noise"].mul_(b).add_(innovation.float().pow(2).sum(dim=0), alpha=1 - b)
        lam = self.metric_lambda_noise
        row_rel = (state["row_signal"] + eps) / (state["row_signal"] + lam * state["row_noise"] + eps)
        col_rel = (state["col_signal"] + eps) / (state["col_signal"] + lam * state["col_noise"] + eps)
        tr = _normalize_temperature(
            (row_rel + eps).pow(self.metric_alpha_row), self.metric_tmin, self.metric_tmax, eps
        )
        tc = _normalize_temperature(
            (col_rel + eps).pow(self.metric_alpha_col), self.metric_tmin, self.metric_tmax, eps
        )
        return tr.sqrt().to(dtype=g.dtype), tc.sqrt().to(dtype=g.dtype)

    def _streaming_basis(self, state: Dict, A: torch.Tensor):
        """Return U, sigma, V with A V = U diag(sigma) for a streaming basis V."""
        n, m = A.shape
        if n >= m:
            return self._streaming_basis_tall(state, A)
        # For A^T V' = U' diag(sigma): A ~ V' diag(sigma) U'^T.
        U_t, sigma, V_t = self._streaming_basis_tall(state, A.t().contiguous())
        return V_t, sigma, U_t

    def _streaming_basis_tall(self, state: Dict, A: torch.Tensor):
        n, m = A.shape
        r = m if self.rank <= 0 else min(self.rank, m, n)
        V = state.get("basis")
        if V is None or tuple(V.shape) != (m, r) or V.device != A.device or V.dtype != A.dtype:
            if r == m:
                V = torch.eye(m, r, device=A.device, dtype=A.dtype)
            else:
                V = torch.randn(m, r, device=A.device, dtype=A.dtype) / math.sqrt(max(m, 1))
                V, _ = torch.linalg.qr(V, mode="reduced")
        for _ in range(max(1, self.power_iters)):
            V, _ = torch.linalg.qr(A.t().matmul(A.matmul(V)), mode="reduced")
        state["basis"] = V
        U, sigma = _normalize_columns(A.matmul(V), self.eps)
        return U, sigma, V

    def _spectral_weights(self, U: torch.Tensor, V: torch.Tensor, G: torch.Tensor,
                          G_replica: torch.Tensor, lr: float) -> torch.Tensor:
        """Select per-direction weights with a cross-fitted surrogate."""
        eps = self.eps
        g1 = (U * G.matmul(V)).sum(dim=0).float()
        g2 = (U * G_replica.matmul(V)).sum(dim=0).float()
        r = g1.numel()
        if r == 0:
            return torch.ones(0, device=G.device, dtype=G.dtype)

        nu = 0.25 * (g1 - g2).pow(2)
        consistency = 2.0 * g1 * g2 / (g1.pow(2) + g2.pow(2) + eps)
        B = (1.0 + self.lambda_noise * nu + self.lambda_consistency * (1.0 - consistency)).clamp_min(eps)
        lr = float(lr)
        a1 = lr * g1 - (lr ** 2) * B
        a2 = lr * g2 - (lr ** 2) * B
        is_soft = lr >= self.soft_lr_threshold
        delta = self.tube_delta
        threshold = self.soft_threshold if is_soft else self.hard_threshold
        ones = torch.ones(r, device=G.device, dtype=torch.float32)
        zero = torch.tensor(0.0, device=G.device)

        def surrogate(gproj, w):
            return lr * (gproj * w).sum() - 0.5 * (lr ** 2) * (B * w.pow(2)).sum()

        def score(temperature):
            if temperature <= 0.0:
                return zero
            w1 = _weights_from_advantage(a1, temperature, delta, eps)
            w2 = _weights_from_advantage(a2, temperature, delta, eps)
            gain_2 = surrogate(g2, w1) - surrogate(g2, ones)
            gain_1 = surrogate(g1, w2) - surrogate(g1, ones)
            gain = 0.5 * (gain_1 + gain_2)
            nu_sum = nu.sum().clamp_min(eps)
            diff = lr * (0.5 * (
                (nu * (w1.pow(2) - 1.0).pow(2)).sum() / nu_sum
                + (nu * (w2.pow(2) - 1.0).pow(2)).sum() / nu_sum
            ))
            uncert = (gain_1 - gain_2).abs()
            tube = lr * 0.5 * ((w1 - 1.0).pow(2).mean() + (w2 - 1.0).pow(2).mean())
            return gain - self.lambda_diff * diff - self.lambda_uncert * uncert - self.lambda_tube * tube

        best_phi = zero
        best_temperature = 0.0
        for temperature in self.temperature_grid:
            phi = score(temperature)
            t_eff = temperature
            if is_soft and temperature > 0.0:
                if phi <= 0.0:
                    phi, t_eff = zero, 0.0
                else:
                    shrink = torch.sigmoid((phi - threshold) / max(self.soft_tau, eps))
                    t_eff = temperature * float(shrink.item())
                    phi = score(t_eff)
            elif (not is_soft) and phi <= threshold:
                phi, t_eff = zero, 0.0
            if float(phi.item()) > float(best_phi.item()):
                best_phi = phi
                best_temperature = t_eff

        a = 0.5 * (a1 + a2)
        return _weights_from_advantage(a, best_temperature, delta, eps).to(dtype=G.dtype)
