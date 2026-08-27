"""Muon optimizer utilities for HamGNN."""

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of a matrix with Newton-Schulz."""
    assert grad.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    orth = grad.bfloat16() if grad.is_cuda else grad.float()
    if grad.size(-2) > grad.size(-1):
        orth = orth.mT

    orth = orth / (orth.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = orth @ orth.mT
        poly = b * gram + c * gram @ gram
        orth = a * orth + poly @ orth

    if grad.size(-2) > grad.size(-1):
        orth = orth.mT
    return orth.to(grad.dtype)


class MuonAdamW(Optimizer):
    """Lightning-compatible hybrid Muon + AdamW optimizer."""

    def __init__(
        self,
        params,
        lr=5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        momentum=0.95,
        ns_steps=5,
        use_muon=False,
        eps=1e-8,
        amsgrad=False,
        momentum_warmup_steps=0,
        momentum_start=0.85,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            use_muon=use_muon,
            eps=eps,
            amsgrad=amsgrad,
            momentum_warmup_steps=momentum_warmup_steps,
            momentum_start=momentum_start,
        )
        super().__init__(params, defaults)
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
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group):
        lr = group["lr"]
        target_momentum = group["momentum"]
        ns_steps = group["ns_steps"]
        weight_decay = group["weight_decay"]

        warmup_steps = group.get("momentum_warmup_steps", 0)
        momentum_start = group.get("momentum_start", 0.85)
        if warmup_steps > 0 and self._global_step <= warmup_steps:
            frac = self._global_step / warmup_steps
            momentum = momentum_start + frac * (target_momentum - momentum_start)
        else:
            momentum = target_momentum

        for param in group["params"]:
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)

            buf = state["momentum_buffer"]
            buf.lerp_(grad, 1 - momentum)
            update = torch.lerp(grad, buf, momentum)

            original_shape = update.shape
            if update.ndim == 4:
                update = update.view(update.size(0), -1)

            if update.ndim >= 2:
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5

            if weight_decay > 0:
                param.mul_(1 - lr * weight_decay)
            param.add_(update.reshape(original_shape), alpha=-lr)

    def _adamw_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]
        amsgrad = group.get("amsgrad", False)

        for param in group["params"]:
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(param)

            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if weight_decay > 0:
                param.mul_(1 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            if amsgrad:
                max_exp_avg_sq = state["max_exp_avg_sq"]
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                second_moment = max_exp_avg_sq
            else:
                second_moment = exp_avg_sq
            denom = (second_moment.sqrt() / bias_correction2**0.5).add_(eps)
            step_size = lr / bias_correction1
            param.addcdiv_(exp_avg, denom, value=-step_size)


def classify_params_for_muon(model):
    """Split trainable parameters into Muon-friendly and AdamW groups."""
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not any(
            token in name.lower()
            for token in ("embedding", "radial", "basis", "output_module", "norm", "bias")
        ):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params
