# Copyright (c) 2021-2026 HamGNN Team
# SPDX-License-Identifier: GPL-3.0-only

"""Tensor-product backend abstraction with safe e3nn fallback."""

from __future__ import annotations

import importlib
import os
import time
import math
import traceback
import warnings
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch
from e3nn import o3
from torch import nn

_DEFAULT_TP_BACKEND = "e3nn"
_OPENEQ_MAX_WEIGHT_NUMEL = None
_OPENEQ_MAX_INSTRUCTION_COUNT = None
_OPENEQ_SPLIT_MAX_WEIGHT_NUMEL = None
_WARNED_BACKENDS = set()
_OPENEQ_TP_CACHE = {}
_OPENEQ_TP_CACHE_HITS = 0
_OPENEQ_TP_CACHE_MISSES = 0
_OPENEQ_IMPORT_ATTEMPTS = 0
_OPENEQ_IMPORT_TOTAL_S = 0.0
_TP_BACKEND_EVENT_LIMIT = 128
_TP_BACKEND_EVENTS = []
_TP_BACKEND_BUILD_COUNTS = {}
_OPENEQ_AUTOCAST_COMPAT_SHIM_APPLIED = False


def _strict_tp_backend_enabled() -> bool:
    return os.environ.get("HAMGNN_STRICT_TP_BACKEND", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _openeq_debug_enabled() -> bool:
    return os.environ.get("HAMGNN_OEQ_DEBUG_TIMINGS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _openeq_debug_print(*parts: Any) -> None:
    if _openeq_debug_enabled():
        print("[openeq debug]", *parts)


def _openeq_traceback_enabled() -> bool:
    return os.environ.get("HAMGNN_OEQ_DEBUG_TRACEBACK", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_openeq_debug_counters() -> dict:
    """Expose lightweight OpenEquivariance counters for probe scripts."""
    return {
        "import_attempts": _OPENEQ_IMPORT_ATTEMPTS,
        "import_total_s": _OPENEQ_IMPORT_TOTAL_S,
        "cache_hits": _OPENEQ_TP_CACHE_HITS,
        "cache_misses": _OPENEQ_TP_CACHE_MISSES,
        "cache_size": len(_OPENEQ_TP_CACHE),
        "autocast_compat_shim": _OPENEQ_AUTOCAST_COMPAT_SHIM_APPLIED,
    }


def reset_tp_backend_debug_state() -> None:
    """Reset backend telemetry between independent benchmark/profiler runs."""
    global _OPENEQ_TP_CACHE_HITS, _OPENEQ_TP_CACHE_MISSES
    global _OPENEQ_IMPORT_ATTEMPTS, _OPENEQ_IMPORT_TOTAL_S
    _OPENEQ_TP_CACHE_HITS = 0
    _OPENEQ_TP_CACHE_MISSES = 0
    _OPENEQ_IMPORT_ATTEMPTS = 0
    _OPENEQ_IMPORT_TOTAL_S = 0.0
    _TP_BACKEND_EVENTS.clear()
    _TP_BACKEND_BUILD_COUNTS.clear()


def _record_tp_backend_event(
    requested_backend: str,
    actual_backend: str,
    *,
    reason: Optional[str] = None,
    weight_numel: Optional[int] = None,
    instruction_count: Optional[int] = None,
    module_kind: str = "tensor_product",
) -> None:
    key = f"{module_kind}:{requested_backend}->{actual_backend}"
    _TP_BACKEND_BUILD_COUNTS[key] = _TP_BACKEND_BUILD_COUNTS.get(key, 0) + 1
    event = {
        "module_kind": module_kind,
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "reason": reason,
        "weight_numel": None if weight_numel is None else int(weight_numel),
        "instruction_count": None if instruction_count is None else int(instruction_count),
    }
    _TP_BACKEND_EVENTS.append(event)
    if len(_TP_BACKEND_EVENTS) > _TP_BACKEND_EVENT_LIMIT:
        del _TP_BACKEND_EVENTS[:-_TP_BACKEND_EVENT_LIMIT]


def get_tp_backend_status() -> dict:
    """Return JSON-friendly status for benchmark and profiler outputs."""
    fallback_events = [
        event for event in _TP_BACKEND_EVENTS
        if event["requested_backend"] != event["actual_backend"]
    ]
    return {
        "requested_default_backend": _DEFAULT_TP_BACKEND,
        "openeq_max_weight_numel": _OPENEQ_MAX_WEIGHT_NUMEL,
        "openeq_max_instruction_count": _OPENEQ_MAX_INSTRUCTION_COUNT,
        "openeq_split_max_weight_numel": _OPENEQ_SPLIT_MAX_WEIGHT_NUMEL,
        "openeq_debug_counters": get_openeq_debug_counters(),
        "build_counts": dict(sorted(_TP_BACKEND_BUILD_COUNTS.items())),
        "fallback_event_count": len(fallback_events),
        "fallback_events": fallback_events,
        "recent_events": list(_TP_BACKEND_EVENTS),
    }


def _normalize_optional_limit(value: Any) -> Optional[int]:
    if value in (None, "", False):
        return None
    normalized = int(value)
    if normalized <= 0:
        return None
    return normalized


def set_default_tp_backend(
    name: str,
    *,
    openeq_max_weight_numel: Optional[int] = None,
    openeq_max_instruction_count: Optional[int] = None,
    openeq_split_max_weight_numel: Optional[int] = None,
) -> None:
    """Set the default tensor-product backend used by HamGNN modules."""
    global _DEFAULT_TP_BACKEND, _OPENEQ_MAX_WEIGHT_NUMEL, _OPENEQ_MAX_INSTRUCTION_COUNT
    global _OPENEQ_SPLIT_MAX_WEIGHT_NUMEL
    _DEFAULT_TP_BACKEND = (name or "e3nn").lower()
    _OPENEQ_MAX_WEIGHT_NUMEL = _normalize_optional_limit(openeq_max_weight_numel)
    _OPENEQ_MAX_INSTRUCTION_COUNT = _normalize_optional_limit(openeq_max_instruction_count)
    _OPENEQ_SPLIT_MAX_WEIGHT_NUMEL = _normalize_optional_limit(
        openeq_split_max_weight_numel
    )


def configure_default_tp_backend(config: Any) -> None:
    """Initialize the default TP backend from a HamGNN representation config."""
    hamgnn_pre = getattr(config, "HamGNN_pre", config)
    set_default_tp_backend(
        getattr(hamgnn_pre, "tp_backend", "e3nn"),
        openeq_max_weight_numel=getattr(
            hamgnn_pre, "tp_backend_max_weight_numel", None),
        openeq_max_instruction_count=getattr(
            hamgnn_pre, "tp_backend_max_instruction_count", None),
        openeq_split_max_weight_numel=getattr(
            hamgnn_pre, "tp_backend_split_max_weight_numel", None),
    )


def get_default_tp_backend() -> str:
    """Return the currently configured tensor-product backend."""
    return _DEFAULT_TP_BACKEND


def _warn_backend_fallback(backend: str, reason: str) -> None:
    cache_key = (backend, reason)
    if cache_key in _WARNED_BACKENDS:
        return
    _WARNED_BACKENDS.add(cache_key)
    if backend != "e3nn" and _strict_tp_backend_enabled():
        raise RuntimeError(
            f"Strict tensor-product backend requested for backend={backend!r}, "
            f"but HamGNN would fall back to e3nn: {reason}"
        )
    warnings.warn(
        f"Falling back to e3nn tensor products for backend={backend!r}: {reason}",
        RuntimeWarning,
        stacklevel=2,
    )


def _instruction_mode(instruction: Any) -> str:
    return getattr(instruction, "connection_mode", instruction[3])


def _instruction_has_weight(instruction: Any) -> bool:
    return bool(getattr(instruction, "has_weight", instruction[4]))


def _instruction_cache_tuple(instruction: Any) -> Tuple[Any, ...]:
    return (
        getattr(instruction, "i_in1", instruction[0]),
        getattr(instruction, "i_in2", instruction[1]),
        getattr(instruction, "i_out", instruction[2]),
        _instruction_mode(instruction),
        _instruction_has_weight(instruction),
        getattr(instruction, "path_weight", instruction[5] if len(instruction) > 5 else None),
    )


def _openeq_cache_key(
    irreps_in1: Any,
    irreps_in2: Any,
    irreps_out: Any,
    instructions: Sequence[Any],
    problem_kwargs: dict,
) -> Tuple[Any, ...]:
    return (
        str(o3.Irreps(irreps_in1)),
        str(o3.Irreps(irreps_in2)),
        str(o3.Irreps(irreps_out)),
        tuple(_instruction_cache_tuple(inst) for inst in instructions),
        tuple(sorted(problem_kwargs.items())),
    )


def _normalize_tp_call(args: Tuple[Any, ...], kwargs: dict) -> Tuple[Any, Any, Any, Sequence[Any]]:
    if len(args) < 3:
        raise ValueError("tensor product backend expects at least irreps_in1, irreps_in2, irreps_out")
    if len(args) > 4:
        raise ValueError("unsupported positional tensor-product arguments beyond instructions")
    irreps_in1, irreps_in2, irreps_out = args[:3]
    instructions = kwargs.get("instructions")
    if len(args) == 4:
        if instructions is not None:
            raise ValueError("instructions provided both positionally and by keyword")
        instructions = args[3]
    if instructions is None:
        raise ValueError("OpenEquivariance adapter requires explicit tensor-product instructions")
    return irreps_in1, irreps_in2, irreps_out, instructions


def _ensure_openeq_torch_compat() -> None:
    """Apply compatibility shims before importing OpenEquivariance.

    OEQ 0.6.5 calls ``torch.library.register_autocast()`` at module-level
    (``TensorProduct.py:259``) to register float32 autocast policies for its
    custom CUDA ops.  ``register_autocast`` was added in torch 2.7; on older
    versions the import crashes with ``AttributeError``.

    This shim injects a no-op stub when the function is absent so that
    ``import openequivariance`` succeeds.  The only consequence is that
    autocast won't auto-promote OEQ ops to float32, which is acceptable
    because HamGNN runs in precision=32 for CrI3 training.
    """
    global _OPENEQ_AUTOCAST_COMPAT_SHIM_APPLIED
    if _OPENEQ_AUTOCAST_COMPAT_SHIM_APPLIED:
        return
    try:
        import torch.library as _tl
        if not hasattr(_tl, "register_autocast"):
            _tl.register_autocast = lambda *_args, **_kwargs: None
            _openeq_debug_print("patched torch.library.register_autocast stub")
        _OPENEQ_AUTOCAST_COMPAT_SHIM_APPLIED = True
        _openeq_debug_print("autocast compat shim applied")
    except Exception as exc:
        _openeq_debug_print("autocast compat shim failed:", exc)
        _OPENEQ_AUTOCAST_COMPAT_SHIM_APPLIED = True


def _openeq_target_forward_max_batch_from_env() -> Optional[int]:
    """Read ``HAMGNN_OEQ_TARGET_FORWARD_MAX_BATCH`` from environment."""
    raw = os.environ.get("HAMGNN_OEQ_TARGET_FORWARD_MAX_BATCH", "")
    if not raw:
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except ValueError:
        return None


def _openeq_split_max_weight_numel_from_env() -> Optional[int]:
    """Read ``HAMGNN_OEQ_SPLIT_MAX_WEIGHT_NUMEL`` from environment."""
    raw = os.environ.get("HAMGNN_OEQ_SPLIT_MAX_WEIGHT_NUMEL", "")
    if not raw:
        return _OPENEQ_SPLIT_MAX_WEIGHT_NUMEL
    try:
        val = int(raw)
        return val if val > 0 else None
    except ValueError:
        return _OPENEQ_SPLIT_MAX_WEIGHT_NUMEL


def _load_openequivariance():
    global _OPENEQ_IMPORT_ATTEMPTS, _OPENEQ_IMPORT_TOTAL_S
    _ensure_openeq_torch_compat()
    start = time.perf_counter()
    _openeq_debug_print("import start")
    module = importlib.import_module("openequivariance")
    duration = time.perf_counter() - start
    _OPENEQ_IMPORT_ATTEMPTS += 1
    _OPENEQ_IMPORT_TOTAL_S += duration
    _openeq_debug_print("import done", f"{duration:.3f}s")
    return module


def _select_openeq_threshold_reason(weight_numel: int, instruction_count: int) -> Optional[str]:
    if (
        _OPENEQ_MAX_WEIGHT_NUMEL is not None
        and weight_numel > _OPENEQ_MAX_WEIGHT_NUMEL
    ):
        return (
            "weight_numel="
            f"{weight_numel} exceeds openeq_max_weight_numel={_OPENEQ_MAX_WEIGHT_NUMEL}"
        )
    if (
        _OPENEQ_MAX_INSTRUCTION_COUNT is not None
        and instruction_count > _OPENEQ_MAX_INSTRUCTION_COUNT
    ):
        return (
            "instruction_count="
            f"{instruction_count} exceeds "
            f"openeq_max_instruction_count={_OPENEQ_MAX_INSTRUCTION_COUNT}"
        )
    return None


def _drop_split_sub_tp_state_dict_hook(
    module: nn.Module,
    destination: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict,
) -> None:
    if hasattr(module, "_drop_sub_tp_state_dict_keys"):
        module._drop_sub_tp_state_dict_keys(destination, prefix)


def _drop_split_sub_tp_load_state_dict_pre_hook(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    if hasattr(module, "_drop_sub_tp_state_dict_keys"):
        module._drop_sub_tp_state_dict_keys(state_dict, prefix)


def _drop_split_sub_tp_load_state_dict_post_hook(
    module: nn.Module,
    incompatible_keys: Any,
) -> None:
    if hasattr(module, "_drop_sub_tp_load_state_keys"):
        module._drop_sub_tp_load_state_keys(incompatible_keys)


class OpenEquivarianceTensorProduct(nn.Module):
    """Drop-in TP adapter that preserves e3nn state_dict semantics."""

    def __init__(self, *args, _reference_tp: Optional[o3.TensorProduct] = None, **kwargs):
        super().__init__()
        if len(args) > 4:
            raise ValueError("OpenEquivariance adapter only supports instructions as the 4th positional arg")

        reference_tp = _reference_tp or o3.TensorProduct(*args, **kwargs)
        irreps_in1, irreps_in2, irreps_out, instructions = _normalize_tp_call(args, kwargs)
        modes = {_instruction_mode(inst) for inst in instructions}
        if len(modes) != 1:
            raise ValueError(f"mixed connection modes are unsupported: {sorted(modes)}")
        if not modes.issubset({"uvu", "uvw"}):
            raise ValueError(f"unsupported connection modes: {sorted(modes)}")
        if not all(_instruction_has_weight(inst) for inst in instructions):
            raise ValueError("OpenEquivariance adapter only supports trainable instructions")

        oeq = _load_openequivariance()
        self.irreps_in1 = reference_tp.irreps_in1
        self.irreps_in2 = reference_tp.irreps_in2
        self.irreps_out = reference_tp.irreps_out
        self.instructions = reference_tp.instructions
        self.shared_weights = reference_tp.shared_weights
        self.internal_weights = reference_tp.internal_weights
        self.weight_numel = reference_tp.weight_numel
        self._mode = next(iter(modes))
        self._needs_weight_reorder = not (
            self._mode == "uvu" and all(mul_ir.mul == 1 for mul_ir in self.irreps_in2)
        )

        problem_kwargs = {}
        for key in (
            "in1_var",
            "in2_var",
            "out_var",
            "irrep_normalization",
            "path_normalization",
        ):
            if key in kwargs and kwargs[key] is not None:
                problem_kwargs[key] = kwargs[key]
        problem_kwargs["shared_weights"] = self.shared_weights
        problem_kwargs["internal_weights"] = False
        cache_key = _openeq_cache_key(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            problem_kwargs,
        )
        global _OPENEQ_TP_CACHE_HITS, _OPENEQ_TP_CACHE_MISSES
        cached_tp = _OPENEQ_TP_CACHE.get(cache_key)
        if cached_tp is None:
            _OPENEQ_TP_CACHE_MISSES += 1
            if _OPENEQ_TP_CACHE_MISSES <= 10 or _OPENEQ_TP_CACHE_MISSES % 10 == 0:
                print(
                    "[openeq tp] cache miss",
                    _OPENEQ_TP_CACHE_MISSES,
                    "weight_numel=",
                    self.weight_numel,
                    "instructions=",
                    len(self.instructions),
                    str(o3.Irreps(irreps_in1)),
                    "x",
                    str(o3.Irreps(irreps_in2)),
                    "->",
                    str(o3.Irreps(irreps_out)),
                )
            problem_start = time.perf_counter()
            problem = oeq.TPProblem(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions,
                **problem_kwargs,
            )
            problem_duration = time.perf_counter() - problem_start
            tp_start = time.perf_counter()
            cached_tp = oeq.TensorProduct(problem, torch_op=True)
            tp_duration = time.perf_counter() - tp_start
            _openeq_debug_print(
                "cache miss built",
                f"problem={problem_duration:.3f}s",
                f"tensor_product={tp_duration:.3f}s",
                f"weight_numel={self.weight_numel}",
                f"instructions={len(self.instructions)}",
            )
            _OPENEQ_TP_CACHE[cache_key] = cached_tp
        else:
            _OPENEQ_TP_CACHE_HITS += 1
        object.__setattr__(self, "_oeq_tp", cached_tp)
        self._openeq_target_forward_max_batch = _openeq_target_forward_max_batch_from_env()
        self._openeq_last_successful_chunk_size: Optional[int] = None

        # Precompute vectorized reorder permutation indices.
        # Replaces the per-spec loop in reorder_weights_from_e3nn() with a single
        # gather op: weight[perm] instead of ~298 small slice/permute/copy ops.
        # Use register_buffer with persistent=False so .to(device) moves them
        # automatically, avoiding forward-time state mutation under DDP.
        if self._needs_weight_reorder:
            perm = self._build_reorder_perm(cached_tp)
            inv_perm = torch.argsort(perm)
        else:
            perm = None
            inv_perm = None
        self.register_buffer('_reorder_perm', perm, persistent=False)
        self.register_buffer('_reorder_inv_perm', inv_perm, persistent=False)

        # Mirror top-level reference state so existing e3nn checkpoints remain loadable.
        self._mirror_reference_state(reference_tp)

        # S1: Store internal weights in OEQ layout to eliminate all runtime reorder.
        # Checkpoints are always saved/loaded in e3nn layout for compatibility.
        self._weights_in_oeq_layout = False
        if self._needs_weight_reorder and self.internal_weights and self._reorder_perm is not None:
            with torch.no_grad():
                self.weight.data = self.weight.data[self._reorder_perm]
            self._weights_in_oeq_layout = True

    @staticmethod
    def _legacy_compiled_state_keys(
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> list[str]:
        """Match legacy e3nn codegen buffers that should not participate in loading.

        Older e3nn checkpoints may contain deeply nested ``_compiled_*`` buffers such as
        ``_compiled_main_left_right._w3j_*``. The OpenEquivariance wrapper rebuilds its
        execution objects internally and does not expose those child modules, so these
        checkpoint entries must be ignored while keeping strict loading for real weights.
        """
        return [
            key
            for key in state_dict.keys()
            if key.startswith(prefix) and "._compiled_" in key
        ]

    def _mirror_reference_state(self, reference_tp: o3.TensorProduct) -> None:
        for name, param in reference_tp.named_parameters(recurse=False):
            if param is None:
                if name in self._parameters:
                    continue
                self.register_parameter(name, None)
                continue
            cloned = nn.Parameter(
                param.detach().clone(),
                requires_grad=param.requires_grad,
            )
            if name in self._buffers:
                del self._buffers[name]
                self._non_persistent_buffers_set.discard(name)
            self._parameters[name] = cloned

        for name, buffer in reference_tp.named_buffers(recurse=False):
            if buffer is None:
                continue
            cloned = buffer.detach().clone()
            if name in self._parameters:
                del self._parameters[name]
            self._buffers[name] = cloned
            self._non_persistent_buffers_set.discard(name)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Convert incoming e3nn-layout weights to OEQ layout before loading.
        if self._weights_in_oeq_layout and self._reorder_perm is not None:
            weight_key = prefix + "weight"
            if weight_key in state_dict:
                weight = state_dict[weight_key].clone()
                perm = self._reorder_perm
                if perm.device != weight.device:
                    perm = perm.to(weight.device)
                state_dict[weight_key] = weight.index_select(-1, perm)
        legacy_compiled_keys = self._legacy_compiled_state_keys(state_dict, prefix)
        for key in legacy_compiled_keys:
            state_dict.pop(key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if unexpected_keys:
            unexpected_keys[:] = [
                key
                for key in unexpected_keys
                if not (key.startswith(prefix) and "._compiled_" in key)
            ]

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Convert OEQ-layout weights back to e3nn layout for checkpoint compatibility.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._weights_in_oeq_layout and self._reorder_inv_perm is not None:
            weight_key = prefix + "weight"
            if weight_key in destination:
                w = destination[weight_key]
                inv_perm = self._reorder_inv_perm
                if inv_perm.device != w.device:
                    inv_perm = inv_perm.to(w.device)
                destination[weight_key] = w.index_select(-1, inv_perm)

    @staticmethod
    def _build_reorder_perm(oeq_tp) -> torch.Tensor:
        """Precompute a permutation index tensor for vectorized weight reorder.

        Passes ``torch.arange(N).float()`` through the OEQ reorder function so
        the resulting tensor encodes the e3nn→OEQ index mapping.  At runtime the
        reorder becomes a single gather: ``weight[perm]``.
        """
        weight_numel = oeq_tp.weight_numel
        probe = torch.arange(weight_numel, dtype=torch.float32)
        if not getattr(oeq_tp.config, "shared_weights", True):
            probe = probe.unsqueeze(0)
        reordered = oeq_tp.reorder_weights_from_e3nn(probe)
        if reordered.dim() == 2:
            reordered = reordered.squeeze(0)
        perm = reordered.long()
        # Sanity: every index in [0, weight_numel) must appear exactly once.
        assert perm.shape == (weight_numel,), f"perm shape {perm.shape} != ({weight_numel},)"
        assert perm.min() >= 0 and perm.max() < weight_numel, "perm out of range"
        assert perm.unique().numel() == weight_numel, "perm is not a valid permutation"
        return perm

    def _reorder_weights_if_needed(self, weights: torch.Tensor, already_oeq: bool = False) -> torch.Tensor:
        if already_oeq or not self._needs_weight_reorder:
            return weights
        if self._reorder_perm is not None:
            perm = self._reorder_perm
            if perm.device != weights.device:
                perm = perm.to(weights.device)
            return weights.index_select(-1, perm)
        return self._oeq_tp.reorder_weights_from_e3nn(
            weights,
            has_batch_dim=not self.shared_weights,
        )

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        """Return *True* if *exc* is a CUDA out-of-memory error."""
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _call_oeq_tp(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return self._oeq_tp(x, y, weight)

    def _forward_with_safe_chunking(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        tp_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with automatic batch chunking on CUDA OOM."""
        batch_size = x.shape[0]
        max_batch = self._openeq_target_forward_max_batch

        if max_batch is None or batch_size <= max_batch:
            try:
                return self._call_oeq_tp(x, y, tp_weight)
            except Exception as exc:
                if not self._is_cuda_oom(exc):
                    raise
                max_batch = max(batch_size // 2, 1)
                if max_batch >= batch_size:
                    raise
                self._openeq_target_forward_max_batch = max_batch
                _openeq_debug_print(
                    f"OOM at batch_size={batch_size}, "
                    f"retrying with chunk_size={max_batch}"
                )
                torch.cuda.empty_cache()

        chunks_x = x.split(max_batch, dim=0)
        chunks_y = y.split(max_batch, dim=0)
        if not self.shared_weights and tp_weight.dim() > 1:
            chunks_w = tp_weight.split(max_batch, dim=0)
        else:
            chunks_w = [tp_weight] * len(chunks_x)
        results = []
        for cx, cy, cw in zip(chunks_x, chunks_y, chunks_w):
            results.append(self._call_oeq_tp(cx, cy, cw))
        self._openeq_last_successful_chunk_size = max_batch
        return torch.cat(results, dim=0)

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.internal_weights:
            if weight is not None:
                raise TypeError("internal-weight tensor product does not accept external weight input")
            tp_weight = self.weight
            already_oeq = self._weights_in_oeq_layout
        else:
            if weight is None:
                raise TypeError("external-weight tensor product requires a weight input")
            tp_weight = weight
            already_oeq = False
        tp_weight = self._reorder_weights_if_needed(tp_weight, already_oeq=already_oeq)
        return self._forward_with_safe_chunking(x, y, tp_weight)


class SplitOpenEquivarianceTensorProduct(nn.Module):
    """Strict OpenEquivariance TP split into smaller instruction groups.

    Some large HamGNN irreps produce a single OEQ kernel that can spend many
    minutes in NVRTC. This wrapper preserves the e3nn TensorProduct contract by
    keeping one public weight tensor in the reference e3nn layout, then calling
    multiple smaller OEQ tensor products and summing their outputs.
    """

    def __init__(
        self,
        *args,
        _reference_tp: Optional[o3.TensorProduct] = None,
        split_max_weight_numel: int,
        **kwargs,
    ):
        super().__init__()
        if split_max_weight_numel <= 0:
            raise ValueError("split_max_weight_numel must be positive")
        if len(args) > 4:
            raise ValueError("OpenEquivariance split adapter only supports instructions as the 4th positional arg")

        reference_tp = _reference_tp or o3.TensorProduct(*args, **kwargs)
        irreps_in1, irreps_in2, irreps_out, _instructions = _normalize_tp_call(args, kwargs)
        if not reference_tp.internal_weights:
            raise ValueError("split OpenEquivariance adapter currently supports internal weights only")
        if not reference_tp.shared_weights:
            raise ValueError("split OpenEquivariance adapter currently supports shared weights only")

        self.irreps_in1 = reference_tp.irreps_in1
        self.irreps_in2 = reference_tp.irreps_in2
        self.irreps_out = reference_tp.irreps_out
        self.instructions = reference_tp.instructions
        self.shared_weights = reference_tp.shared_weights
        self.internal_weights = reference_tp.internal_weights
        self.weight_numel = reference_tp.weight_numel
        self.split_max_weight_numel = int(split_max_weight_numel)

        problem_kwargs = {}
        for key in (
            "in1_var",
            "in2_var",
            "out_var",
            "irrep_normalization",
            "path_normalization",
        ):
            if key in kwargs and kwargs[key] is not None:
                problem_kwargs[key] = kwargs[key]

        self._mirror_reference_state(reference_tp)
        groups = self._instruction_groups(
            reference_tp,
            self.split_max_weight_numel,
            problem_kwargs,
        )
        self._weight_slices: list[tuple[int, int]] = []
        output_masks = []
        self.sub_tps = nn.ModuleList()

        for start, stop, insts in groups:
            sub_tp = OpenEquivarianceTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                insts,
                shared_weights=True,
                internal_weights=False,
                **problem_kwargs,
            )
            self.sub_tps.append(sub_tp)
            self._weight_slices.append((start, stop))
            output_masks.append(self._group_output_mask(self.irreps_out, insts))
        self.register_buffer(
            "_split_output_masks",
            torch.stack(output_masks, dim=0),
            persistent=False,
        )
        self.register_state_dict_post_hook(_drop_split_sub_tp_state_dict_hook)
        self.register_load_state_dict_pre_hook(
            _drop_split_sub_tp_load_state_dict_pre_hook
        )
        self.register_load_state_dict_post_hook(
            _drop_split_sub_tp_load_state_dict_post_hook
        )

    @staticmethod
    def _instruction_weight_numel(instruction: Any) -> int:
        shape = getattr(instruction, "path_shape", instruction[6] if len(instruction) > 6 else None)
        if shape is None:
            raise ValueError("instruction is missing path_shape")
        return int(math.prod(shape))

    @staticmethod
    def _instruction_tuple(instruction: Any, path_weight: Optional[float] = None) -> tuple:
        return (
            getattr(instruction, "i_in1", instruction[0]),
            getattr(instruction, "i_in2", instruction[1]),
            getattr(instruction, "i_out", instruction[2]),
            _instruction_mode(instruction),
            _instruction_has_weight(instruction),
            float(path_weight)
            if path_weight is not None
            else getattr(instruction, "path_weight", instruction[5] if len(instruction) > 5 else None),
        )

    @classmethod
    def _group_output_mask(
        cls,
        irreps_out: o3.Irreps,
        instructions: Sequence[Any],
    ) -> torch.Tensor:
        mask = torch.zeros(irreps_out.dim, dtype=torch.float32)
        output_slices = irreps_out.slices()
        for instruction in instructions:
            i_out = getattr(instruction, "i_out", instruction[2])
            mask[output_slices[i_out]] = 1.0
        return mask

    @classmethod
    def _renormalized_instruction_tuples(
        cls,
        reference_tp: o3.TensorProduct,
        instructions: Sequence[Any],
        problem_kwargs: dict,
    ) -> list[tuple]:
        """Convert final e3nn path coefficients back to constructor weights."""
        unit_instructions = [
            cls._instruction_tuple(instruction, path_weight=1.0)
            for instruction in instructions
        ]
        unit_tp = o3.TensorProduct(
            reference_tp.irreps_in1,
            reference_tp.irreps_in2,
            reference_tp.irreps_out,
            unit_instructions,
            shared_weights=True,
            internal_weights=False,
            **problem_kwargs,
        )
        converted = []
        for reference_instruction, unit_instruction in zip(
            instructions,
            unit_tp.instructions,
        ):
            target = float(getattr(reference_instruction, "path_weight", reference_instruction[5]))
            base = float(unit_instruction.path_weight)
            if base == 0.0:
                raise ValueError("cannot split tensor product with zero path normalization coefficient")
            converted.append(
                cls._instruction_tuple(
                    reference_instruction,
                    path_weight=(target / base) ** 2,
                )
            )
        return converted

    @classmethod
    def _instruction_groups(
        cls,
        reference_tp: o3.TensorProduct,
        split_max_weight_numel: int,
        problem_kwargs: dict,
    ) -> list[tuple[int, int, list[tuple]]]:
        groups = []
        current = []
        group_start = 0
        cursor = 0
        group_weight = 0
        for instruction in reference_tp.instructions:
            inst_weight = cls._instruction_weight_numel(instruction)
            if current and group_weight + inst_weight > split_max_weight_numel:
                groups.append(
                    (
                        group_start,
                        cursor,
                        cls._renormalized_instruction_tuples(
                            reference_tp,
                            current,
                            problem_kwargs,
                        ),
                    )
                )
                group_start = cursor
                current = []
                group_weight = 0
            current.append(instruction)
            cursor += inst_weight
            group_weight += inst_weight
        if current:
            groups.append(
                (
                    group_start,
                    cursor,
                    cls._renormalized_instruction_tuples(
                        reference_tp,
                        current,
                        problem_kwargs,
                    ),
                )
            )
        if cursor != reference_tp.weight_numel:
            raise ValueError(
                f"split weight coverage {cursor} != tensor product weight_numel {reference_tp.weight_numel}"
            )
        return groups

    def _mirror_reference_state(self, reference_tp: o3.TensorProduct) -> None:
        for name, param in reference_tp.named_parameters(recurse=False):
            if param is None:
                self.register_parameter(name, None)
                continue
            self._parameters[name] = nn.Parameter(
                param.detach().clone(),
                requires_grad=param.requires_grad,
            )
        for name, buffer in reference_tp.named_buffers(recurse=False):
            if buffer is None:
                continue
            self._buffers[name] = buffer.detach().clone()

    @staticmethod
    def _drop_sub_tp_state_dict_keys(
        destination: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        sub_prefix = prefix + "sub_tps."
        for key in list(destination.keys()):
            if key.startswith(sub_prefix):
                del destination[key]

    @staticmethod
    def _is_sub_tp_state_key(key: str) -> bool:
        return key.startswith("sub_tps.") or ".sub_tps." in key

    @classmethod
    def _drop_sub_tp_load_state_keys(cls, incompatible_keys: Any) -> None:
        incompatible_keys.missing_keys[:] = [
            key
            for key in incompatible_keys.missing_keys
            if not cls._is_sub_tp_state_key(key)
        ]
        incompatible_keys.unexpected_keys[:] = [
            key
            for key in incompatible_keys.unexpected_keys
            if not cls._is_sub_tp_state_key(key)
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight is not None:
            raise TypeError("internal-weight split tensor product does not accept external weight input")
        result = None
        for group_idx, (sub_tp, (start, stop)) in enumerate(zip(self.sub_tps, self._weight_slices)):
            out = sub_tp(x, y, self.weight[start:stop])
            out = out * self._split_output_masks[group_idx].to(dtype=out.dtype)
            result = out if result is None else result + out
        if result is None:
            raise RuntimeError("split tensor product has no instruction groups")
        return result


def build_tensor_product(*args, backend: str = None, **kwargs):
    """Build a tensor product module from the selected backend."""
    selected = (backend or get_default_tp_backend()).lower()
    if selected == "e3nn":
        module = o3.TensorProduct(*args, **kwargs)
        _record_tp_backend_event(
            selected,
            "e3nn",
            weight_numel=module.weight_numel,
            instruction_count=len(module.instructions),
        )
        return module
    if selected == "cuequivariance":
        _warn_backend_fallback(selected, "optional backend adapter is not enabled in this environment")
        module = o3.TensorProduct(*args, **kwargs)
        _record_tp_backend_event(
            selected,
            "e3nn",
            reason="optional backend adapter is not enabled in this environment",
            weight_numel=module.weight_numel,
            instruction_count=len(module.instructions),
        )
        return module
    if selected == "openequivariance":
        try:
            reference_tp = o3.TensorProduct(*args, **kwargs)
            threshold_reason = _select_openeq_threshold_reason(
                reference_tp.weight_numel,
                len(reference_tp.instructions),
            )
            if threshold_reason is not None:
                _warn_backend_fallback(selected, threshold_reason)
                _record_tp_backend_event(
                    selected,
                    "e3nn",
                    reason=threshold_reason,
                    weight_numel=reference_tp.weight_numel,
                    instruction_count=len(reference_tp.instructions),
                )
                return reference_tp
            split_max_weight_numel = _openeq_split_max_weight_numel_from_env()
            if (
                split_max_weight_numel is not None
                and reference_tp.internal_weights
                and reference_tp.shared_weights
                and reference_tp.weight_numel > split_max_weight_numel
            ):
                module = SplitOpenEquivarianceTensorProduct(
                    *args,
                    _reference_tp=reference_tp,
                    split_max_weight_numel=split_max_weight_numel,
                    **kwargs,
                )
            else:
                module = OpenEquivarianceTensorProduct(
                    *args,
                    _reference_tp=reference_tp,
                    **kwargs,
                )
            _record_tp_backend_event(
                selected,
                "openequivariance",
                weight_numel=reference_tp.weight_numel,
                instruction_count=len(reference_tp.instructions),
            )
            return module
        except Exception as exc:
            if _openeq_traceback_enabled():
                print("[openeq debug] backend construction traceback follows")
                traceback.print_exc()
            _warn_backend_fallback(selected, str(exc))
            module = o3.TensorProduct(*args, **kwargs)
            _record_tp_backend_event(
                selected,
                "e3nn",
                reason=str(exc),
                weight_numel=module.weight_numel,
                instruction_count=len(module.instructions),
            )
            return module
    _warn_backend_fallback(selected, "unknown backend")
    module = o3.TensorProduct(*args, **kwargs)
    _record_tp_backend_event(
        selected,
        "e3nn",
        reason="unknown backend",
        weight_numel=module.weight_numel,
        instruction_count=len(module.instructions),
    )
    return module


def build_fully_connected_tensor_product(*args, backend: str = None, **kwargs):
    """Build a fully connected tensor product module from the selected backend."""
    selected = (backend or get_default_tp_backend()).lower()
    if selected == "e3nn":
        module = o3.FullyConnectedTensorProduct(*args, **kwargs)
        _record_tp_backend_event(
            selected,
            "e3nn",
            module_kind="fully_connected_tensor_product",
        )
        return module
    if selected in {"openequivariance", "cuequivariance"}:
        _warn_backend_fallback(selected, "FullyConnectedTensorProduct adapter is not implemented yet")
        module = o3.FullyConnectedTensorProduct(*args, **kwargs)
        _record_tp_backend_event(
            selected,
            "e3nn",
            reason="FullyConnectedTensorProduct adapter is not implemented yet",
            module_kind="fully_connected_tensor_product",
        )
        return module
    _warn_backend_fallback(selected, "unknown backend")
    module = o3.FullyConnectedTensorProduct(*args, **kwargs)
    _record_tp_backend_event(
        selected,
        "e3nn",
        reason="unknown backend",
        module_kind="fully_connected_tensor_product",
    )
    return module
