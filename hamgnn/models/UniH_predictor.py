"""
Delta learning predictor for HamGNN.

Loads a pre-trained HamGNN model from a pickle file to provide H0 baseline
predictions for delta learning training.

This module re-exports the essential functions from the Uni-HamGNN predictor
utilities, adapting imports for use within the hamgnn package.
"""
import os
import sys
import types
import pickle
from typing import Optional

import torch
import torch.nn as nn


class HamiltonianPredictor:
    """Stub class for unpickling old predictor.pkl files.

    The predictor was originally saved from HamGNN_v_2_1.models.UniH_predictor.
    This class provides the minimum interface needed for delta learning.
    """
    def __init__(self):
        self.non_soc_model = None
        self.soc_model = None
        self.soc_enabled = False
        self.device = 'cpu'


def _register_legacy_package_aliases() -> None:
    """Register legacy package aliases so that pickled models saved under
    the old package name (HamGNN_v_2_1) can be unpickled with the current
    hamgnn package."""
    legacy_prefix = 'HamGNN_v_2_1'
    current_prefix = 'hamgnn'

    for module_name, module in list(sys.modules.items()):
        if module_name == current_prefix or module_name.startswith(f'{current_prefix}.'):
            legacy_name = module_name.replace(current_prefix, legacy_prefix, 1)
            sys.modules.setdefault(legacy_name, module)


def _patch_legacy_attributes(predictor) -> None:
    """Patch modules that lack attributes added in newer HamGNN versions."""
    models = [predictor.non_soc_model] + \
        ([predictor.soc_model] if getattr(predictor, 'soc_model', None) else [])
    for model in models:
        for _, module in model.named_modules():
            cls_name = type(module).__name__
            if cls_name == 'MessagePackBlock' and not hasattr(module, 'lite_mode'):
                module.lite_mode = False
            if cls_name == 'PairInteractionBlock' and not hasattr(module, 'legacy_edge_update'):
                module.legacy_edge_update = True


def _fix_predictor_contractions(model):
    """Patch MACE SymmetricContraction modules for PyTorch 2.2 compatibility.

    Replaces TorchScript-compiled forward with pure torch.einsum implementation.
    """
    _ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]
    patched = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ == "SymmetricContraction":
            for i, contraction in enumerate(mod.contractions):
                irrep_out = mod.irreps_out[i]
                lmax = irrep_out.ir.l
                correlation = contraction.correlation

                # Extract weights as plain list
                fixed_weights = []
                for wi in range(correlation - 1):
                    key = str(wi)
                    param = None
                    if key in contraction.weights._modules:
                        param = contraction.weights._modules[key]
                    elif hasattr(contraction.weights, '_parameters') and key in contraction.weights._parameters:
                        param = contraction.weights._parameters[key]
                    else:
                        for pname, p in contraction.named_parameters():
                            if pname == f'weights.{key}':
                                param = p
                                break
                    if param is not None:
                        fixed_weights.append(param)
                contraction._fixed_weights = fixed_weights

                # Build einsum equations
                eqs = {}
                for nu in range(correlation, 0, -1):
                    if nu == correlation:
                        parse = ([_ALPHABET[j] for j in range(nu + min(lmax, 1) - 1)]
                                 + ["ik,ekc,bci,be -> bc"]
                                 + [_ALPHABET[j] for j in range(nu + min(lmax, 1) - 1)])
                        eqs[nu] = ("".join(parse), None, None)
                    else:
                        parse_w = ([_ALPHABET[j] for j in range(nu + min(lmax, 1))]
                                   + ["k,ekc,be->bc"]
                                   + [_ALPHABET[j] for j in range(nu + min(lmax, 1))])
                        parse_f = (["bc"]
                                   + [_ALPHABET[j] for j in range(nu - 1 + min(lmax, 1))]
                                   + ["i,bci->bc"]
                                   + [_ALPHABET[j] for j in range(nu - 1 + min(lmax, 1))])
                        eqs[nu] = (None, "".join(parse_w), "".join(parse_f))
                contraction._einsum_eqs = eqs

                def _einsum_forward(self, x, y):
                    eqs = self._einsum_eqs
                    main_eq = eqs[self.correlation][0]
                    out = torch.einsum(main_eq, self.U_tensors(self.correlation),
                                       self.weights_max, x, y)
                    for i, weight in enumerate(self._fixed_weights):
                        nu = self.correlation - i - 1
                        _, weighting_eq, features_eq = eqs[nu]
                        c_tensor = torch.einsum(weighting_eq, self.U_tensors(nu), weight, y)
                        c_tensor = c_tensor + out
                        out = torch.einsum(features_eq, c_tensor, x)
                    return out.reshape(out.shape[0], -1)

                contraction.forward = types.MethodType(_einsum_forward, contraction)
                patched += 1
    if patched > 0:
        print(f"[predictor fix] Patched {patched} MACE Contraction modules for PyTorch 2.2 compat")
    return patched


def load_model_predictor(model_filepath: str, device: Optional[str] = None):
    """
    Loads a model predictor from a pickle file and assigns it to the specified device.

    Args:
        model_filepath (str): Path to the pickle file containing the predictor model.
        device (Optional[str]): The computation device ('cpu' or 'cuda') to assign the model to.
                                 If None, the device assignment is skipped.

    Returns:
        Predictor: The loaded predictor model with device assigned if specified.
    """
    _register_legacy_package_aliases()

    # Ensure hamgnn package is importable (for unpickling)
    _hamgnn_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _hamgnn_root not in sys.path:
        sys.path.insert(0, _hamgnn_root)

    # The predictor.pkl was pickled with V2.0 classes from Universal_magham.
    # Add the Universal_magham path so pickle can find HamGNN_v_2_0.models.HamGNN.net
    _v20_search_paths = [
        os.path.join(_hamgnn_root, '..'),  # parent of repo root
        '/public/home/yuhongyu/Universal_magham/HamGNN_v2.1',  # jinsi cluster
    ]
    for _p in _v20_search_paths:
        _p = os.path.abspath(_p)
        if os.path.isdir(os.path.join(_p, 'HamGNN_v_2_0')) and _p not in sys.path:
            sys.path.insert(0, _p)
            break

    # NOTE: Do NOT alias hamgnn.* -> HamGNN_v_2_0.* because V2.0 has a completely
    # different module structure (HamGNN_v_2_0.models.HamGNN.net etc.).
    # The real V2.0 package is importable via sys.path above.

    # Patch ParameterList for PyTorch 2.2 compatibility (old pickle missing _size)
    _orig_pl_len = nn.ParameterList.__len__
    def _safe_pl_len(self):
        if not hasattr(self, '_size'):
            self._size = len(self._modules) if hasattr(self, '_modules') else 0
        return self._size
    nn.ParameterList.__len__ = _safe_pl_len

    with open(model_filepath, 'rb') as file_handle:
        model_predictor = pickle.load(file_handle)

    # Fix MACE SymmetricContraction for PyTorch 2.2
    _fix_predictor_contractions(model_predictor.non_soc_model)
    if hasattr(model_predictor, 'soc_model') and model_predictor.soc_model:
        _fix_predictor_contractions(model_predictor.soc_model)

    _patch_legacy_attributes(model_predictor)

    if device:
        model_predictor.device = device
        model_predictor.non_soc_model = model_predictor.non_soc_model.to(device)
        if hasattr(model_predictor, 'soc_model') and model_predictor.soc_model:
            model_predictor.soc_model = model_predictor.soc_model.to(device)

    return model_predictor
