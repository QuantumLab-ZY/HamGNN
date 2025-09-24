import torch
from e3nn import o3
from typing import Callable, Dict, List, Optional, Tuple
from ..toolbox.nequip.nn.nonlinearities import ShiftedSoftPlus


def extract_scalar_irreps(irreps: o3.Irreps) -> o3.Irreps:
    """
    Extracts and returns the scalar irreducible representations (irreps) from the given irreps.

    A scalar irrep is defined as one with l=0 and p=1. This function calculates the total 
    multiplicity of such scalar irreps and constructs a new Irreps object containing only these.

    Parameters:
    - irreps (o3.Irreps): The input irreps from which to extract scalar components.

    Returns:
    - o3.Irreps: An Irreps object containing only the scalar components.
    """
    scalar_multiplicity = sum(
        multiplicity for multiplicity, irrep in irreps if irrep.l == 0 and irrep.p == 1
    )
    return o3.Irreps(f"{scalar_multiplicity}x0e")


acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}

def irreps2gate(
    irreps: o3.Irreps,
    nonlinearity_scalars: Dict[int, str] = {1: "ssp", -1: "tanh"},
    nonlinearity_gates: Dict[int, str] = {1: "ssp", -1: "abs"},
) -> Tuple[o3.Irreps, o3.Irreps, o3.Irreps, List[Callable], List[Callable]]:
    """
    Splits irreducible representations into scalar and gated components and associates activation functions.

    Parameters:
    - irreps (o3.Irreps): The input irreducible representations.
    - nonlinearity_scalars (Dict[int, str]): Activation functions for scalar components.
    - nonlinearity_gates (Dict[int, str]): Activation functions for gate components.

    Returns:
    - Tuple containing:
        - irreps_scalars (o3.Irreps): Scalar irreps.
        - irreps_gates (o3.Irreps): Gate irreps.
        - irreps_gated (o3.Irreps): Gated irreps.
        - act_scalars (List[Callable]): Activation functions for scalars.
        - act_gates (List[Callable]): Activation functions for gates.
    """
    # Split the irreps into scalar and gated components
    irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0]).simplify()
    irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l != 0]).simplify()

    # Determine the gate irreps based on the presence of gated components
    irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in irreps_gated]).simplify() if irreps_gated.dim > 0 else o3.Irreps([])

    # Retrieve the activation functions for scalars and gates
    act_scalars = [acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars]
    act_gates = [acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates]

    return irreps_scalars, irreps_gates, irreps_gated, act_scalars, act_gates

def scale_irreps(irreps: o3.Irreps, factor: float) -> o3.Irreps:
    """
    Scales the multiplicities of the irreducible representations (irreps) by a given factor,
    ensuring they remain at least 1.

    Parameters:
    - irreps (o3.Irreps): The input irreps.
    - factor (float): The scaling factor.

    Returns:
    - o3.Irreps: The scaled irreps.
    """
    return o3.Irreps([(max(1, int(mul * factor)), ir) for mul, ir in irreps])

def filter_and_split_irreps(irreps: o3.Irreps, num_channels: int, min_l: int, max_l: int) -> o3.Irreps:
    """
    Filters and splits irreducible representations (irreps) based on specified angular momentum range.

    Parameters:
    - irreps (o3.Irreps): The input irreducible representations.
    - num_channels (int): The number of channels to split the multiplicity by.
    - min_l (int): The minimum angular momentum (inclusive).
    - max_l (int): The maximum angular momentum (inclusive).

    Returns:
    - o3.Irreps: The resulting irreducible representations after filtering and splitting.
    """
    result_irreps = o3.Irreps()
    for multiplicity, irrep in irreps:
        if irrep.l < min_l or irrep.l > max_l:
            # Retain irreps outside the specified l range
            result_irreps += o3.Irreps([(multiplicity, irrep)])
        else:
            # Split multiplicity by num_channels for irreps within the range
            split_multiplicity = multiplicity // num_channels
            if split_multiplicity > 0:
                result_irreps += split_multiplicity * o3.Irreps([(num_channels, irrep)])
    
    return result_irreps
