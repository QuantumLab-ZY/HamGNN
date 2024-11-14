from typing import Union

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from ....nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin
from ..radial_basis import BesselBasis
from ..cutoffs import PolynomialCutoff


class SphericalHarmonicEdgeAttrs(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field
        self.coord_change = torch.LongTensor([1,2,0])

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        j, i = data.edge_index
        nbr_shift = data.nbr_shift
        pos = data.pos
        edge_vec = (pos[i]+nbr_shift) - pos[j]  # j->i: ri-rj = rji
        edge_vec = torch.nn.functional.normalize(edge_vec, dim=-1) # eji Shape(Nedges, 3)
        edge_sh = self.sh(edge_vec[:, self.coord_change])
        data[self.out_field] = edge_sh
        return data


class RadialBasisEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.basis = basis
        self.cutoff = cutoff
        self.out_field = out_field
        if type(basis).__name__.split(".")[-1] == 'BesselBasis':
            num_basis = basis.freqs.size()[0]
        elif type(basis).__name__.split(".")[-1] == 'GaussianSmearing':
            num_basis = basis.offset.size()[0]
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(num_basis, (0, 1))])},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:        
        j, i = data.edge_index
        nbr_shift = data.nbr_shift
        pos = data.pos
        edge_dir = (pos[i]+nbr_shift) - pos[j]  # j->i: ri-rj = rji
        edge_length = edge_dir.pow(2).sum(dim=-1).sqrt()
        
        edge_length_embedded = (
            self.basis(edge_length) * self.cutoff(edge_length)[:, None]
        )
        data[self.out_field] = edge_length_embedded
        return data
