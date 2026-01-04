import torch
from e3nn import o3
from easydict import EasyDict

from .base_model import BaseModel

from ..nn.convolution import ConvBlockE3
from ..nn.embeddings import PairInteractionEmbeddingBlock, RadialBasisEdgeEncoding
from ..nn.interaction_blocks import CorrProductBlock, PairInteractionBlock
from ..toolbox.nequip.data import AtomicDataDict
from ..toolbox.nequip.nn import AtomwiseLinear
from ..toolbox.nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs
)
from ..utils.basis_functions import (
    BernsteinRadialBasisFunctions,
    BesselBasis,
    ExponentialBernsteinRadialBasisFunctions,
    ExponentialGaussianRadialBasisFunctions,
    GaussianRadialBasisFunctions,
    GaussianSmearing
)
from ..utils.cutoff_functions import CosineCutoff, cuttoff_envelope
from ..utils.math_utils import upgrade_tensor_precision


class HamGNNConvE3(BaseModel):
    def __init__(self, config):
        if 'radius_scale' not in config.HamGNN_pre:
            config.HamGNN_pre.radius_scale = 1.0
        else:
            assert config.HamGNN_pre.radius_scale > 1.0, "The radius scaling factor must be greater than 1.0."
        super().__init__(radius_type=config.HamGNN_pre.radius_type,
                         radius_scale=config.HamGNN_pre.radius_scale)

        # Configuration settings
        self.num_types = config.HamGNN_pre.num_types  # Number of atomic species
        self.set_features = True  # Whether to set one-hot encoding as node features
        # Irreps for edge spherical harmonics
        self.irreps_edge_sh = o3.Irreps(config.HamGNN_pre.irreps_edge_sh)
        self.edge_sh_normalization = config.HamGNN_pre.edge_sh_normalization
        self.edge_sh_normalize = config.HamGNN_pre.edge_sh_normalize
        self.build_internal_graph = config.HamGNN_pre.build_internal_graph
        if 'use_corr_prod' not in config.HamGNN_pre:
            self.use_corr_prod = False
        else:
            self.use_corr_prod = config.HamGNN_pre.use_corr_prod

        # Set product mode
        self.lite_mode = getattr(config.HamGNN_pre, 'lite_mode', False)

        # Radial basis function
        self.cutoff = config.HamGNN_pre.cutoff
        self.rbf_func = config.HamGNN_pre.rbf_func.lower()
        self.num_radial = config.HamGNN_pre.num_radial
        if self.rbf_func == 'gaussian':
            self.radial_basis_functions = GaussianSmearing(
                start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=None)
        elif self.rbf_func == 'bessel':
            self.radial_basis_functions = BesselBasis(
                cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=None)
        elif self.rbf_func == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(
                self.num_radial, self.cutoff)
        elif self.rbf_func == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(
                self.num_radial, self.cutoff)
        elif self.rbf_func == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(
                self.num_radial, self.cutoff)
        else:
            raise ValueError(
                f'Unsupported radial basis function: {self.rbf_func}')

        self.num_layers = config.HamGNN_pre.num_layers  # Number of transformer layers
        self.irreps_node_features = o3.Irreps(
            config.HamGNN_pre.irreps_node_features)  # Irreps for node features

        # Atomic embedding
        self.atomic_embedding = OneHotAtomEncoding(
            num_types=self.num_types, set_features=self.set_features)

        # Spherical harmonics for edges
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh,
                                                       edge_sh_normalization=self.edge_sh_normalization,
                                                       edge_sh_normalize=self.edge_sh_normalize)

        # Radial basis for edges
        self.cutoff_func = CosineCutoff(self.cutoff)
        self.radial_basis = RadialBasisEdgeEncoding(basis=self.radial_basis_functions,
                                                    cutoff=self.cutoff_func)

        # Edge features embedding
        use_kan = config.HamGNN_pre.use_kan
        self.radial_MLP = config.HamGNN_pre.radial_MLP
        self.pair_embedding = PairInteractionEmbeddingBlock(irreps_node_feats=self.atomic_embedding.irreps_out['node_attrs'],
                                                            irreps_edge_attrs=self.spharm_edges.irreps_out[
                                                                AtomicDataDict.EDGE_ATTRS_KEY],
                                                            irreps_edge_embed=self.radial_basis.irreps_out[
                                                                AtomicDataDict.EDGE_EMBEDDING_KEY],
                                                            irreps_edge_feats=self.irreps_node_features,
                                                            irreps_node_attrs=self.atomic_embedding.irreps_out[
                                                                'node_attrs'],
                                                            use_kan=use_kan,
                                                            radial_MLP=self.radial_MLP,
                                                            lite_mode=self.lite_mode)

        # Chemical embedding
        self.chemical_embedding = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.atomic_embedding.irreps_out['node_attrs']},
                                                 irreps_out=self.irreps_node_features)

        # Define the OrbTransformer layers
        correlation = config.HamGNN_pre.correlation
        num_hidden_features = config.HamGNN_pre.num_hidden_features

        self.convolutions = torch.nn.ModuleList()
        if self.use_corr_prod:
            self.corr_products = torch.nn.ModuleList()
        self.pair_interactions = torch.nn.ModuleList()

        for i in range(self.num_layers):
            conv = ConvBlockE3(irreps_in=self.irreps_node_features,
                               irreps_out=self.irreps_node_features,
                               irreps_node_attrs=self.atomic_embedding.irreps_out['node_attrs'],
                               irreps_edge_attrs=self.spharm_edges.irreps_out[AtomicDataDict.EDGE_ATTRS_KEY],
                               irreps_edge_embed=self.radial_basis.irreps_out[
                                   AtomicDataDict.EDGE_EMBEDDING_KEY],
                               radial_MLP=self.radial_MLP,
                               use_skip_connections=True,
                               use_kan=use_kan,
                               lite_mode=self.lite_mode)
            self.convolutions.append(conv)

            if self.use_corr_prod:
                corr_product = CorrProductBlock(
                    irreps_node_feats=self.irreps_node_features,
                    num_hidden_features=num_hidden_features,
                    correlation=correlation,
                    num_elements=self.num_types,
                    use_skip_connections=True
                )
                self.corr_products.append(corr_product)

            pair_interaction = PairInteractionBlock(irreps_node_feats=self.irreps_node_features,
                                                    irreps_node_attrs=self.atomic_embedding.irreps_out[
                                                        'node_attrs'],
                                                    irreps_edge_attrs=self.spharm_edges.irreps_out[
                                                        AtomicDataDict.EDGE_ATTRS_KEY],
                                                    irreps_edge_embed=self.radial_basis.irreps_out[
                                                        AtomicDataDict.EDGE_EMBEDDING_KEY],
                                                    irreps_edge_feats=self.irreps_node_features,
                                                    use_skip_connections=True if i > 0 else False,
                                                    use_kan=use_kan,
                                                    radial_MLP=self.radial_MLP,
                                                    lite_mode=self.lite_mode)
            self.pair_interactions.append(pair_interaction)

    def forward(self, data):
        if torch.get_default_dtype() == torch.float64:
            upgrade_tensor_precision(data)

        if self.build_internal_graph:
            graph = self.generate_graph(data)
        else:
            graph = data

        self.atomic_embedding(graph)
        self.spharm_edges(graph)
        self.radial_basis(graph)
        self.pair_embedding(graph)
        self.chemical_embedding(graph)
        # Orbital convolution
        for i in range(self.num_layers):
            self.convolutions[i](graph)
            if self.use_corr_prod:
                self.corr_products[i](graph)
            self.pair_interactions[i](graph)

        graph_representation = EasyDict()
        graph_representation['node_attr'] = graph[AtomicDataDict.NODE_FEATURES_KEY]
        if self.build_internal_graph:
            graph_representation['edge_attr'] = graph[AtomicDataDict.EDGE_FEATURES_KEY][graph.matching_edges]
        else:
            graph_representation['edge_attr'] = graph[AtomicDataDict.EDGE_FEATURES_KEY]
        return graph_representation
