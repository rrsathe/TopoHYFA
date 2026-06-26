"""
Defines hypergraph neural network
"""

from typing import Any

import torch
import torch.nn as nn

from src.hnn_utils import MLP
from src.hypergraph_layer import GATHypergraphLayer, MPNNHypergraphLayer
from src.metagene_decoders import get_decoder
from src.metagene_encoders import PlainEncoder


class HypergraphNeuralNet(torch.nn.Module):
    def __init__(self, config):
        """
        :param config: configuration object (e.g. wandb config) with hyperparameters
        :param patient_features: Initial node features of each individual. Shape=(nb_individuals, feature_dim)
        """
        super().__init__()
        self.config = config
        self.var_eps = 1e-4

        params_dict: dict[str, nn.Parameter] = {}

        total_dim = 0
        for (
            k,
            v,
        ) in config.static_node_types.items():
            n, dim = v
            params_dict[k] = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((n, dim))))
            total_dim += dim
        self.params = nn.ParameterDict(params_dict)

        for (
            _k,
            v,
        ) in config.dynamic_node_types.items():
            n, dim = v
            total_dim += dim

        meta_G = config.meta_G
        self.meta_G = meta_G
        self.metagenes = nn.Parameter(torch.arange(meta_G), requires_grad=False)

        self.metagenes_encoder = PlainEncoder(in_dim=config.G, out_dim=meta_G * config.d_edge_attr)

        layer = {"gat": GATHypergraphLayer, "mpnn": MPNNHypergraphLayer}[config.layer]

        self.hypergraph_layers = nn.ModuleList(
            [
                layer(
                    config.dynamic_node_types,
                    config.static_node_types,
                    d_edge_attr=config.d_edge_attr,
                    d_edge=config.d_edge,
                    dropout=config.dropout,
                    n_hidden_layers=config.n_hidden_layers,
                    n_heads=config.n_heads,
                    norm=config.norm,
                    activation=config.activation,
                    update_edge_attr=config.update_edge_attr,
                    attention_strategy=config.attention_strategy,
                )
            ]
            * config.n_graph_layers
        )

        self.metagenes_decoder = get_decoder(config.loss_type)(
            in_dim=meta_G * config.d_edge_attr, out_dim=config.G
        )

        self.prediction_mlp = MLP(
            in_dim=total_dim,
            h_dim=total_dim,
            out_dim=config.d_edge_attr,
            n_layers=config.n_hidden_layers_pred,
            dropout=config.dropout,
            norm=config.norm,
            activation=config.activation,
        )

    def encode_metagenes(self, x, **kwargs):
        """
        Compute metagene values (i.e. this reduces the dimensionality of high-dimensional expression data)
        :param x: torch tensor with gene expression values. Shape=(nb_samples, nb_genes)
        :param kwargs: keyword arguments for the encoder
        :return: torch tensor with shape (nb_samples, nb_metagenes, metagene_dim)
        """
        out = self.metagenes_encoder(x, **kwargs)
        return torch.reshape(out, (-1, self.meta_G, self.config.d_edge_attr))

    def decode_metagenes(self, x, **kwargs):
        """
        Map metagene values back to high-dimensional space of gene expression
        :param x: torch tensor with metagene values. Shape=(nb_samples, nb_metagenes * metagene_dim)
        :param kwargs: keyword arguments for the decoder
        :return: torch tensor with shape (nb_samples, nb_genes)
        """
        return self.metagenes_decoder(x, **kwargs)

    def forward(self, hyperedge_index, hyperedge_attr, dynamic_node_features):
        """
        Applies hypergaph neural layers
        :param hyperedge_index: indices of hyperedges (similar to edge_index of PyTorch Geometric). Shape=(3, nb_hyperedges)
        :param hyperedge_attr: hyperedge features (similar to edge_attr of PyTorch Geometric). Shape=(nb_hyperedges, d_edge_attr)
        :return: tuple of node features (individual features, tissue features, metagene features), where:
                - patient_features: features of individual nodes. Shape=(nb_patients, d_patient)
                - tissue_features: features of tissue nodes. Shape=(nb_tissues, d_tissue)
                - metagene_features: features of tissue nodes. Shape=(nb_metagenes, d_metagene)
        """
        static_node_features = self.params

        dynamic_node_features_: dict[str, Any] = {}
        for k, v in self.config.dynamic_node_types.items():
            _, spec_dim = v
            features = dynamic_node_features[k]
            n, actual_dim = features.shape
            assert spec_dim >= actual_dim
            dynamic_node_features_[k] = torch.ones((n, spec_dim))

            if self.config.use_demographic_information:
                dynamic_node_features_[k][:, :actual_dim] = features
            dynamic_node_features_[k] = dynamic_node_features_[k].to(features.device)

        node_features = (
            dynamic_node_features_,
            static_node_features,
        )

        for layer in self.hypergraph_layers:
            dynamic_updates, hyperedge_attr = layer(hyperedge_index, hyperedge_attr, node_features)

            for k, v in dynamic_node_features_.items():
                dynamic_node_features_[k] = v + dynamic_updates[k]

            node_features = (dynamic_node_features_, static_node_features)

        for k in dynamic_node_features_:
            q = dynamic_node_features_[k]

            dynamic_node_features_[k] = {"latent": q, "mu": q}

        node_features = (dynamic_node_features_, static_node_features)

        return node_features

    def predict(self, target_hyperedge_index, node_features, use_latent_mean=False, **kwargs):
        """
        Given the latent features of all nodes in the hypergraph, predicts the metagene values (i.e. hyperedge attributes)
        of all hyperedges in target_hyperedge_index
        :param target_hyperedge_index: indices of the indices of the hyperedges (similar to edge_index of
               PyTorch Geometric) whose hyperedge values need to be predicted. Shape=(3, nb_hyperedges)
        :param node_features: tuple of node features (individual features, tissue features, metagene features), where:
                - patient_features: features of individual nodes. Shape=(nb_patients, d_patient)
                - tissue_features: features of tissue nodes. Shape=(nb_tissues, d_tissue)
                - metagene_features: features of tissue nodes. Shape=(nb_metagenes, d_metagene)
        :return: torch tensor with predicted hyperedge features for each hyperedge in target_hyperedge_index.
                Shape=(nb_hyperedges, d_edge_attr).
        """
        dynamic_node_features, static_node_features = node_features

        if use_latent_mean:
            dynamic_node_features_ = {k: v["mu"] for k, v in dynamic_node_features.items()}
        else:
            dynamic_node_features_ = {k: v["latent"] for k, v in dynamic_node_features.items()}

        node_features_ = {**dynamic_node_features_, **static_node_features}
        catted_features = torch.cat(
            [node_features_[k][target_hyperedge_index[k]] for k in sorted(node_features_.keys())],
            dim=-1,
        )

        return self.prediction_mlp(catted_features)
