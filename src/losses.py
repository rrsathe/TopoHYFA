"""
Adapted from SCVI-tools
Copyright (c) 2020 Romain Lopez, Adam Gayoso, Galen Xing, Yosef Lab
Copyright (c) 2022 Ramon Vinas
All rights reserved.
"""

import torch

from src.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial


def compute_laplacian(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute graph Laplacian L = D - A from adjacency matrix A.
    """
    deg = adjacency_matrix.sum(dim=1)
    return torch.diag(deg) - adjacency_matrix


def graph_laplacian_regularization(
    gene_weights: torch.Tensor, laplacian_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Compute normalized trace penalty Tr(W^T L W) / |W|.
    """
    laplacian_matrix = laplacian_matrix.to(gene_weights.device)

    # Ensure W has shape (n_genes, latent_dim) for Tr(W^T L W)
    if gene_weights.shape[0] != laplacian_matrix.shape[0]:
        if gene_weights.shape[1] == laplacian_matrix.shape[0]:
            gene_weights = gene_weights.t()
        else:
            raise ValueError(
                "Incompatible shapes for Laplacian regularization: "
                f"W={tuple(gene_weights.shape)}, L={tuple(laplacian_matrix.shape)}"
            )

    reg_loss = torch.einsum("ik,ij,jk->", gene_weights, laplacian_matrix, gene_weights)
    reg_loss = reg_loss / max(gene_weights.numel(), 1)
    return reg_loss


def get_reconstruction_loss(
    x, px_rate, px_r=None, px_dropout=None, gene_likelihood="nb", aggr="mean", **kwargs
) -> torch.Tensor:
    if gene_likelihood == "zinb":
        reconst_loss = -ZeroInflatedNegativeBinomial(
            mu=px_rate, theta=px_r, zi_logits=px_dropout
        ).log_prob(x)
    elif gene_likelihood == "nb":
        reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x)
    elif gene_likelihood == "poisson":
        reconst_loss = -Poisson(px_rate).log_prob(x)
    elif gene_likelihood == "normal":  # For normalised gene expression
        reconst_loss = -torch.distributions.normal.Normal(loc=px_rate, scale=px_r).log_prob(x)
        # reconst_loss = F.mse_loss(px_rate, x)  # Note: Ignoring sd
    else:
        raise ValueError(f"Unknown gene_likelihood: {gene_likelihood}")

    if aggr == "mean":
        reconst_loss = reconst_loss.mean(dim=-1)
    else:
        reconst_loss = reconst_loss.sum(dim=-1)

    return reconst_loss
