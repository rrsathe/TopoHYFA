"""
Unit tests for src/losses.py

Coverage targets
----------------
- compute_laplacian : output shape, symmetry, row-sum property, identity matrix
- graph_laplacian_regularization : non-negativity, normalisation, transpose input
- get_reconstruction_loss : all four likelihoods, mean vs sum aggregation, bad key
"""

import pytest
import torch

from src.losses import compute_laplacian, get_reconstruction_loss, graph_laplacian_regularization

# ---------------------------------------------------------------------------
# compute_laplacian
# ---------------------------------------------------------------------------


class TestComputeLaplacian:
    def test_output_shape(self, n_genes):
        A = torch.rand(n_genes, n_genes)
        L = compute_laplacian(A)
        assert L.shape == (n_genes, n_genes)

    def test_diagonal_equals_row_degree(self, n_genes):
        """Diagonal of L must equal the row-sum of A when A has no self-loops."""
        A = torch.rand(n_genes, n_genes).abs()
        A.fill_diagonal_(0.0)  # remove self-loops so L_ii = deg(i) - 0 = deg(i)
        L = compute_laplacian(A)
        expected_diag = A.sum(dim=1)
        assert torch.allclose(L.diagonal(), expected_diag)

    def test_off_diagonal_negates_A(self, n_genes):
        """Off-diagonal of L must equal -A."""
        A = torch.rand(n_genes, n_genes).abs()
        A.fill_diagonal_(0.0)  # use hollow A for a clean Laplacian
        L = compute_laplacian(A)
        mask = ~torch.eye(n_genes, dtype=torch.bool)
        assert torch.allclose(L[mask], -A[mask])

    def test_symmetry_preserved_for_symmetric_input(self, n_genes):
        A = torch.rand(n_genes, n_genes)
        A = (A + A.t()) / 2  # symmetrise
        L = compute_laplacian(A)
        assert torch.allclose(L, L.t())

    def test_identity_matrix(self):
        """Laplacian of the identity = 0 because deg(i) = 1 and A_ii = 1."""
        A = torch.eye(4)
        L = compute_laplacian(A)
        assert torch.allclose(L, torch.zeros(4, 4))

    def test_zero_matrix(self):
        A = torch.zeros(5, 5)
        L = compute_laplacian(A)
        assert torch.allclose(L, torch.zeros(5, 5))


# ---------------------------------------------------------------------------
# graph_laplacian_regularization
# ---------------------------------------------------------------------------


class TestGraphLaplacianRegularization:
    def _make_laplacian(self, n: int) -> torch.Tensor:
        A = torch.rand(n, n).abs()
        return compute_laplacian(A)

    def test_nonnegative(self, n_genes, n_latent):
        """Tr(W^T L W) is always >= 0 for a valid (PSD) Laplacian with non-negative A."""
        A = torch.rand(n_genes, n_genes).abs()
        L = compute_laplacian(A)
        W = torch.randn(n_genes, n_latent)
        reg = graph_laplacian_regularization(W, L)
        assert reg.item() >= 0.0

    def test_normalisation(self, n_genes, n_latent):
        """Regularisation is divided by |W|; scaling W by k multiplies result by k^2."""
        A = torch.rand(n_genes, n_genes).abs()
        L = compute_laplacian(A)
        W = torch.randn(n_genes, n_latent)
        reg1 = graph_laplacian_regularization(W, L)
        reg2 = graph_laplacian_regularization(2 * W, L)
        assert torch.isclose(reg2, 4 * reg1)

    def test_transposed_W_gives_same_result(self, n_genes, n_latent):
        """Function should handle W with swapped dimensions transparently."""
        A = torch.rand(n_genes, n_genes).abs()
        L = compute_laplacian(A)
        W = torch.randn(n_genes, n_latent)
        reg_normal = graph_laplacian_regularization(W, L)
        reg_transposed = graph_laplacian_regularization(W.t(), L)
        assert torch.isclose(reg_normal, reg_transposed)

    def test_incompatible_shapes_raise(self):
        L = torch.eye(4)
        W = torch.randn(7, 3)  # neither dim matches L.shape[0] == 4
        with pytest.raises(ValueError, match="Incompatible shapes"):
            graph_laplacian_regularization(W, L)


# ---------------------------------------------------------------------------
# get_reconstruction_loss
# ---------------------------------------------------------------------------


class TestGetReconstructionLoss:
    def test_nb_output_shape_mean(self, count_obs, positive_rate, dispersion):
        loss = get_reconstruction_loss(
            count_obs, positive_rate, px_r=dispersion, gene_likelihood="nb", aggr="mean"
        )
        assert loss.shape == (count_obs.shape[0],)

    def test_nb_output_shape_sum(self, count_obs, positive_rate, dispersion):
        loss = get_reconstruction_loss(
            count_obs, positive_rate, px_r=dispersion, gene_likelihood="nb", aggr="sum"
        )
        assert loss.shape == (count_obs.shape[0],)

    def test_zinb(self, count_obs, positive_rate, dispersion):
        dropout_logits = torch.zeros_like(count_obs)
        loss = get_reconstruction_loss(
            count_obs,
            positive_rate,
            px_r=dispersion,
            px_dropout=dropout_logits,
            gene_likelihood="zinb",
        )
        assert loss.shape == (count_obs.shape[0],)
        assert torch.isfinite(loss).all()

    def test_poisson(self, count_obs, positive_rate):
        loss = get_reconstruction_loss(count_obs, positive_rate, gene_likelihood="poisson")
        assert loss.shape == (count_obs.shape[0],)
        assert torch.isfinite(loss).all()

    def test_normal(self, count_obs, positive_rate, dispersion):
        loss = get_reconstruction_loss(
            count_obs, positive_rate, px_r=dispersion, gene_likelihood="normal"
        )
        assert loss.shape == (count_obs.shape[0],)
        assert torch.isfinite(loss).all()

    def test_unknown_likelihood_raises(self, count_obs, positive_rate):
        with pytest.raises(ValueError, match="Unknown gene_likelihood"):
            get_reconstruction_loss(count_obs, positive_rate, gene_likelihood="bad_key")

    def test_nb_loss_is_nonnegative(self, count_obs, positive_rate, dispersion):
        """Negative log-likelihood is non-negative by construction."""
        loss = get_reconstruction_loss(
            count_obs, positive_rate, px_r=dispersion, gene_likelihood="nb"
        )
        assert (loss >= 0).all()

    def test_mean_vs_sum_aggregation(self, count_obs, positive_rate, dispersion):
        """sum-aggregated loss should equal mean * n_genes for matching shape."""
        n_genes = count_obs.shape[1]
        loss_mean = get_reconstruction_loss(
            count_obs, positive_rate, px_r=dispersion, gene_likelihood="nb", aggr="mean"
        )
        loss_sum = get_reconstruction_loss(
            count_obs, positive_rate, px_r=dispersion, gene_likelihood="nb", aggr="sum"
        )
        assert torch.allclose(loss_sum, loss_mean * n_genes)
