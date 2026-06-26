"""
Unit tests for src/eval_utils.py

Coverage targets
----------------
- pearson_correlation : shape, known values, symmetry (transpose)
- r2 : perfect prediction, zero prediction, shape
- pearson_correlation_score : diagonal values equal per-gene correlation
- r2_score : consistent with r2
"""

import numpy as np
import pytest

from src.eval_utils import pearson_correlation, pearson_correlation_score, r2, r2_score

# ---------------------------------------------------------------------------
# pearson_correlation
# ---------------------------------------------------------------------------


class TestPearsonCorrelation:
    def test_output_shape(self, numpy_matrix):
        n, g = numpy_matrix.shape
        corr = pearson_correlation(numpy_matrix, numpy_matrix)
        assert corr.shape == (g, g)

    def test_self_correlation_is_identity(self, numpy_matrix):
        """corr(X, X) should be the identity matrix (each gene is perfectly correlated
        with itself and, in general, not perfectly with others)."""
        corr = pearson_correlation(numpy_matrix, numpy_matrix)
        diag = np.diagonal(corr)
        assert np.allclose(diag, 1.0, atol=1e-5)

    def test_perfectly_correlated_columns(self):
        """Two identical columns must yield corr = 1."""
        rng = np.random.default_rng(0)
        x = rng.random((20, 1)).astype(np.float32)
        y = x.copy()  # identical
        corr = pearson_correlation(x, y)
        assert np.allclose(corr[0, 0], 1.0, atol=1e-5)

    def test_anticorrelated_columns(self):
        """A column and its negation must yield corr = -1."""
        rng = np.random.default_rng(1)
        x = rng.random((20, 1)).astype(np.float32)
        y = -x
        corr = pearson_correlation(x, y)
        assert np.allclose(corr[0, 0], -1.0, atol=1e-5)

    def test_sample_count_mismatch_raises(self):
        x = np.ones((5, 3))
        y = np.ones((6, 3))
        with pytest.raises(AssertionError):
            pearson_correlation(x, y)

    def test_different_gene_dims(self):
        rng = np.random.default_rng(2)
        x = rng.random((10, 4))
        y = rng.random((10, 6))
        corr = pearson_correlation(x, y)
        assert corr.shape == (4, 6)


# ---------------------------------------------------------------------------
# r2
# ---------------------------------------------------------------------------


class TestR2:
    def test_perfect_prediction_is_one(self, numpy_matrix):
        """When pred == gt, R² must be exactly 1."""
        r_sq = r2(numpy_matrix, numpy_matrix)
        assert np.allclose(r_sq, 1.0, atol=1e-5)

    def test_constant_prediction_is_zero_or_less(self, numpy_matrix):
        """Predicting the mean for all samples gives R² = 0."""
        pred = np.mean(numpy_matrix, axis=0, keepdims=True) * np.ones_like(numpy_matrix)
        r_sq = r2(numpy_matrix, pred)
        assert np.allclose(r_sq, 0.0, atol=1e-5)

    def test_output_shape(self, numpy_matrix):
        n, g = numpy_matrix.shape
        r_sq = r2(numpy_matrix, numpy_matrix)
        assert r_sq.shape == (g,)

    def test_worse_than_mean_is_negative(self, numpy_matrix):
        """Predictions far from ground truth should produce negative R²."""
        bad_pred = numpy_matrix + 100.0
        r_sq = r2(numpy_matrix, bad_pred)
        assert (r_sq < 0).all()


# ---------------------------------------------------------------------------
# Score wrappers
# ---------------------------------------------------------------------------


class TestPearsonCorrelationScore:
    def test_gene_wise_diagonal(self, numpy_matrix):
        """Gene-wise score == diagonal of the full correlation matrix."""
        scores = pearson_correlation_score(numpy_matrix, numpy_matrix, sample_corr=False)
        expected = np.diagonal(pearson_correlation(numpy_matrix, numpy_matrix))
        assert np.allclose(scores, expected, atol=1e-5)

    def test_sample_wise_mode(self, numpy_matrix):
        """sample_corr=True transposes x & y before computing; shape must match n_samples."""
        scores = pearson_correlation_score(numpy_matrix, numpy_matrix, sample_corr=True)
        n_samples = numpy_matrix.shape[0]
        assert scores.shape == (n_samples,)

    def test_self_score_is_one(self, numpy_matrix):
        scores = pearson_correlation_score(numpy_matrix, numpy_matrix)
        assert np.allclose(scores, 1.0, atol=1e-5)


class TestR2Score:
    def test_perfect_score_is_one(self, numpy_matrix):
        scores = r2_score(numpy_matrix, numpy_matrix, sample_corr=False)
        assert np.allclose(scores, 1.0, atol=1e-5)

    def test_sample_wise_mode(self, numpy_matrix):
        scores = r2_score(numpy_matrix, numpy_matrix, sample_corr=True)
        n_samples = numpy_matrix.shape[0]
        assert scores.shape == (n_samples,)
