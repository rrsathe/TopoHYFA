"""Tests for HD case-study reproducibility utilities."""

import numpy as np

from src.topohyfa_hd import auc_rank, col_pearson, inverse_normal


def test_inverse_normal_preserves_shape_and_center():
    matrix = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    transformed = inverse_normal(matrix)
    assert transformed.shape == matrix.shape
    assert np.allclose(transformed.mean(axis=0), 0.0)


def test_col_pearson_matches_perfect_correlation():
    matrix = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    assert np.allclose(col_pearson(matrix, matrix), 1.0)


def test_auc_rank_handles_ties_deterministically():
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auc_rank(scores, labels) == 0.875
