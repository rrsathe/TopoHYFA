"""Small deterministic helpers shared by the HD case-study workflow."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

CASE_STUDY_SEED = 0


def inverse_normal(matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """Apply a Blom inverse-normal rank transform along columns or rows.

    Parameters
    ----------
    matrix:
        Numeric matrix to transform.
    axis:
        ``0`` ranks each column across samples. ``1`` ranks each row across genes.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        msg = "inverse_normal expects a 2D matrix"
        raise ValueError(msg)
    if axis not in {0, 1}:
        msg = "axis must be 0 or 1"
        raise ValueError(msg)

    work = values if axis == 0 else values.T
    out = np.zeros_like(work, dtype=float)
    n = work.shape[0]
    for j in range(work.shape[1]):
        ranks = rankdata(work[:, j], method="average")
        out[:, j] = norm.ppf((ranks - 0.375) / (n + 0.25))
    return out if axis == 0 else out.T


def col_pearson(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute column-wise Pearson correlation for equally shaped matrices."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        msg = f"shape mismatch: {a.shape} != {b.shape}"
        raise ValueError(msg)

    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a**2).sum(axis=0)) * np.sqrt((b**2).sum(axis=0))
    result = np.full(a.shape[1], np.nan, dtype=float)
    ok = den > eps
    result[ok] = num[ok] / den[ok]
    return result


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary ROC-AUC using ranks, with deterministic tie handling."""
    y_score = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    if y_score.shape[0] != y_true.shape[0]:
        msg = "scores and labels must have the same length"
        raise ValueError(msg)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    numerator = ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(numerator / (len(pos) * len(neg)))
