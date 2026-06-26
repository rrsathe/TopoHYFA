"""
Unit tests for src/baselines.py

Coverage targets
----------------
- PCA_linear_regression_baseline : output shape, no-covariate path,
                                   covariate-augmented path
- impute_knn                     : output shape, no NaN in result
- impute_simple                  : output shape, no NaN in result
"""

import numpy as np
import pytest

from src.baselines import PCA_linear_regression_baseline, impute_knn, impute_simple

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def regression_data():
    """Synthetic source / target split for regression baseline tests."""
    rng = np.random.default_rng(0)
    n_train, n_test = 40, 10
    n_source_genes, n_target_genes = 50, 30

    x_source = rng.random((n_train, n_source_genes)).astype(np.float32)
    x_target = rng.random((n_train, n_target_genes)).astype(np.float32)
    x_source_test = rng.random((n_test, n_source_genes)).astype(np.float32)
    return x_source, x_target, x_source_test, n_test, n_target_genes


@pytest.fixture()
def covariate_data(regression_data):
    x_source, x_target, x_source_test, n_test, _ = regression_data
    rng = np.random.default_rng(1)
    cov_train = rng.random((len(x_source), 3)).astype(np.float32)
    cov_test = rng.random((n_test, 3)).astype(np.float32)
    return x_source, x_target, x_source_test, cov_train, cov_test


# ---------------------------------------------------------------------------
# PCA_linear_regression_baseline
# ---------------------------------------------------------------------------


class TestPCALinearRegressionBaseline:
    def test_output_shape(self, regression_data):
        x_source, x_target, x_source_test, n_test, n_target_genes = regression_data
        pred = PCA_linear_regression_baseline(x_source, x_target, x_source_test, n_components=10)
        assert pred.shape == (n_test, n_target_genes)

    def test_no_nan_in_output(self, regression_data):
        x_source, x_target, x_source_test, _, _ = regression_data
        pred = PCA_linear_regression_baseline(x_source, x_target, x_source_test, n_components=10)
        assert not np.isnan(pred).any()

    def test_with_covariates(self, covariate_data):
        x_source, x_target, x_source_test, cov_train, cov_test = covariate_data
        pred = PCA_linear_regression_baseline(
            x_source,
            x_target,
            x_source_test,
            x_source_covs=cov_train,
            x_source_test_covs=cov_test,
            n_components=10,
        )
        n_test = x_source_test.shape[0]
        n_target_genes = x_target.shape[1]
        assert pred.shape == (n_test, n_target_genes)
        assert not np.isnan(pred).any()

    def test_n_components_boundary(self, regression_data):
        """n_components=1 should still return a valid prediction."""
        x_source, x_target, x_source_test, n_test, n_target_genes = regression_data
        pred = PCA_linear_regression_baseline(x_source, x_target, x_source_test, n_components=1)
        assert pred.shape == (n_test, n_target_genes)


# ---------------------------------------------------------------------------
# impute_knn
# ---------------------------------------------------------------------------


class TestImputeKnn:
    @pytest.fixture()
    def knn_data(self):
        rng = np.random.default_rng(2)
        N, G = 12, 8
        n_cov = 2
        # Introduce some NaNs to simulate missing observations
        y = rng.random((N, G)).astype(np.float32)
        covariates = rng.random((N, n_cov)).astype(np.float32)
        return y, covariates

    def test_output_shape(self, knn_data):
        y, covariates = knn_data
        y_imp = impute_knn(y, covariates, k=3)
        assert y_imp.shape == y.shape

    def test_no_nan_after_imputation(self, knn_data):
        y, covariates = knn_data
        y_imp = impute_knn(y, covariates, k=3)
        assert not np.isnan(y_imp).any()


# ---------------------------------------------------------------------------
# impute_simple
# ---------------------------------------------------------------------------


class TestImputeSimple:
    @pytest.fixture()
    def simple_data(self):
        rng = np.random.default_rng(3)
        N, T, G = 10, 4, 6
        y = rng.random((N, T, G)).astype(np.float64)
        # Introduce NaNs to simulate missingness
        y[0, 1, :] = np.nan
        y[2, 3, :] = np.nan
        covariates = rng.random((N, 2)).astype(np.float64)
        return y, covariates

    def test_output_shape(self, simple_data):
        y, covariates = simple_data
        y_imp = impute_simple(y, covariates, strategy="mean")
        assert y_imp.shape == y.shape

    def test_no_nan_after_imputation(self, simple_data):
        y, covariates = simple_data
        y_imp = impute_simple(y, covariates, strategy="mean")
        assert not np.isnan(y_imp).any()

    def test_median_strategy(self, simple_data):
        y, covariates = simple_data
        y_imp = impute_simple(y, covariates, strategy="median")
        assert y_imp.shape == y.shape
        assert not np.isnan(y_imp).any()
