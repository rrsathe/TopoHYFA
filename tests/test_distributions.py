"""
Unit tests for src/distributions.py

Coverage targets
----------------
- log_nb_positive          : finite outputs, monotone in mu
- log_zinb_positive        : finite outputs, zero-inflation behaviour
- NegativeBinomial         : construction, mean/variance, log_prob shape, sampling
- ZeroInflatedNegativeBinomial : construction, mean ≤ NB mean, log_prob shape
- NegativeBinomialMixture  : construction, mean property, log_prob shape
"""

import pytest
import torch

from src.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial,
    log_nb_positive,
    log_zinb_positive,
)

BATCH, GENES = 8, 16


@pytest.fixture
def mu():
    return torch.rand(BATCH, GENES) + 0.5


@pytest.fixture
def theta():
    return torch.rand(BATCH, GENES) + 0.1


@pytest.fixture
def x():
    return torch.randint(0, 10, (BATCH, GENES)).float()


class TestLogNbPositive:
    def test_output_shape(self, x, mu, theta):
        out = log_nb_positive(x, mu, theta)
        assert out.shape == (BATCH, GENES)

    def test_all_finite(self, x, mu, theta):
        out = log_nb_positive(x, mu, theta)
        assert torch.isfinite(out).all()

    def test_log_prob_non_positive(self, x, mu, theta):
        """Log probability must be ≤ 0 (it is the log of a value in [0, 1])."""
        out = log_nb_positive(x, mu, theta)
        assert (out <= 0).all()

    def test_higher_mu_raises_log_prob_for_zero_obs(self):
        """P(X=0 | NB(mu, theta)) decreases as mu increases."""
        x = torch.zeros(1, 1)
        theta = torch.tensor([[2.0]])
        lp_low = log_nb_positive(x, torch.tensor([[0.1]]), theta)
        lp_high = log_nb_positive(x, torch.tensor([[5.0]]), theta)
        assert lp_low.item() > lp_high.item()


class TestLogZinbPositive:
    def test_output_shape(self, x, mu, theta):
        pi = torch.zeros_like(mu)
        out = log_zinb_positive(x, mu, theta, pi)
        assert out.shape == (BATCH, GENES)

    def test_all_finite(self, x, mu, theta):
        pi = torch.zeros_like(mu)
        out = log_zinb_positive(x, mu, theta, pi)
        assert torch.isfinite(out).all()

    def test_zero_inflation_increases_zero_prob(self, mu, theta):
        """High pi (strong zero inflation) should raise P(X=0) relative to low pi."""
        x_zero = torch.zeros(BATCH, GENES)
        lp_no_zi = log_zinb_positive(x_zero, mu, theta, pi=torch.full_like(mu, -10.0))
        lp_strong_zi = log_zinb_positive(x_zero, mu, theta, pi=torch.full_like(mu, 10.0))
        assert (lp_strong_zi > lp_no_zi).all()


class TestNegativeBinomial:
    def test_construction_mu_theta(self, mu, theta):
        dist = NegativeBinomial(mu=mu, theta=theta)
        assert dist.mu is not None
        assert dist.theta is not None

    def test_mean_equals_mu(self, mu, theta):
        dist = NegativeBinomial(mu=mu, theta=theta)
        assert torch.allclose(dist.mean, mu)

    def test_variance_geq_mean(self, mu, theta):
        """For NB, Var = mu + mu² / theta ≥ mu."""
        dist = NegativeBinomial(mu=mu, theta=theta)
        assert (dist.variance >= dist.mean).all()

    def test_log_prob_shape(self, x, mu, theta):
        dist = NegativeBinomial(mu=mu, theta=theta)
        lp = dist.log_prob(x)
        assert lp.shape == (BATCH, GENES)

    def test_log_prob_finite(self, x, mu, theta):
        dist = NegativeBinomial(mu=mu, theta=theta)
        assert torch.isfinite(dist.log_prob(x)).all()

    def test_sample_shape(self, mu, theta):
        dist = NegativeBinomial(mu=mu, theta=theta)
        samp = dist.sample()
        assert samp.shape == (BATCH, GENES)

    def test_invalid_construction_raises(self, mu, theta):
        with pytest.raises(ValueError, match="one of the two possible parameterizations"):
            NegativeBinomial(mu=mu, theta=theta, total_count=theta)

    def test_counts_logits_parameterization(self, mu, theta):
        """Alternative (total_count, logits) parameterization should construct."""

        total_count = theta
        logits = (mu + 1e-8).log() - (theta + 1e-8).log()
        dist = NegativeBinomial(total_count=total_count, logits=logits)
        assert dist.mu is not None


class TestZeroInflatedNegativeBinomial:
    def test_construction(self, mu, theta):
        zi_logits = torch.zeros_like(mu)
        dist = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
        assert dist.zi_logits is not None

    def test_missing_zi_logits_raises(self, mu, theta):
        with pytest.raises(ValueError, match="zi_logits"):
            ZeroInflatedNegativeBinomial(mu=mu, theta=theta)

    def test_mean_leq_nb_mean(self, mu, theta):
        """ZINB mean = (1 - pi) * mu ≤ mu."""
        zi_logits = torch.zeros_like(mu)
        dist = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
        nb_dist = NegativeBinomial(mu=mu, theta=theta)
        assert (dist.mean <= nb_dist.mean + 1e-6).all()

    def test_log_prob_shape(self, x, mu, theta):
        zi_logits = torch.zeros_like(mu)
        dist = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
        lp = dist.log_prob(x)
        assert lp.shape == (BATCH, GENES)

    def test_log_prob_finite(self, x, mu, theta):
        zi_logits = torch.zeros_like(mu)
        dist = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
        assert torch.isfinite(dist.log_prob(x)).all()

    def test_sample_shape(self, mu, theta):
        zi_logits = torch.zeros_like(mu)
        dist = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
        samp = dist.sample()
        assert samp.shape == (BATCH, GENES)


class TestNegativeBinomialMixture:
    def test_construction(self, mu, theta):
        mixture_logits = torch.zeros_like(mu)
        dist = NegativeBinomialMixture(
            mu1=mu, mu2=mu * 2, theta1=theta, mixture_logits=mixture_logits
        )
        assert dist.mu1 is not None

    def test_mean_between_components(self, mu, theta):
        """Mixture mean must lie between the two component means."""
        mu1 = mu * 0.5
        mu2 = mu * 2.0
        mixture_logits = torch.zeros_like(mu)
        dist = NegativeBinomialMixture(
            mu1=mu1, mu2=mu2, theta1=theta, mixture_logits=mixture_logits
        )
        mean = dist.mean
        assert (mean >= mu1 - 1e-5).all()
        assert (mean <= mu2 + 1e-5).all()

    def test_log_prob_shape(self, x, mu, theta):
        mixture_logits = torch.zeros_like(mu)
        dist = NegativeBinomialMixture(
            mu1=mu, mu2=mu * 2, theta1=theta, mixture_logits=mixture_logits
        )
        lp = dist.log_prob(x)
        assert lp.shape == (BATCH, GENES)

    def test_log_prob_finite(self, x, mu, theta):
        mixture_logits = torch.zeros_like(mu)
        dist = NegativeBinomialMixture(
            mu1=mu, mu2=mu * 2, theta1=theta, mixture_logits=mixture_logits
        )
        assert torch.isfinite(dist.log_prob(x)).all()

    def test_sample_shape(self, mu, theta):
        mixture_logits = torch.zeros_like(mu)
        dist = NegativeBinomialMixture(
            mu1=mu, mu2=mu * 2, theta1=theta, mixture_logits=mixture_logits
        )
        samp = dist.sample()
        assert samp.shape == (BATCH, GENES)
