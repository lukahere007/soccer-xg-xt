import numpy as np
import pytest
from scipy import stats

from goalvalue import shrinkage as sh


def test_posterior_mean_closed_form():
    # a = 2, G = 3, X = 1  ->  E[theta] = (2 + 3) / (2 + 1) = 5/3
    post = sh.posterior([3], [1.0], a=2.0)
    assert post["theta_mean"].iloc[0] == pytest.approx(5 / 3)
    assert post["shrunk_gax"].iloc[0] == pytest.approx(2 / 3)  # vs raw +2
    assert post["raw_gax"].iloc[0] == pytest.approx(2.0)


def test_shrinkage_stronger_for_small_samples():
    # same 50% over-performance, 10x the sample -> less shrinkage
    post = sh.posterior([3, 30], [2.0, 20.0], a=10.0)
    assert post["theta_mean"].iloc[1] > post["theta_mean"].iloc[0] > 1.0


def test_loglik_matches_scipy_negative_binomial():
    a, g, x = 3.0, np.array([0, 2, 5]), np.array([0.5, 1.7, 3.2])
    ref = stats.nbinom.logpmf(g, a, a / (a + x)).sum()
    assert sh.negbin_loglik(a, g, x) == pytest.approx(ref)


def test_fit_prior_recovers_concentration():
    rng = np.random.default_rng(1)
    x = rng.uniform(2, 12, 4000)
    theta = rng.gamma(5.0, 1 / 5.0, size=x.size)
    g = rng.poisson(theta * x)
    assert sh.fit_prior(g, x) == pytest.approx(5.0, rel=0.25)


def test_fit_prior_pure_luck_goes_to_bound():
    rng = np.random.default_rng(2)
    x = rng.uniform(1, 8, 3000)
    g = rng.poisson(x)  # every player is an exactly average finisher
    assert sh.fit_prior(g, x) > 100


def test_signal_share_zero_when_no_variance_beyond_noise():
    out = sh.signal_share([1, 0], [np.array([0.5, 0.5]), np.array([0.5])])
    # observed SS = (1-1)^2 + (0-0.5)^2 = 0.25; noise = 0.5 + 0.25 = 0.75
    assert out["excess_share"] == 0.0


def test_loglik_sd_continuous_at_zero():
    g, x = np.array([0, 1, 4]), np.array([0.4, 1.1, 2.5])
    assert sh.loglik_sd(1e-4, g, x) == pytest.approx(sh.loglik_sd(0.0, g, x), abs=1e-4)


def test_sd_upper_bound_brackets_truth():
    rng = np.random.default_rng(3)
    x = rng.uniform(2, 12, 3000)
    g = rng.poisson(rng.gamma(4.0, 1 / 4.0, size=x.size) * x)  # true sd = 0.5
    mle, upper = sh.sd_upper_bound(g, x)
    assert mle < 0.5 < upper or abs(mle - 0.5) < 0.05
    assert upper > mle
