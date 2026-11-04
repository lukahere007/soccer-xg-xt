"""How much of "finishing skill" is noise? Empirical Bayes with a Gamma-Poisson model.

For player ``i`` with ``G_i`` goals from shots totalling ``X_i`` expected goals, let

    G_i | theta_i ~ Poisson(theta_i * X_i),    theta_i ~ Gamma(shape=a, rate=a).

``theta_i`` is a finishing multiplier (1 = average finisher); the prior has mean 1
and variance ``1 / a``. Integrating ``theta`` out gives a negative-binomial
marginal likelihood, so ``a`` is estimated from all players at once (empirical
Bayes). The posterior is conjugate:

    theta_i | G_i ~ Gamma(a + G_i, a + X_i),  E[theta_i] = (a + G_i) / (a + X_i).

A player with little xG is pulled almost all the way back to 1; only a large,
sustained over-performance survives. (This is the count-data analogue of the
beta-binomial shrinkage used for batting averages; Poisson-on-xG respects that
every shot has a different scoring chance.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


def negbin_loglik(a: float, goals, xg) -> float:
    """Marginal log-likelihood of goals given xG under the Gamma(a, a) prior."""
    g = np.asarray(goals, float)
    x = np.asarray(xg, float)
    return float(np.sum(
        special.gammaln(a + g) - special.gammaln(a) - special.gammaln(g + 1)
        + a * np.log(a / (a + x)) + g * np.log(x / (a + x))
    ))


def fit_prior(goals, xg, bounds=(1e-2, 1e4)) -> float:
    """Maximum-marginal-likelihood prior concentration ``a`` (searched on log scale).

    A value at the upper bound means the data show no detectable spread in
    finishing skill beyond Poisson noise.
    """
    res = optimize.minimize_scalar(lambda la: -negbin_loglik(np.exp(la), goals, xg),
                                   bounds=np.log(bounds), method="bounded")
    return float(np.exp(res.x))


def posterior(goals, xg, a: float) -> pd.DataFrame:
    """Posterior mean finishing multiplier, 90% interval and shrunk goals above xG."""
    g = np.asarray(goals, float)
    x = np.asarray(xg, float)
    shape, rate = a + g, a + x
    lo = stats.gamma.ppf(0.05, shape, scale=1 / rate)
    hi = stats.gamma.ppf(0.95, shape, scale=1 / rate)
    mean = shape / rate
    return pd.DataFrame({
        "theta_mean": mean, "theta_lo": lo, "theta_hi": hi,
        "raw_gax": g - x, "shrunk_gax": (mean - 1) * x,
    })


def signal_share(goals, shot_probs_by_player) -> dict:
    """Method-of-moments check: how much of the spread in goals - xG is beyond noise?

    ``shot_probs_by_player`` is a list of arrays (each player's per-shot xG). If
    finishing were pure luck, Var(G - xG) summed over players would equal the
    sum of Bernoulli variances ``sum p (1 - p)``.
    """
    g = np.asarray(goals, float)
    x = np.array([np.sum(p) for p in shot_probs_by_player])
    noise = np.array([np.sum(np.asarray(p) * (1 - np.asarray(p))) for p in shot_probs_by_player])
    observed = float(np.sum((g - x) ** 2))
    expected = float(noise.sum())
    return {"observed_ss": observed, "expected_noise_ss": expected,
            "excess_share": max(0.0, 1 - expected / observed) if observed > 0 else 0.0}


def loglik_sd(sd: float, goals, xg) -> float:
    """Marginal log-likelihood as a function of the prior sd of theta (sd = 1/sqrt(a)).

    ``sd = 0`` is the no-skill limit, where the marginal is exactly Poisson(xG).
    """
    if sd <= 0:
        return float(np.sum(stats.poisson.logpmf(np.asarray(goals), np.asarray(xg, float))))
    return negbin_loglik(1.0 / sd**2, goals, xg)


def sd_upper_bound(goals, xg, level: float = 0.95, sd_max: float = 2.0) -> tuple[float, float]:
    """Profile-likelihood interval for the prior sd: returns (MLE sd, upper bound).

    The upper bound is the largest sd whose log-likelihood is within
    ``chi2_1(level) / 2`` of the maximum, i.e. the most generous spread in
    finishing skill that the data cannot reject.
    """
    grid = np.linspace(0.0, sd_max, 801)
    ll = np.array([loglik_sd(s, goals, xg) for s in grid])
    best = int(np.argmax(ll))
    cut = ll[best] - stats.chi2.ppf(level, 1) / 2
    above = np.nonzero(ll >= cut)[0]
    return float(grid[best]), float(grid[above.max()])
