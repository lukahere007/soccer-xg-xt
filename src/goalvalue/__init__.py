"""goalvalue: expected goals (xG) and expected threat (xT) from StatsBomb open data.

Modules
-------
geometry   shot geometry: visible goal angle, distances, point-in-triangle
data       parse raw StatsBomb events/lineups into tidy shot, action and minutes tables
xg         feature engineering, models and match-grouped cross-validation for xG
xt         expected-threat Markov model (iterative and linear-system solvers)
shrinkage  empirical-Bayes (Gamma-Poisson) finishing-skill estimates
plots      pitch drawing and the project figures
"""

__version__ = "0.1.0"
