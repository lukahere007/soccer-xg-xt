"""Run the full analysis on the derived tables and write results/ and figures/.

Usage::

    python scripts/run_analysis.py

Reads only data/derived/*.parquet (committed), so it runs without downloading
anything. Takes about a minute on a laptop CPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Small models: one OpenMP thread is fastest and avoids oversubscription on shared CPUs.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from goalvalue import plots, shrinkage, xg, xt
from goalvalue.plots import INK, INK_2, MUTED, SERIES, plt

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
SEED = 7
MIN_MINUTES = 270  # three full matches
CAL_EDGES = [0, 0.02, 0.04, 0.06, 0.08, 0.11, 0.15, 0.2, 0.27, 0.35, 0.45, 0.6, 1.0]


def r(v, d=3):
    return float(np.round(v, d))


# --------------------------------------------------------------------------- xG
def run_xg(shots: pd.DataFrame, summary: dict) -> pd.DataFrame:
    np_shots = shots[shots["shot_type"] != "Penalty"].reset_index(drop=True)
    X = xg.prepare_features(np_shots)
    y = np_shots["goal"].to_numpy()
    groups = np_shots["match_id"].to_numpy()

    oof, chosen_C = xg.cross_validated_predictions(X, y, groups, n_splits=5, seed=SEED)
    base = np.full(len(y), np.nan)
    for k in range(5):  # base-rate baseline, also out-of-fold
        te = oof["fold"].to_numpy() == k
        base[te] = y[~te].mean()

    preds = {
        "Base rate (constant)": base,
        "Geometry logistic (distance, angle, body part)": oof["geometry"].to_numpy(),
        "Freeze-frame logistic GAM (ours)": oof["freeze"].to_numpy(),
        "Gradient boosting (ours)": oof["gbm"].to_numpy(),
        "StatsBomb xG (reference)": np_shots["statsbomb_xg"].to_numpy(),
    }
    table = xg.score_table(y, preds)
    table.loc["Base rate (constant)", ["roc_auc", "calib_slope"]] = np.nan  # undefined for a constant
    table.to_csv(RESULTS / "xg_scores.csv", float_format="%.4f")
    print(table.round(4).to_string())

    comparisons = {}
    for a, b, label in [
        ("freeze", "geometry", "freeze_minus_geometry"),
        ("gbm", "freeze", "gbm_minus_freeze"),
        ("freeze", "sb", "freeze_minus_statsbomb"),
        ("gbm", "sb", "gbm_minus_statsbomb"),
    ]:
        pa = oof[a].to_numpy()
        pb = np_shots["statsbomb_xg"].to_numpy() if b == "sb" else oof[b].to_numpy()
        d, ci = xg.paired_bootstrap_logloss(y, pa, pb, groups, seed=SEED)
        comparisons[label] = {"mean_diff": r(d, 4), "ci95": [r(ci[0], 4), r(ci[1], 4)]}

    # Final interpretable model fit on all non-penalty shots, for effect sizes.
    C_final = float(pd.Series(chosen_C["freeze"]).mode().iloc[0])
    final = xg.make_logistic("freeze", C_final).fit(X, y)

    def scenario(**over):
        row = X.median(numeric_only=True).to_frame().T
        row = row.assign(body="foot", set_piece_shot="open", technique_grp="normal",
                         pattern_grp="other", under_pressure=0.0, first_time=0.0,
                         gk_in_triangle=1.0, n_att_triangle=0.0, n_def_within_3=0.0)
        for k, v in over.items():
            row[k] = v
        return float(final.predict_proba(row[X.columns])[:, 1][0])

    from goalvalue.geometry import goal_angle
    spot = dict(distance=12.0, angle=float(goal_angle(108, 40)))
    effects = {
        "penalty_spot_open_goal_keeper_on_line": r(scenario(**spot, n_def_triangle=0, gk_dist_line=1.0,
                                                            gk_dist_shotline=0.3, nearest_def_dist=4.0)),
        "penalty_spot_two_defenders_in_triangle": r(scenario(**spot, n_def_triangle=2, gk_dist_line=1.0,
                                                             gk_dist_shotline=0.3, nearest_def_dist=4.0)),
        "penalty_spot_keeper_off_shot_line_3yd": r(scenario(**spot, n_def_triangle=0, gk_dist_line=1.0,
                                                            gk_dist_shotline=3.0, nearest_def_dist=4.0)),
        "penalty_spot_defender_1yd_away": r(scenario(**spot, n_def_triangle=0, gk_dist_line=1.0,
                                                     gk_dist_shotline=0.3, nearest_def_dist=1.0)),
    }

    # Which shots create goals? Concentration of goals in high-xG shots.
    p = oof["freeze"].to_numpy()
    bands = pd.cut(p, [0, 0.05, 0.15, 0.3, 1.0], labels=["<0.05", "0.05-0.15", "0.15-0.30", ">0.30"])
    conc = pd.DataFrame({"band": bands, "goal": y}).groupby("band", observed=False).agg(
        shots=("goal", "size"), goals=("goal", "sum"))
    conc["share_of_shots"] = conc["shots"] / conc["shots"].sum()
    conc["share_of_goals"] = conc["goals"] / conc["goals"].sum()
    conc["conversion"] = conc["goals"] / conc["shots"]
    conc.to_csv(RESULTS / "goal_concentration.csv", float_format="%.4f")
    by_triangle = pd.DataFrame({"n": X["n_def_triangle"], "goal": y, "dist": X["distance"]})
    inside_box = by_triangle["dist"] <= 18
    tri = by_triangle[inside_box].groupby("n").agg(shots=("goal", "size"), conversion=("goal", "mean"))

    pens = shots[shots["shot_type"] == "Penalty"]
    summary["xg"] = {
        "n_shots_total": int(len(shots)),
        "n_non_penalty_shots": int(len(np_shots)),
        "n_non_penalty_goals": int(y.sum()),
        "n_matches": int(shots["match_id"].nunique()),
        "penalties": {"n": int(len(pens)), "goals": int(pens["goal"].sum()),
                      "conversion": r(pens["goal"].mean())},
        "scores": {k: {m: r(v, 4) for m, v in row.items()} for k, row in table.iterrows()},
        "bootstrap_logloss_diff": comparisons,
        "chosen_C": chosen_C,
        "final_C": C_final,
        "scenario_xg": effects,
        "goal_concentration": {str(k): {c: r(v) for c, v in row.items()} for k, row in conc.iterrows()},
        "box_conversion_by_defenders_in_triangle": {
            str(int(k)): {"shots": int(v["shots"]), "conversion": r(v["conversion"])}
            for k, v in tri.iterrows()},
    }

    np_shots = np_shots.assign(xg_freeze=oof["freeze"], xg_gbm=oof["gbm"], xg_geometry=oof["geometry"])
    fig_xg(np_shots, final, X, scenario)
    fig_calibration(y, preds)
    return np_shots


def fig_xg(np_shots, final, X, scenario):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.12})
    plots.draw_pitch(a1, half=True)
    s = np_shots.sort_values("xg_freeze")
    sc = a1.scatter(s["x"], s["y"], c=s["xg_freeze"], cmap=plots.SEQ, vmin=0, vmax=0.5,
                    s=3 + 40 * s["xg_freeze"], lw=0, alpha=0.9)
    g = s[s["goal"] == 1]
    a1.scatter(g["x"], g["y"], s=5, color=SERIES[1], lw=0, alpha=0.8, label=f"goal ({len(g)})")
    a1.set_title(f"{len(s):,} non-penalty shots, coloured by out-of-fold xG")
    a1.legend(loc="lower left", fontsize=9, markerscale=2)
    cb = fig.colorbar(sc, ax=a1, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40)
    cb.set_label("xG (freeze-frame model)")
    cb.outline.set_visible(False)

    from goalvalue.geometry import goal_angle
    dists = np.linspace(6, 30, 60)
    for k, n_def in enumerate([0, 1, 2, 3]):
        vals = [scenario(distance=d, angle=float(goal_angle(120 - d, 40)), n_def_triangle=n_def,
                         gk_dist_line=1.0, gk_dist_shotline=0.3, nearest_def_dist=3.0) for d in dists]
        a2.plot(dists, vals, color=SERIES[k], lw=2, label=f"{n_def} outfield defender{'s' if n_def != 1 else ''} in the shot triangle")
    a2.set_xlabel("distance from goal centre, central shot (yards)")
    a2.set_ylabel("predicted xG")
    a2.set_title("Same spot, different picture: bodies between ball and goal")
    a2.legend(loc="upper right", fontsize=9)
    a2.set_ylim(0, None)
    plots.save(fig, FIGS / "xg_shot_map.png")


def fig_calibration(y, preds):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 0.7], [0, 0.7], color=MUTED, lw=1, ls="--")
    show = {"Freeze-frame logistic GAM (ours)": SERIES[0], "Gradient boosting (ours)": SERIES[2],
            "StatsBomb xG (reference)": SERIES[1]}
    for name, col in show.items():
        cal = xg.binned_calibration(y, preds[name], edges=CAL_EDGES)
        cal = cal[cal["n"] >= 20]
        ax.errorbar(cal["mean_pred"], cal["obs"], yerr=1.64 * cal["se"], color=col, lw=2,
                    marker="o", ms=5, capsize=0, elinewidth=1, label=name)
    ax.set_xlabel("mean predicted xG (bins with ≥ 20 shots)")
    ax.set_ylabel("observed goal rate (90% CI)")
    ax.set_title("Calibration on out-of-fold, match-grouped predictions")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.8)
    plots.save(fig, FIGS / "xg_calibration.png")


# --------------------------------------------------------------------------- xT
def run_xt(actions, np_shots, minutes, summary):
    moves = actions[~actions["set_piece"]]
    model = xt.fit_xt(moves, np_shots, nx=16, ny=12, shot_value="xg_freeze")
    raw_goal_model = xt.fit_xt(moves, np_shots, nx=16, ny=12, shot_value=None)
    xt_iter, _ = xt.solve_iterative(model.shoot, model.move, model.goal, model.transition, tol=1e-12)

    valued = moves.assign(xt_added=model.value(moves["x"], moves["y"], moves["end_x"], moves["end_y"]))
    valued.loc[~valued["success"], "xt_added"] = 0.0
    ranking = xt.player_xt_per90(valued, minutes, MIN_MINUTES)

    # Match-bootstrap 90% intervals for the per-90 rate of the top players.
    per_match = (valued[valued["success"]].groupby(["player_id", "match_id"])["xt_added"].sum()
                 .rename("xt").reset_index()
                 .merge(minutes[["player_id", "match_id", "minutes"]], on=["player_id", "match_id"], how="right")
                 .fillna({"xt": 0.0}))
    rng = np.random.default_rng(SEED)
    lo, hi = [], []
    for pid in ranking.index:
        pm = per_match[per_match["player_id"] == pid]
        xs, ms = pm["xt"].to_numpy(), pm["minutes"].to_numpy()
        idx = rng.integers(0, len(xs), size=(2000, len(xs)))
        b = xs[idx].sum(1) / ms[idx].sum(1) * 90
        lo.append(np.percentile(b, 5))
        hi.append(np.percentile(b, 95))
    ranking["p90_lo"], ranking["p90_hi"] = lo, hi
    teams = minutes.groupby("player_id")["team_id"].agg(lambda s: s.mode().iloc[0])
    ranking = ranking.join(teams)
    ranking.to_csv(RESULTS / "player_xt_per90.csv", float_format="%.4f")

    # Which passes create the most threat? Top start->end zone pairs (by total xT added).
    ok = valued[valued["success"] & (valued["action"] == "Pass")]
    pass_share_pos = ok.loc[ok["xt_added"] > 0, "xt_added"].sum()
    into_box = ok[(ok["end_x"] >= 102) & (ok["end_y"].between(18, 62)) & ~((ok["x"] >= 102) & ok["y"].between(18, 62))]

    surf = model.surface()
    np.savetxt(RESULTS / "xt_surface_16x12.csv", surf, delimiter=",", fmt="%.5f")
    top = ranking.head(15)
    summary["xt"] = {
        "grid": "16x12",
        "n_moves": int(len(moves)),
        "n_successful_moves": int(moves["success"].sum()),
        "iterations_to_converge": int(model.n_iter),
        "max_abs_diff_iterative_vs_linear": float(np.max(np.abs(model.xt - xt_iter))),
        "max_zone_xt": r(model.xt.max()),
        "own_box_zone_xt": r(surf[6, 0], 4),
        "centre_circle_zone_xt": r(surf[6, 8], 4),
        "corr_surface_xg_vs_goal_counts": r(np.corrcoef(model.xt, raw_goal_model.xt)[0, 1]),
        "share_of_positive_pass_xt_from_passes_into_box": r(into_box["xt_added"].clip(lower=0).sum() / pass_share_pos),
        "share_of_successful_passes_into_box": r(len(into_box) / len(ok)),
        "players_ranked": int(len(ranking)),
        "min_minutes": MIN_MINUTES,
        "top10": [{"player": p.player, "minutes": int(p.minutes), "xt_p90": r(p.xt_total_p90),
                   "pass_p90": r(p.xt_pass_p90), "carry_p90": r(p.xt_carry_p90),
                   "ci90": [r(p.p90_lo), r(p.p90_hi)]} for p in ranking.head(10).itertuples()],
    }
    fig_xt_surface(surf)
    fig_ranking(top)
    return model


def fig_xt_surface(surf):
    fig, ax = plt.subplots(figsize=(9, 6))
    from matplotlib.colors import PowerNorm
    im = ax.imshow(surf, extent=(0, 120, 80, 0), cmap=plots.SEQ, aspect="equal",
                   norm=PowerNorm(0.5, vmin=0, vmax=surf.max()), interpolation="nearest", zorder=0)
    plots.draw_pitch(ax, color=MUTED, lw=1.0)
    ax.images[0].set_zorder(0)
    for (iy, ix), v in np.ndenumerate(surf):
        if v >= 0.03:
            ax.text((ix + 0.5) * 7.5, (iy + 0.5) * 80 / 12, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="#ffffff" if v > 0.06 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("xT = P(goal within the possession's next actions), sqrt colour scale")
    cb.outline.set_visible(False)
    ax.set_title("Expected threat surface, attacking left to right (16 x 12 zones)")
    plots.save(fig, FIGS / "xt_surface.png")


def fig_ranking(top):
    t = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    yy = np.arange(len(t))
    ax.barh(yy, t["xt_pass_p90"], color=SERIES[0], height=0.62, label="passes")
    ax.barh(yy, t["xt_carry_p90"], left=t["xt_pass_p90"], color=SERIES[2], height=0.62,
            label="carries", ec=plots.SURFACE, lw=1.5)
    ax.hlines(yy, t["p90_lo"], t["p90_hi"], color=INK, lw=1.2, label="90% match-bootstrap interval")
    ax.set_yticks(yy, [f"{p}  ({int(m)} min)" for p, m in zip(t["player"], t["minutes"])], fontsize=8.5)
    ax.set_xlabel("xT added per 90 minutes (successful open-play passes + carries)")
    ax.set_title(f"Top 15 ball progressors, min. {MIN_MINUTES} minutes")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    plots.save(fig, FIGS / "player_xt_ranking.png")


# --------------------------------------------------------------------------- finishing
def run_finishing(np_shots, minutes, summary):
    names = minutes.groupby("player_id")["player"].first()  # lineup nicknames ("Cody Gakpo")
    np_shots = np_shots.assign(player=np_shots["player_id"].map(names).fillna(np_shots["player"]))
    grp = np_shots.groupby("player_id")
    df = grp.agg(player=("player", "first"), shots=("goal", "size"), goals=("goal", "sum"),
                 xg=("xg_freeze", "sum"))
    probs = [g["xg_freeze"].to_numpy() for _, g in grp]
    a = shrinkage.fit_prior(df["goals"], df["xg"])
    post = shrinkage.posterior(df["goals"], df["xg"], a)
    post.index = df.index
    df = df.join(post)
    share = shrinkage.signal_share(df["goals"].to_numpy(), probs)
    # The MLE sits at "no detectable skill". To be generous to finishers, also shrink with the
    # largest prior sd the data cannot reject (95% profile-likelihood upper bound).
    sd_mle, sd_hi = shrinkage.sd_upper_bound(df["goals"], df["xg"])
    gen = shrinkage.posterior(df["goals"], df["xg"], 1 / sd_hi**2)
    gen.index = df.index
    df = df.join(gen[["theta_mean", "theta_lo", "theta_hi", "shrunk_gax"]].add_suffix("_generous"))
    # Same exercise with StatsBomb's xG as the yardstick (robustness check).
    sb = grp.agg(goals=("goal", "sum"), xg=("statsbomb_xg", "sum"))
    a_sb = shrinkage.fit_prior(sb["goals"], sb["xg"])
    df.sort_values("raw_gax", ascending=False).to_csv(RESULTS / "finishing_shrinkage.csv", float_format="%.4f")

    top = df.sort_values("raw_gax", ascending=False).head(12)
    summary["finishing"] = {
        "players": int(len(df)),
        "prior_concentration_a": r(a, 2),
        "prior_sd_of_finishing_multiplier": r(1 / np.sqrt(a)),
        "prior_concentration_a_with_statsbomb_xg": r(a_sb, 2),
        "prior_sd_mle_grid": r(sd_mle),
        "prior_sd_upper95": r(sd_hi),
        "moment_check": {k: r(v) for k, v in share.items()},
        "top_raw_overperformers": [
            {"player": p.player, "shots": int(p.shots), "goals": int(p.goals), "xg": r(p.xg, 2),
             "raw_gax": r(p.raw_gax, 2), "shrunk_gax": r(p.shrunk_gax, 2),
             "theta": r(p.theta_mean, 2), "shrunk_gax_generous": r(p.shrunk_gax_generous, 2),
             "theta_generous": r(p.theta_mean_generous, 2),
             "theta90_generous": [r(p.theta_lo_generous, 2), r(p.theta_hi_generous, 2)]}
            for p in top.itertuples()],
        "mean_retained_share_top12_generous": r((top["shrunk_gax_generous"] / top["raw_gax"]).mean()),
        "players_with_theta90_generous_above_1": int((df["theta_lo_generous"] > 1).sum()),
    }
    fig_finishing(top, sd_hi)


def fig_finishing(top, sd_hi):
    t = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    yy = np.arange(len(t))
    ax.hlines(yy, t["shrunk_gax"], t["raw_gax"], color=MUTED, lw=2)
    ax.scatter(t["shrunk_gax_generous"], yy, color=SERIES[2], s=55, zorder=3, ec=plots.SURFACE, lw=1.5,
               label=f"shrunk, most generous prior the data allow (sd {sd_hi:.2f})")
    ax.scatter(t["raw_gax"], yy, color=SERIES[1], s=55, zorder=3, label="raw goals − xG", ec=plots.SURFACE, lw=1.5)
    ax.scatter(t["shrunk_gax"], yy, color=SERIES[0], s=55, zorder=3, label="shrunk, empirical-Bayes prior (sd ≈ 0)",
               ec=plots.SURFACE, lw=1.5)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(yy, [f"{p}  ({g} G / {x:.1f} xG)" for p, g, x in zip(t["player"], t["goals"], t["xg"])],
                  fontsize=8.5)
    ax.set_xlabel("non-penalty goals above expected")
    ax.set_title("Biggest finishing over-performers, before and after shrinkage")
    ax.grid(axis="y", visible=False)
    h, lab = ax.get_legend_handles_labels()
    order = [1, 0, 2]
    ax.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center",
              bbox_to_anchor=(0.4, -0.12), ncol=1, fontsize=9)
    plots.save(fig, FIGS / "finishing_shrinkage.png")


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    shots = pd.read_parquet(DERIVED / "shots.parquet")
    actions = pd.read_parquet(DERIVED / "actions.parquet")
    minutes = pd.read_parquet(DERIVED / "minutes.parquet")
    matches = pd.read_parquet(DERIVED / "matches.parquet")
    summary = {"competitions": matches.groupby(["competition", "season"]).size()
               .rename("matches").reset_index().to_dict(orient="records")}
    np_shots = run_xg(shots, summary)
    run_xt(actions, np_shots, minutes, summary)
    run_finishing(np_shots, minutes, summary)
    np_shots[["shot_id", "match_id", "player", "team", "goal", "statsbomb_xg", "xg_geometry",
              "xg_freeze", "xg_gbm"]].to_parquet(RESULTS / "shot_xg_oof.parquet", index=False)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("xt", "finishing")}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
