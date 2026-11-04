"""Turn raw StatsBomb JSON into three tidy tables.

* **shots**   one row per shot (penalty shoot-outs excluded) with geometry and
  freeze-frame features plus StatsBomb's own ``statsbomb_xg`` for reference.
* **actions** one row per pass or carry, with start/end location and success.
* **minutes** minutes played per player per match (from lineup position spells).

The raw files are the gzip-compressed JSON written by ``scripts/fetch_data.py``.
Only JSON parsing happens here; nothing downloaded is executed.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as geo

SET_PIECE_PASS_TYPES = {"Corner", "Free Kick", "Kick Off", "Goal Kick", "Throw-in"}
PERIOD_START_MIN = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0}


def _load(path: Path):
    with gzip.open(path) as fh:
        return json.load(fh)


def _name(d, key):
    v = d.get(key)
    return v.get("name") if isinstance(v, dict) else None


def parse_shots(events: list[dict], match_id: int) -> list[dict]:
    """Extract shot rows (excluding the penalty shoot-out, period 5)."""
    rows = []
    for e in events:
        if _name(e, "type") != "Shot" or e.get("period") == 5:
            continue
        s = e["shot"]
        x, y = e["location"][:2]
        row = {
            "shot_id": e["id"],
            "match_id": match_id,
            "team_id": e["team"]["id"],
            "team": e["team"]["name"],
            "player_id": e["player"]["id"],
            "player": e["player"]["name"],
            "period": e["period"],
            "minute": e["minute"],
            "x": x,
            "y": y,
            "distance": float(geo.distance_to_goal(x, y)),
            "angle": float(geo.goal_angle(x, y)),
            "body_part": _name(s, "body_part"),
            "shot_type": _name(s, "type"),
            "technique": _name(s, "technique"),
            "play_pattern": _name(e, "play_pattern"),
            "under_pressure": bool(e.get("under_pressure", False)),
            "first_time": bool(s.get("first_time", False)),
            "outcome": _name(s, "outcome"),
            "goal": int(_name(s, "outcome") == "Goal"),
            "statsbomb_xg": s.get("statsbomb_xg"),
            "key_pass_id": s.get("key_pass_id"),
        }
        row.update(geo.freeze_frame_features((x, y), s.get("freeze_frame")))
        rows.append(row)
    return rows


def parse_actions(events: list[dict], match_id: int) -> list[dict]:
    """Extract passes and carries with start/end location and success flag.

    A pass is successful when StatsBomb records no ``outcome`` (completed).
    Carries are always successful (a lost ball is a separate event).
    ``set_piece`` flags corners, free kicks, kick-offs, goal kicks and throw-ins.
    """
    rows = []
    for e in events:
        t = _name(e, "type")
        if t not in ("Pass", "Carry") or e.get("period") == 5 or "location" not in e:
            continue
        body = e["pass"] if t == "Pass" else e["carry"]
        end = body.get("end_location")
        if end is None:
            continue
        ptype = _name(body, "type") if t == "Pass" else None
        rows.append(
            {
                "event_id": e["id"],
                "match_id": match_id,
                "team_id": e["team"]["id"],
                "player_id": e["player"]["id"],
                "action": t,
                "x": e["location"][0],
                "y": e["location"][1],
                "end_x": end[0],
                "end_y": end[1],
                "success": bool(t == "Carry" or body.get("outcome") is None),
                "set_piece": bool(ptype in SET_PIECE_PASS_TYPES),
            }
        )
    return rows


def _clock_to_min(s: str | None) -> float | None:
    if s is None:
        return None
    mm, ss = s.split(":")
    return int(mm) + int(ss) / 60.0


def period_lengths(events: list[dict]) -> dict[int, float]:
    """Length in minutes of each played period (1-4), from the Half End events."""
    out = {}
    for e in events:
        p = e.get("period")
        if _name(e, "type") == "Half End" and p in PERIOD_START_MIN:
            end = e["minute"] + e["second"] / 60.0
            out[p] = max(out.get(p, 0.0), end - PERIOD_START_MIN[p])
    return out


def _elapsed(clock: float, period: int, lengths: dict[int, float]) -> float:
    """Convert a match-clock minute in ``period`` to minutes actually elapsed."""
    period = min(period, 4)
    before = sum(lengths.get(p, 0.0) for p in range(1, period))
    within = min(max(clock - PERIOD_START_MIN[period], 0.0), lengths.get(period, 0.0))
    return before + within


def parse_minutes(lineups: list[dict], events: list[dict], match_id: int) -> list[dict]:
    """Minutes played per player, summing every position spell in the lineup file.

    Stoppage time is counted (the match clock restarts at 45:00 for the second
    half, so clock times are converted to elapsed time per period). A spell with
    no ``to`` runs to the end of the final period played (shoot-outs excluded).
    """
    lengths = period_lengths(events)
    if not lengths:
        return []
    last = max(lengths)
    total = sum(lengths.values())
    rows = []
    for team in lineups:
        for pl in team["lineup"]:
            mins = 0.0
            for pos in pl.get("positions", []):
                a = _elapsed(_clock_to_min(pos["from"]), pos["from_period"], lengths)
                if pos.get("to") is None:
                    b = total
                else:
                    b = _elapsed(_clock_to_min(pos["to"]), pos.get("to_period") or last, lengths)
                mins += max(b - a, 0.0)
            if mins > 0:
                rows.append(
                    {
                        "match_id": match_id,
                        "team_id": team["team_id"],
                        "player_id": pl["player_id"],
                        "player": pl.get("player_nickname") or pl["player_name"],
                        "minutes": min(mins, total),
                    }
                )
    return rows


def build_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Parse every downloaded match into ``shots``, ``actions``, ``minutes``, ``matches``."""
    raw_dir = Path(raw_dir)
    match_meta = []
    for f in sorted((raw_dir / "matches").glob("*.json.gz")):
        comp_season = f.name.split(".")[0]
        for m in _load(f):
            match_meta.append(
                {
                    "match_id": m["match_id"],
                    "competition": m["competition"]["competition_name"],
                    "season": m["season"]["season_name"],
                    "comp_season": comp_season,
                    "match_date": m["match_date"],
                    "home_team": m["home_team"]["home_team_name"],
                    "away_team": m["away_team"]["away_team_name"],
                }
            )
    matches = pd.DataFrame(match_meta)
    shots, actions, minutes = [], [], []
    for mid in matches["match_id"]:
        ev_path = raw_dir / "events" / f"{mid}.json.gz"
        lu_path = raw_dir / "lineups" / f"{mid}.json.gz"
        if not ev_path.exists() or not lu_path.exists():
            continue
        events = _load(ev_path)
        shots += parse_shots(events, mid)
        actions += parse_actions(events, mid)
        minutes += parse_minutes(_load(lu_path), events, mid)
    matches = matches[matches["match_id"].isin({r["match_id"] for r in shots})]
    shots_df = pd.DataFrame(shots).merge(
        matches[["match_id", "competition", "season"]], on="match_id", how="left"
    )
    act = pd.DataFrame(actions)
    for c in ("x", "y", "end_x", "end_y"):
        act[c] = act[c].astype(np.float32)
    act["action"] = act["action"].astype("category")
    return {
        "shots": shots_df,
        "actions": act,
        "minutes": pd.DataFrame(minutes),
        "matches": matches.reset_index(drop=True),
    }
