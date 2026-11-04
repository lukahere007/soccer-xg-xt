import pytest

from goalvalue import data


def _ev(t, period, minute, second=0, **kw):
    return {"type": {"name": t}, "period": period, "minute": minute, "second": second, **kw}


def test_period_lengths_and_minutes_with_stoppage_time():
    events = [_ev("Half End", 1, 47), _ev("Half End", 2, 94)]
    lineups = [{"team_id": 1, "lineup": [
        {"player_id": 10, "player_name": "Starter", "positions": [
            {"from": "00:00", "to": None, "from_period": 1, "to_period": None}]},
        {"player_id": 11, "player_name": "Subbed off", "positions": [
            {"from": "00:00", "to": "60:00", "from_period": 1, "to_period": 2}]},
        {"player_id": 12, "player_name": "Sub", "positions": [
            {"from": "60:00", "to": None, "from_period": 2, "to_period": None}]},
        {"player_id": 13, "player_name": "Unused", "positions": []},
    ]}]
    assert data.period_lengths(events) == {1: 47.0, 2: 49.0}
    mins = {r["player_id"]: r["minutes"] for r in data.parse_minutes(lineups, events, 1)}
    assert mins[10] == pytest.approx(96.0)
    assert mins[11] == pytest.approx(47 + 15)
    assert mins[12] == pytest.approx(34.0)
    assert 13 not in mins


def test_parse_shots_skips_shootout_and_flags_goal():
    shot = {"statsbomb_xg": 0.3, "outcome": {"name": "Goal"}, "body_part": {"name": "Head"},
            "type": {"name": "Open Play"}, "technique": {"name": "Normal"}, "freeze_frame": []}
    base = dict(id="a", team={"id": 1, "name": "A"}, player={"id": 9, "name": "P"},
                location=[108, 40], shot=shot, play_pattern={"name": "Regular Play"})
    rows = data.parse_shots([_ev("Shot", 1, 10, **base), _ev("Shot", 5, 121, **base)], 1)
    assert len(rows) == 1
    assert rows[0]["goal"] == 1
    assert rows[0]["distance"] == pytest.approx(12.0)


def test_parse_actions_success_and_set_pieces():
    ev = [
        _ev("Pass", 1, 1, id="p1", team={"id": 1}, player={"id": 9}, location=[50, 40],
            **{"pass": {"end_location": [60, 40]}}),
        _ev("Pass", 1, 2, id="p2", team={"id": 1}, player={"id": 9}, location=[50, 40],
            **{"pass": {"end_location": [60, 40], "outcome": {"name": "Incomplete"}}}),
        _ev("Pass", 1, 3, id="p3", team={"id": 1}, player={"id": 9}, location=[119, 0.1],
            **{"pass": {"end_location": [110, 40], "type": {"name": "Corner"}}}),
        _ev("Carry", 1, 4, id="c1", team={"id": 1}, player={"id": 9}, location=[60, 40],
            carry={"end_location": [70, 40]}),
    ]
    rows = data.parse_actions(ev, 1)
    assert [r["success"] for r in rows] == [True, False, True, True]
    assert [r["set_piece"] for r in rows] == [False, False, True, False]
