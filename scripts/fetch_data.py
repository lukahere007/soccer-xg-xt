"""Download the StatsBomb Open Data subset used in this project.

Raw files are saved gzip-compressed under ``data/raw/`` (git-ignored). Nothing
downloaded is ever executed; files are only parsed as JSON.

Usage::

    python scripts/fetch_data.py              # default competitions (see COMPETITIONS)
    python scripts/fetch_data.py --max-matches 20   # quick smoke test

Data: StatsBomb Open Data, https://github.com/statsbomb/open-data
(free for non-commercial use under the StatsBomb Public Data User Agreement;
please credit StatsBomb as the data source).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# (competition_id, season_id, label). International tournaments, men's and
# women's, all with StatsBomb 360 coverage and complete event data.
COMPETITIONS = [
    (43, 106, "FIFA World Cup 2022"),
    (55, 43, "UEFA Euro 2020"),
    (55, 282, "UEFA Euro 2024"),
    (72, 107, "FIFA Women's World Cup 2023"),
    (53, 106, "UEFA Women's Euro 2022"),
    (53, 315, "UEFA Women's Euro 2025"),
]


def _get(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retry on any network error
            if attempt == retries - 1:
                raise RuntimeError(f"failed to download {url}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _save_json_gz(payload: bytes, path: Path) -> None:
    json.loads(payload)  # validate it is JSON before writing
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(payload)


def fetch(max_matches: int | None = None, workers: int = 8) -> None:
    jobs: list[tuple[str, Path]] = []
    for comp, season, label in COMPETITIONS:
        out = RAW / "matches" / f"{comp}_{season}.json.gz"
        if not out.exists():
            _save_json_gz(_get(f"{BASE}/matches/{comp}/{season}.json"), out)
        with gzip.open(out) as fh:
            matches = json.load(fh)
        ids = sorted(m["match_id"] for m in matches)[:max_matches]
        print(f"{label}: {len(ids)} matches", file=sys.stderr)
        for mid in ids:
            jobs.append((f"{BASE}/events/{mid}.json", RAW / "events" / f"{mid}.json.gz"))
            jobs.append((f"{BASE}/lineups/{mid}.json", RAW / "lineups" / f"{mid}.json.gz"))

    todo = [(u, p) for u, p in jobs if not p.exists()]
    print(f"{len(todo)} files to download ({len(jobs) - len(todo)} cached)", file=sys.stderr)

    def work(job: tuple[str, Path]) -> None:
        url, path = job
        _save_json_gz(_get(url), path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(work, todo), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-matches", type=int, default=None, help="per competition (for smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    fetch(args.max_matches, args.workers)
