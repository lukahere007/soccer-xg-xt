"""Parse raw StatsBomb JSON (data/raw/) into the derived parquet tables (data/derived/).

Usage::

    python scripts/fetch_data.py && python scripts/build_tables.py
"""
from __future__ import annotations

from pathlib import Path

from goalvalue.data import build_tables

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    out = ROOT / "data" / "derived"
    out.mkdir(parents=True, exist_ok=True)
    tables = build_tables(ROOT / "data" / "raw")
    # the event id is only needed while parsing; drop it to keep the committed file small
    tables["actions"] = tables["actions"].drop(columns=["event_id"])
    for name, df in tables.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        print(f"{name:8s} {len(df):>8,d} rows -> {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
