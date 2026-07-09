"""Load NT Historical 'Last' tick exports into tick_data_full (no ts-dedup).

The existing emini `tick_data` table has a UNIQUE(symbol, ts) constraint; the
original load deduped by timestamp and dropped every trade that shared a
microsecond stamp with another — losing ~56% of true volume (NT emits many
fills at the same ts). That silently broke every footprint strategy. This loader
preserves EVERY trade row into a separate `tick_data_full` table (no unique-ts
constraint) so volume matches NT.

Export format (NT Tools > Historical Data > Export, Type=Last):
    yyyyMMdd HHmmss fffffff;price;bid;ask;volume   (semicolon, UTC, open-floored)

Usage: python scripts/load_nt_ticks_full.py <symbol> <export.txt> [<export2.txt> ...]
Example: python scripts/load_nt_ticks_full.py MNQ_M26 "/mnt/f/MNQ/MNQ 06-26.Last.txt"
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import text
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_platform.data import loader  # noqa: E402

DDL = """
CREATE TABLE IF NOT EXISTS tick_data_full (
    id      BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    symbol  VARCHAR(10)  NOT NULL,
    ts      DATETIME(6)  NOT NULL,
    price   DECIMAL(14,4) NOT NULL,
    bid     DECIMAL(14,4) NULL,
    ask     DECIMAL(14,4) NULL,
    volume  INT UNSIGNED NOT NULL,
    KEY idx_symbol_ts (symbol, ts)
) ENGINE=InnoDB
"""


def _parse_chunk(df: pd.DataFrame) -> pd.DataFrame:
    d = df["ts"].str.slice(0, 8)
    hms = df["ts"].str.slice(9, 15)
    frac = df["ts"].str.slice(16)
    df["ts"] = pd.to_datetime(
        d + hms + frac.str.pad(7, side="right", fillchar="0"),
        format="%Y%m%d%H%M%S%f")
    return df[["ts", "price", "bid", "ask", "volume"]]


def _nt_ts(raw: str) -> pd.Timestamp:
    """Parse an NT export timestamp 'yyyyMMdd HHmmss fffffff'."""
    d, hms, frac = raw.split(" ")
    return pd.to_datetime(d + hms + frac.ljust(7, "0"), format="%Y%m%d%H%M%S%f")


def _file_span(p):
    """First + last tick timestamp of an export, read cheaply (head + tail)."""
    with open(p, "rb") as f:
        first = f.readline().decode()
        f.seek(max(0, f.seek(0, 2) - 4096))
        last = [ln for ln in f.read().decode().splitlines() if ln.strip()][-1]
    return _nt_ts(first.split(";")[0]), _nt_ts(last.split(";")[0])


def load(symbol: str, paths, host=None, read_rows=500000, insert_rows=20000):
    """Stream each export in row-chunks so a multi-GB file never loads whole.

    Idempotency: clear EXACTLY the file's own [first_ts, last_ts] span up
    front, then append streamed chunks. The delete used to be a blanket
    first_day+45-days window, which silently destroyed ADJACENT months of the
    same symbol when a shorter export was (re)loaded — e.g. loading a June
    export would have wiped already-loaded July rows whose source file had
    been overwritten (2026-07-09 near-miss).
    """
    eng = loader._engine(host)
    sym = symbol.upper()
    with eng.begin() as c:
        c.execute(text(DDL))
    total = 0
    for p in paths:
        t0, t1 = _file_span(p)
        print(f"  {Path(p).name}: span {t0} -> {t1}", flush=True)
        with eng.begin() as c:
            c.execute(text("DELETE FROM tick_data_full WHERE symbol=:s "
                           "AND ts>=:a AND ts<=:b"),
                      {"s": sym, "a": str(t0), "b": str(t1)})
        vol = 0
        reader = pd.read_csv(p, sep=";", header=None,
                             names=["ts", "price", "bid", "ask", "volume"],
                             chunksize=read_rows)
        for ci, chunk in enumerate(reader):
            chunk = _parse_chunk(chunk)
            chunk.insert(0, "symbol", sym)
            chunk.to_sql("tick_data_full", eng, if_exists="append",
                         index=False, method="multi", chunksize=insert_rows)
            total += len(chunk)
            vol += int(chunk["volume"].sum())
            if ci % 10 == 0:
                print(f"  ...{total:,} rows ({vol:,} vol) so far", flush=True)
        print(f"  loaded from {Path(p).name}: running total {total:,} rows, {vol:,} vol", flush=True)
    print(f"DONE: {total:,} rows into tick_data_full for {sym}")


if __name__ == "__main__":
    load(sys.argv[1], sys.argv[2:])
