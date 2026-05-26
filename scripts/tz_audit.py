"""Timezone audit — query the emini database directly to determine the actual
timestamp convention of each table.

Method: for each table, find the bar nearest to a known wall-clock event and
inspect its stored datetime value. Cross-reference with a known NT8 bar.

Known event: NT8 market replay on 2026-05-15 (a Friday) at 10:15 NY ET produced
an entry on MNQ. That entry was on a 5-min bar whose CLOSE time = 10:15:00 ET.
That bar's OPEN time = 10:10:00 ET.

If the DB stores ET-naive: query WHERE datetime = '2026-05-15 10:10:00' on
historical_data_1m → should find a row whose open price matches NT's entry $29307.

If the DB stores UTC-naive: the same NT 10:15 ET bar lives at datetime
'2026-05-15 14:10:00' (DST: ET = UTC-4 in May).

Run from v2 root:
    cd /home/ad/strategy-platform-v2 && python scripts/tz_audit.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, '/home/ad/strategy-platform-v2')

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

load_dotenv('/home/ad/strategy-platform-v2/.env')

DB_HOST = os.getenv('DB_HOST', '192.168.1.228')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_USER = os.getenv('DB_USER', 'adam')
DB_PASS = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'emini')

url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(url)


def banner(t):
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def main():
    banner("1. SCHEMAS — datetime / date column types")
    for tbl in ('historical_data', 'historical_data_1m', 'tick_data'):
        try:
            df = pd.read_sql(f"SHOW COLUMNS FROM {tbl}", engine)
            print(f"\n[{tbl}]")
            print(df[['Field', 'Type', 'Null']].to_string(index=False))
        except Exception as e:
            print(f"\n[{tbl}] ERROR: {e}")

    banner("2. KNOWN-EVENT TEST — MNQ Friday 2026-05-15")
    print("NT8 reports: entry $29307 at 10:15:01 ET. That's the close of the 10:10-10:15 bar.")
    print("On 2026-05-15 (DST), ET = UTC-4. So that bar in UTC = 14:10:00.\n")

    print("[historical_data_1m] - try ET-naive interpretation (10:10 ET):")
    df_et = pd.read_sql(
        text("SELECT datetime, open, high, low, close, volume FROM historical_data_1m "
             "WHERE symbol='MNQ' AND datetime BETWEEN '2026-05-15 10:08:00' AND '2026-05-15 10:18:00' "
             "ORDER BY datetime"),
        engine,
    )
    print(df_et.to_string(index=False) if not df_et.empty else "(no rows)")

    print("\n[historical_data_1m] - try UTC-naive interpretation (14:10 UTC):")
    df_utc = pd.read_sql(
        text("SELECT datetime, open, high, low, close, volume FROM historical_data_1m "
             "WHERE symbol='MNQ' AND datetime BETWEEN '2026-05-15 14:08:00' AND '2026-05-15 14:18:00' "
             "ORDER BY datetime"),
        engine,
    )
    print(df_utc.to_string(index=False) if not df_utc.empty else "(no rows)")

    banner("3. KNOWN-EVENT TEST — historical_data legacy 5M, NQ=F 2026-05-15 09:30 cash open")
    print("Cash open should be a recognizable 5-min bar with high volume around 09:30 ET.")
    print("[historical_data] try ET-naive (09:30 ET):")
    df = pd.read_sql(
        text("SELECT datetime, open, high, low, close, volume FROM historical_data "
             "WHERE symbol='NQ=F' AND datetime BETWEEN '2026-05-15 09:25:00' AND '2026-05-15 09:45:00' "
             "ORDER BY datetime"),
        engine,
    )
    print(df.to_string(index=False) if not df.empty else "(no rows)")

    print("\n[historical_data] try UTC-naive (13:30 UTC):")
    df = pd.read_sql(
        text("SELECT datetime, open, high, low, close, volume FROM historical_data "
             "WHERE symbol='NQ=F' AND datetime BETWEEN '2026-05-15 13:25:00' AND '2026-05-15 13:45:00' "
             "ORDER BY datetime"),
        engine,
    )
    print(df.to_string(index=False) if not df.empty else "(no rows)")

    banner("4. OLDEST 5M data — when does historical_data start? Same convention?")
    df = pd.read_sql(
        text("SELECT symbol, MIN(datetime) AS first_bar, MAX(datetime) AS last_bar, COUNT(*) AS n "
             "FROM historical_data GROUP BY symbol ORDER BY first_bar LIMIT 10"),
        engine,
    )
    print(df.to_string(index=False))

    banner("5. 1M coverage")
    df = pd.read_sql(
        text("SELECT symbol, MIN(datetime) AS first_bar, MAX(datetime) AS last_bar, COUNT(*) AS n "
             "FROM historical_data_1m GROUP BY symbol ORDER BY first_bar LIMIT 10"),
        engine,
    )
    print(df.to_string(index=False))

    banner("6. tick_data sample row")
    try:
        df = pd.read_sql(
            text("SELECT * FROM tick_data WHERE symbol='MNQ' ORDER BY datetime DESC LIMIT 3"),
            engine,
        )
        print(df.to_string(index=False))
    except Exception as e:
        print(f"(error or no tick_data: {e})")


if __name__ == '__main__':
    main()
