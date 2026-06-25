import os
import textwrap
import pandas as pd
import pytest

import parity_check as pc


def test_money_parsing():
    assert pc._money("-$18.98") == -18.98
    assert pc._money("$1,126.02") == 1126.02
    assert pc._money("$0.00") == 0.0


def test_parse_nt_trade_log(tmp_path):
    csv = textwrap.dedent("""\
        Trade number,Instrument,Account,Strategy,Market pos.,Qty,Entry price,Exit price,Entry time,Exit time,Entry name,Exit name,Profit,Cum. net profit,Commission,Clearing Fee,Exchange Fee,IP Fee,NFA Fee,MAE,MFE,ETD,Bars
        1,NQ SEP26,Sim,,Long,1,30815.5,30814.75,16/06/2026 09:45:59,16/06/2026 09:46:00,STF_Long,Stop loss,-$18.98,-$18.98,$3.98,$0,$0,$0,$0,$15.00,$0.00,$18.98,1
        2,NQ SEP26,Sim,,Short,1,30818.25,30828.75,16/06/2026 10:00:13,16/06/2026 10:00:40,STF_Short,Stop loss,-$213.98,-$232.96,$3.98,$0,$0,$0,$0,$210.00,$355.00,$568.98,11
    """)
    p = tmp_path / "trades.csv"
    p.write_text(csv)
    df = pc.parse_nt_trade_log(str(p))
    assert list(df.columns) == ["entry_time", "exit_time", "direction",
                                "entry_price", "exit_price", "pnl"]
    assert len(df) == 2
    # day-first: 16/06 is 16 June, not an error
    assert df.iloc[0]["entry_time"] == pd.Timestamp("2026-06-16 09:45:59")
    assert df.iloc[0]["direction"] == "Long"
    assert df.iloc[0]["entry_price"] == 30815.5
    assert df.iloc[1]["pnl"] == -213.98


def test_parse_nt_ohlc_export(tmp_path):
    # 'YYYYMMDD HHMMSS;O;H;L;C;V', UTC -> ET-naive (UTC-4 in June DST)
    data = "20260616 140000;100.0;101.0;99.5;100.5;10\n20260616 140100;100.5;102.0;100.0;101.5;12\n"
    p = tmp_path / "ohlc.txt"
    p.write_text(data)
    df = pc.parse_nt_ohlc_export(str(p))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # 14:00 UTC in June = 10:00 ET
    assert df.index[0] == pd.Timestamp("2026-06-16 10:00:00")
    assert df.iloc[0]["high"] == 101.0


def test_parse_nt_tick_export(tmp_path):
    # 'YYYYMMDD HHMMSS<frac>;price;bid?;ask?;volume' — STF format: ts;price;...;vol
    # Real lines look like '20260616 040003 0780000;30832.75;30832.75;30833.25;1'
    data = ("20260616 140000 0000000;100.0;100.0;100.25;1\n"
            "20260616 140000 5000000;100.5;100.25;100.5;2\n")
    p = tmp_path / "ticks.txt"
    p.write_text(data)
    df = pc.parse_nt_tick_export(str(p))
    assert list(df.columns) == ["price", "volume"]
    assert df.iloc[0]["price"] == 100.0
    assert df.index[0] == pd.Timestamp("2026-06-16 10:00:00")  # 14:00 UTC -> 10:00 ET
    assert df.iloc[1]["price"] == 100.5
    assert df.iloc[1]["volume"] == 2


def test_ticks_to_bars():
    idx = pd.to_datetime([
        "2026-06-16 10:00:00", "2026-06-16 10:00:01",
        "2026-06-16 10:00:02", "2026-06-16 10:00:03",
    ])
    ticks = pd.DataFrame({"price": [100, 101, 99, 102], "volume": [1, 1, 1, 1]}, index=idx)
    bars = pc.ticks_to_bars(ticks, bar_size=2)
    assert len(bars) == 2
    assert bars.iloc[0]["open"] == 100 and bars.iloc[0]["high"] == 101
    assert bars.iloc[0]["low"] == 100 and bars.iloc[0]["close"] == 101
    assert bars.iloc[1]["open"] == 99 and bars.iloc[1]["high"] == 102
    assert bars.iloc[0]["volume"] == 2
    assert bars.iloc[1]["close"] == 102
    assert bars.iloc[1]["low"] == 99
    assert bars.iloc[1]["volume"] == 2


def test_ticks_to_bars_drops_partial():
    idx = pd.to_datetime([
        "2026-06-16 10:00:00", "2026-06-16 10:00:01",
        "2026-06-16 10:00:02", "2026-06-16 10:00:03",
        "2026-06-16 10:00:04",
    ])
    ticks = pd.DataFrame({"price": [100, 101, 99, 102, 103], "volume": [1, 1, 1, 1, 1]}, index=idx)
    bars = pc.ticks_to_bars(ticks, bar_size=2)
    assert len(bars) == 2  # 5th tick is trailing partial, dropped


def test_parse_nt_tick_export_real_format(tmp_path):
    # Exact real-file format: YYYYMMDD HHMMSS FRACTION;price;bid;ask;volume
    # 04:00:03.078 UTC in June (EDT=UTC-4) -> 00:00:03.078 ET
    data = "20260616 040003 0780000;30832.75;30832.75;30833.25;1\n"
    p = tmp_path / "ticks_real.txt"
    p.write_text(data)
    df = pc.parse_nt_tick_export(str(p))
    assert df.iloc[0]["price"] == 30832.75
    assert df.iloc[0]["volume"] == 1.0
    assert df.index[0] == pd.Timestamp("2026-06-16 00:00:03.078")
