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
