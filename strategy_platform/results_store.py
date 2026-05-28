"""
Shared MySQL-backed results store for optimizer runs and saved backtests.

The platform still writes local CSV/JSON artifacts to reports/, but this module
lets multiple machines share the same optimization history and saved backtests
through a dedicated MySQL database.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import OperationalError

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

DEFAULT_RESULTS_DB_NAME = "strategy_results"


def _settings(use_database: bool = True) -> Dict[str, object]:
    host = os.getenv("RESULTS_DB_HOST") or os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("RESULTS_DB_PORT") or os.getenv("DB_PORT", "3306"))
    user = os.getenv("RESULTS_DB_USER") or os.getenv("DB_USER", "adam")
    password = os.getenv("RESULTS_DB_PASSWORD") or os.getenv("DB_PASSWORD", "")
    if use_database:
        database = os.getenv("RESULTS_DB_NAME") or DEFAULT_RESULTS_DB_NAME
    else:
        database = None
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
    }


def _auto_create_db() -> bool:
    raw = (os.getenv("RESULTS_DB_AUTO_CREATE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _engine_from_settings(settings: Dict[str, object]) -> Engine:
    url = URL.create(
        "mysql+pymysql",
        username=str(settings["user"]),
        password=str(settings["password"]),
        host=str(settings["host"]),
        port=int(settings["port"]),
        database=(str(settings["database"]) if settings.get("database") else None),
    )
    return create_engine(url, pool_pre_ping=True)


@lru_cache(maxsize=1)
def _results_engine() -> Engine:
    return _engine_from_settings(_settings(use_database=True))


@lru_cache(maxsize=1)
def _admin_engine() -> Engine:
    return _engine_from_settings(_settings(use_database=False))


def _json_dumps(value: object) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _df_to_csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def _csv_text_to_df(csv_text: str) -> pd.DataFrame:
    if not csv_text:
        return pd.DataFrame()
    return pd.read_csv(StringIO(csv_text))


def ensure_results_store() -> None:
    """Create the shared results database and tables if they do not already exist."""
    db_name = str(_settings(use_database=True)["database"])
    if _auto_create_db():
        try:
            with _admin_engine().begin() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        except OperationalError as e:
            raise RuntimeError(
                f"Could not create results database '{db_name}'. "
                "Your MySQL user likely does not have CREATE DATABASE privileges. "
                f"Create the database manually, then set RESULTS_DB_AUTO_CREATE=0 and rerun. Original error: {e}"
            ) from e

    try:
        with _results_engine().begin() as conn:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sp_optimizer_runs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                strategy_name VARCHAR(128) NOT NULL,
                symbol VARCHAR(64) NOT NULL,
                sym_safe VARCHAR(64) NOT NULL,
                run_ts VARCHAR(13) NOT NULL,
                label VARCHAR(255) NULL,
                run_meta_json LONGTEXT NULL,
                settings_json LONGTEXT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uq_sp_optimizer_runs (strategy_name, symbol, run_ts),
                KEY idx_sp_optimizer_runs_lookup (strategy_name, sym_safe, run_ts)
            )
        """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sp_optimizer_run_stages (
                    run_id BIGINT NOT NULL,
                    stage VARCHAR(8) NOT NULL,
                    csv_text LONGTEXT NOT NULL,
                    row_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, stage),
                    CONSTRAINT fk_sp_optimizer_run_stages_run
                        FOREIGN KEY (run_id) REFERENCES sp_optimizer_runs(id)
                        ON DELETE CASCADE
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sp_backtests (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    strategy_name VARCHAR(128) NOT NULL,
                    symbol VARCHAR(64) NOT NULL,
                    sym_safe VARCHAR(64) NOT NULL,
                    bt_ts VARCHAR(15) NOT NULL,
                    label VARCHAR(255) NULL,
                    payload_json LONGTEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_sp_backtests (strategy_name, symbol, bt_ts),
                    KEY idx_sp_backtests_lookup (strategy_name, sym_safe, bt_ts)
                )
            """))
    except OperationalError as e:
        raise RuntimeError(
            f"Could not connect to or initialize results database '{db_name}'. "
            "If the database already exists, confirm the RESULTS_DB_* settings and "
            "that the current MySQL user has privileges on it. "
            f"Original error: {e}"
        ) from e


def save_optimizer_run(
    strategy_name: str,
    symbol: str,
    run_ts: str,
    run_meta: Optional[dict],
    settings: Optional[dict],
    stage_frames: Dict[str, pd.DataFrame],
    label: Optional[str] = None,
) -> None:
    """Persist one optimizer run plus IS/MC/OOS stage frames."""
    ensure_results_store()
    sym_safe = symbol.replace("=", "_")

    with _results_engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO sp_optimizer_runs (
                strategy_name, symbol, sym_safe, run_ts, label, run_meta_json, settings_json
            ) VALUES (
                :strategy_name, :symbol, :sym_safe, :run_ts, :label, :run_meta_json, :settings_json
            )
            ON DUPLICATE KEY UPDATE
                label = COALESCE(:label, label),
                run_meta_json = VALUES(run_meta_json),
                settings_json = VALUES(settings_json)
        """), {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "sym_safe": sym_safe,
            "run_ts": run_ts,
            "label": label,
            "run_meta_json": _json_dumps(run_meta or {}),
            "settings_json": _json_dumps(settings or {}),
        })

        run_id = conn.execute(text("""
            SELECT id
            FROM sp_optimizer_runs
            WHERE strategy_name = :strategy_name
              AND symbol = :symbol
              AND run_ts = :run_ts
        """), {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "run_ts": run_ts,
        }).scalar_one()

        for stage, df in stage_frames.items():
            if df is None:
                continue
            conn.execute(text("""
                INSERT INTO sp_optimizer_run_stages (run_id, stage, csv_text, row_count)
                VALUES (:run_id, :stage, :csv_text, :row_count)
                ON DUPLICATE KEY UPDATE
                    csv_text = VALUES(csv_text),
                    row_count = VALUES(row_count)
            """), {
                "run_id": run_id,
                "stage": stage.upper(),
                "csv_text": _df_to_csv_text(df),
                "row_count": int(len(df)),
            })


def list_optimizer_run_timestamps(strategy_name: str, sym_safe: str) -> List[str]:
    """Return run timestamps for a strategy/symbol pair, newest first."""
    try:
        ensure_results_store()
        with _results_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT run_ts
                FROM sp_optimizer_runs
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                ORDER BY run_ts DESC
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
            }).fetchall()
        return [str(row[0]) for row in rows]
    except Exception:
        return []


def load_optimizer_stage(strategy_name: str, sym_safe: str, run_ts: str, stage: str) -> Optional[pd.DataFrame]:
    """Load a stored IS/MC/OOS DataFrame from the shared store."""
    try:
        ensure_results_store()
        with _results_engine().connect() as conn:
            csv_text = conn.execute(text("""
                SELECT s.csv_text
                FROM sp_optimizer_run_stages s
                JOIN sp_optimizer_runs r ON r.id = s.run_id
                WHERE r.strategy_name = :strategy_name
                  AND r.sym_safe = :sym_safe
                  AND r.run_ts = :run_ts
                  AND s.stage = :stage
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
                "run_ts": run_ts,
                "stage": stage.upper(),
            }).scalar_one_or_none()
        if csv_text is None:
            return None
        return _csv_text_to_df(str(csv_text))
    except Exception:
        return None


def get_run_label(strategy_name: str, sym_safe: str, run_ts: str) -> str:
    try:
        ensure_results_store()
        with _results_engine().connect() as conn:
            label = conn.execute(text("""
                SELECT label
                FROM sp_optimizer_runs
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                  AND run_ts = :run_ts
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
                "run_ts": run_ts,
            }).scalar_one_or_none()
        return str(label or "")
    except Exception:
        return ""


def load_optimizer_run_settings(strategy_name: str, sym_safe: str, run_ts: str) -> dict:
    """Load saved settings_json for one optimizer run."""
    try:
        ensure_results_store()
        with _results_engine().connect() as conn:
            payload = conn.execute(text("""
                SELECT settings_json
                FROM sp_optimizer_runs
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                  AND run_ts = :run_ts
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
                "run_ts": run_ts,
            }).scalar_one_or_none()
        return json.loads(payload) if payload else {}
    except Exception:
        return {}


def set_run_label(strategy_name: str, sym_safe: str, run_ts: str, label: str) -> None:
    try:
        ensure_results_store()
        with _results_engine().begin() as conn:
            conn.execute(text("""
                UPDATE sp_optimizer_runs
                SET label = :label
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                  AND run_ts = :run_ts
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
                "run_ts": run_ts,
                "label": label.strip() or None,
            })
    except Exception:
        return


def save_backtest(
    strategy_name: str,
    symbol: str,
    bt_ts: str,
    payload: dict,
    label: Optional[str] = None,
) -> None:
    """Persist one saved manual backtest payload."""
    ensure_results_store()
    sym_safe = symbol.replace("=", "_")
    with _results_engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO sp_backtests (
                strategy_name, symbol, sym_safe, bt_ts, label, payload_json
            ) VALUES (
                :strategy_name, :symbol, :sym_safe, :bt_ts, :label, :payload_json
            )
            ON DUPLICATE KEY UPDATE
                label = COALESCE(:label, label),
                payload_json = VALUES(payload_json)
        """), {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "sym_safe": sym_safe,
            "bt_ts": bt_ts,
            "label": label,
            "payload_json": _json_dumps(payload),
        })


def list_backtests(strategy_name: str, sym_safe: str) -> List[str]:
    try:
        ensure_results_store()
        with _results_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT bt_ts
                FROM sp_backtests
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                ORDER BY bt_ts DESC
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
            }).fetchall()
        return [str(row[0]) for row in rows]
    except Exception:
        return []


def load_backtest(strategy_name: str, sym_safe: str, bt_ts: str) -> Optional[dict]:
    try:
        ensure_results_store()
        with _results_engine().connect() as conn:
            payload = conn.execute(text("""
                SELECT payload_json
                FROM sp_backtests
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                  AND bt_ts = :bt_ts
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
                "bt_ts": bt_ts,
            }).scalar_one_or_none()
        if payload is None:
            return None
        return json.loads(str(payload))
    except Exception:
        return None


def get_backtest_label(strategy_name: str, sym_safe: str, bt_ts: str) -> str:
    try:
        ensure_results_store()
        with _results_engine().connect() as conn:
            label = conn.execute(text("""
                SELECT label
                FROM sp_backtests
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                  AND bt_ts = :bt_ts
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
                "bt_ts": bt_ts,
            }).scalar_one_or_none()
        return str(label or "")
    except Exception:
        return ""


def set_backtest_label(strategy_name: str, sym_safe: str, bt_ts: str, label: str) -> None:
    try:
        ensure_results_store()
        with _results_engine().begin() as conn:
            conn.execute(text("""
                UPDATE sp_backtests
                SET label = :label
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                  AND bt_ts = :bt_ts
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
                "bt_ts": bt_ts,
                "label": label.strip() or None,
            })
    except Exception:
        return


def delete_optimizer_run(strategy_name: str, sym_safe: str, run_ts: str) -> None:
    """Delete an optimizer run and all its stages (IS/MC/OOS) from the shared results store."""
    import glob as _glob
    import os as _os
    try:
        ensure_results_store()
        with _results_engine().begin() as conn:
            run_id_row = conn.execute(text("""
                SELECT id FROM sp_optimizer_runs
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                  AND run_ts = :run_ts
            """), {"strategy_name": strategy_name, "sym_safe": sym_safe, "run_ts": run_ts}).fetchone()
            if run_id_row:
                run_id = run_id_row[0]
                conn.execute(text("DELETE FROM sp_optimizer_run_stages WHERE run_id = :run_id"),
                             {"run_id": run_id})
                conn.execute(text("DELETE FROM sp_optimizer_runs WHERE id = :run_id"),
                             {"run_id": run_id})
    except Exception:
        pass
    # Also remove CSV files if present (check strategy subfolder + flat root fallback).
    try:
        reports_dir   = _os.path.join(_os.path.dirname(__file__), '..', 'reports')
        strat_dir     = _os.path.join(reports_dir, strategy_name)
        for stage in ("IS", "MC", "OOS"):
            for d in (strat_dir, reports_dir):
                for f in _glob.glob(_os.path.join(d, f"{stage}_{strategy_name}_{sym_safe}_{run_ts}.csv")):
                    try:
                        _os.remove(f)
                    except Exception:
                        pass
    except Exception:
        pass


def delete_backtest(strategy_name: str, sym_safe: str, bt_ts: str) -> None:
    """Delete a backtest row from the shared results store."""
    try:
        ensure_results_store()
        with _results_engine().begin() as conn:
            conn.execute(text("""
                DELETE FROM sp_backtests
                WHERE strategy_name = :strategy_name
                  AND sym_safe = :sym_safe
                  AND bt_ts = :bt_ts
            """), {
                "strategy_name": strategy_name,
                "sym_safe": sym_safe,
                "bt_ts": bt_ts,
            })
    except Exception:
        return
