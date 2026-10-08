"""Дымовые тесты на *синтетических* рядах: проверяют, что код работает.

Это не результаты на данных СберИндекса — только проверка корректности кода.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sberindex_forecast.changepoints import cusum_alarms, detect_pelt, match_changepoints  # noqa: E402
from sberindex_forecast.config import load_config  # noqa: E402
from sberindex_forecast.evaluation import rolling_origins  # noqa: E402
from sberindex_forecast.models.baselines import seasonal_naive_forecast  # noqa: E402
from sberindex_forecast.pipeline import backtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def synthetic_panel(n_mo: int = 5, months: int = 48, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=months, freq="MS")
    rows = []
    for mo in range(n_mo):
        level = rng.uniform(50, 150)
        season = 10 * np.sin(2 * np.pi * dates.month / 12)
        trend = np.linspace(0, 20, months)
        shock = np.where(np.arange(months) >= 30, -25, 0)  # искусственный шок
        y = level + season + trend + shock + rng.normal(0, 2, months)
        rows += [(mo, "total", d, v) for d, v in zip(dates, y)]
    return pd.DataFrame(rows, columns=["territory_id", "category", "date", "value"])


def test_seasonal_naive():
    s = pd.Series(np.arange(24.0), index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    assert seasonal_naive_forecast(s, 3).tolist() == [12.0, 13.0, 14.0]


def test_rolling_origins_leave_room_for_horizon():
    dates = pd.Series(pd.date_range("2020-01-01", periods=36, freq="MS"))
    cuts = rolling_origins(dates, n_folds=3, step=1, horizon=12)
    assert cuts[-1] + pd.DateOffset(months=12) <= dates.max()


def test_changepoints_find_synthetic_shock():
    y = synthetic_panel(1)["value"].to_numpy()
    cps = detect_pelt(y, model="l2", penalty=500)
    assert match_changepoints([30], cps, tolerance=2)["recall"] == 1.0
    alarms = cusum_alarms(np.r_[np.zeros(30), np.full(10, 5.0)] + np.random.default_rng(1).normal(0, 1, 40))
    assert any(30 <= a <= 33 for a in alarms)


def test_backtest_runs():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["validation"]["n_folds"] = 2
    cfg["forecast"]["horizons"] = [1, 3]
    cfg["models"]["lightgbm"]["n_estimators"] = 50
    preds, metrics = backtest(synthetic_panel(), cfg)
    assert set(metrics["model"]) == {"naive", "seasonal_naive", "lightgbm"}
    assert metrics["MAE"].notna().all()
