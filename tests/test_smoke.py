"""Тесты корректности кода. Синтетические ряды — только для проверки логики,
это не результаты на данных СберИндекса."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberindex_forecast.changepoints import cusum_alarms, detect_binseg, detect_pelt  # noqa: E402
from sberindex_forecast.config import load_config  # noqa: E402
from sberindex_forecast.data import Panel  # noqa: E402
from sberindex_forecast.evaluation import summarize  # noqa: E402
from sberindex_forecast.models.baselines import naive, seasonal_naive, seasonal_naive_growth  # noqa: E402
from sberindex_forecast.models.gbm import StrictDirectGBM, features_at  # noqa: E402
from sberindex_forecast.news import event_matrix  # noqa: E402
from sberindex_forecast.pipeline import add_combos, backtest  # noqa: E402

MONTHS = [f"{y}-{m:02d}" for y in (2023, 2024) for m in range(1, 13)]


def synthetic_panel(n_mo=30, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    cats = ["A", "B"]
    rows, Y = [], []
    t = np.arange(24)
    for mo in range(n_mo):
        for c in cats:
            level = rng.uniform(5_000, 30_000)
            y = level * (1 + 0.01 * t) * (1 + 0.1 * np.sin(2 * np.pi * t / 12)) * rng.normal(1, 0.01, 24)
            Y.append(y)
            rows.append({"territory_id": mo, "category": c, "name": f"МО {mo}", "region_name": "Р",
                         "region_code": mo % 3, "market_access": rng.uniform(100, 500)})
    return Panel(pd.DataFrame(rows), np.array(Y), MONTHS)


def test_baselines_use_only_past():
    Y = np.arange(24, dtype=float)[None, :] + 1
    assert naive(Y, 10, 3)[0] == 11
    assert seasonal_naive(Y, 20, 3)[0] == Y[0, 11]      # T-12 = 11 <= o
    assert seasonal_naive(Y, 5, 3)[0] == Y[0, 5]        # нет года истории -> наивный
    assert np.isclose(seasonal_naive_growth(Y, 20, 1)[0], Y[0, 9] * Y[0, 20] / Y[0, 8])


def test_gbm_is_strict():
    p = synthetic_panel()
    gbm = StrictDirectGBM({"n_estimators": 20, "verbose": -1})
    pred, _ = gbm.fit_predict(p.Y, p.meta, p.months, 20, 3)
    assert pred.shape == (len(p.Y),) and np.isfinite(pred).all()
    assert gbm.fit_predict(p.Y, p.meta, p.months, 11, 12) == (None, None)  # нет обучающих пар
    f = features_at(p.Y, p.meta, p.months, 5, 1)
    Y2 = p.Y.copy(); Y2[:, 6:] = 0                       # будущее не должно влиять на признаки
    pd.testing.assert_frame_equal(f, features_at(Y2, p.meta, p.months, 5, 1))


def test_backtest_and_combos():
    p = synthetic_panel()
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["models"]["lightgbm"]["n_estimators"] = 20
    preds, _ = backtest(p, cfg, ["naive", "seasonal_naive", "snaive_growth", "lightgbm"])
    m = summarize(preds)
    assert set(m.horizon) == {1, 3, 6, 12}
    assert "lightgbm" not in set(m[m.horizon == 12].model)
    assert "combo_gbm_fm" in set(m.model)
    assert (m.MAE >= 0).all()


def test_changepoints_detect_step():
    rng = np.random.default_rng(1)
    x = np.r_[np.zeros(12), np.full(12, 1.0)] + rng.normal(0, 0.1, 24)
    for fn in (detect_pelt, detect_binseg):
        assert any(abs(c - 12) <= 1 for c in fn(x, penalty=3.0))
    assert any(11 <= a <= 13 for a in cusum_alarms(x, threshold=4.0))
    assert detect_pelt(rng.normal(0, 0.1, 24), penalty=8.0) == []


def test_event_matrix_no_future():
    meta = pd.DataFrame({"territory_id": [1, 2]})
    ev = pd.DataFrame({"event_id": ["e"], "territory_id": [1], "month": ["2024-04"]})
    E = event_matrix(ev, meta, MONTHS)
    assert E[0, MONTHS.index("2024-03")] == 0 and E[0, MONTHS.index("2024-04")] == 1 and E[1].sum() == 0
