"""Сквозной бэктест: базовые модели и LightGBM на rolling-origin срезах."""
from __future__ import annotations

import pandas as pd

from .evaluation import rolling_origins, summarize
from .features import make_features
from .models.baselines import naive_forecast, prophet_forecast, seasonal_naive_forecast
from .models.gbm import DirectGBM

GROUP = ["territory_id", "category"]


def backtest(df: pd.DataFrame, cfg: dict, models: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Запустить бэктест.

    Returns
    -------
    preds : длинная таблица прогнозов (model, horizon, origin, territory_id, category, y_true, y_pred)
    metrics : сводная таблица MAE / R² / WAPE по моделям и горизонтам
    """
    fc, val, mcfg = cfg["forecast"], cfg["validation"], cfg["models"]
    models = models or ["naive", "seasonal_naive", "lightgbm"]
    feat = make_features(df, target_transform=fc["target_transform"], **cfg["features"])
    actual = df.set_index(GROUP + ["date"])["value"]
    rows = []

    for h in fc["horizons"]:
        origins = rolling_origins(df["date"], val["n_folds"], val["step"], h)
        for origin in origins:
            target_date = origin + pd.DateOffset(months=h)
            # --- локальные базовые модели (по каждому ряду) ---
            for key, series in df[df["date"] <= origin].groupby(GROUP):
                hist = series.set_index("date")["value"].asfreq("MS")
                y_true = actual.get((*key, target_date))
                if y_true is None or hist.isna().all():
                    continue
                hist = hist.ffill()
                for name in ("naive", "seasonal_naive", "prophet"):
                    if name not in models:
                        continue
                    if name == "naive":
                        p = naive_forecast(hist, h)
                    elif name == "seasonal_naive":
                        p = seasonal_naive_forecast(hist, h, fc["season_length"])
                    else:
                        p = prophet_forecast(hist, h, **mcfg["prophet"])
                    rows.append((name, h, origin, *key, y_true, float(p[h - 1])))
            # --- глобальная модель ---
            if "lightgbm" in models:
                gbm = DirectGBM([h], mcfg["lightgbm"], fc["target_transform"]).fit(feat, origin)
                pred = gbm.predict(feat, origin, h)
                for r in pred.itertuples(index=False):
                    y_true = actual.get((r.territory_id, r.category, target_date))
                    if y_true is not None:
                        rows.append(("lightgbm", h, origin, r.territory_id, r.category, y_true, r.y_pred))

    preds = pd.DataFrame(
        rows, columns=["model", "horizon", "origin", "territory_id", "category", "y_true", "y_pred"]
    )
    return preds, summarize(preds)
