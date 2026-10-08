"""Генерация признаков для глобальной модели градиентного бустинга.

Используется прямая (direct) стратегия: для каждого горизонта h отдельная
модель, целевая переменная — значение ряда через h месяцев. Все признаки
строятся только из прошлого относительно точки прогноза, чтобы исключить
утечку будущего.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP = ["territory_id", "category"]


def transform_target(y: pd.Series | np.ndarray, kind: str = "log1p"):
    return np.log1p(y) if kind == "log1p" else y


def inverse_target(y: pd.Series | np.ndarray, kind: str = "log1p"):
    return np.expm1(y) if kind == "log1p" else y


def make_features(
    df: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
    calendar: bool = True,
    yoy_growth: bool = True,
    target_transform: str = "log1p",
) -> pd.DataFrame:
    """Добавить лаги, скользящие статистики, календарь и годовой прирост.

    На вход — длинный DataFrame (territory_id, category, date, value).
    Признак lag_k означает значение ряда k месяцев назад относительно
    строки (сама строка = «последнее известное наблюдение» при k=0).
    """
    out = df.sort_values(GROUP + ["date"]).copy()
    out["y"] = transform_target(out["value"], target_transform)
    g = out.groupby(GROUP, sort=False)["y"]

    out["lag_0"] = out["y"]
    for k in lags:
        out[f"lag_{k}"] = g.shift(k)
    for w in rolling_windows:
        out[f"roll_mean_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        out[f"roll_std_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=2).std())
    if yoy_growth:
        out["yoy_diff"] = out["y"] - g.shift(12)
        out["mom_diff"] = out["y"] - g.shift(1)
    if calendar:
        out["month"] = out["date"].dt.month
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["category_code"] = out["category"].astype("category").cat.codes
    return out


def add_horizon_target(feat: pd.DataFrame, h: int) -> pd.DataFrame:
    """Добавить колонку target_h — значение y через h месяцев и дату цели."""
    out = feat.copy()
    g = out.groupby(GROUP, sort=False)
    out[f"target_{h}"] = g["y"].shift(-h)
    out["target_date"] = out["date"] + pd.DateOffset(months=h)
    return out


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """Список колонок-признаков (всё, кроме ключей и целевых)."""
    drop = {"territory_id", "category", "date", "value", "y", "target_date"}
    return [c for c in feat.columns if c not in drop and not c.startswith("target_")]
