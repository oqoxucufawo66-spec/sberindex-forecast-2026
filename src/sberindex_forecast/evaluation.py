"""Метрики качества и протокол валидации.

Протокол (строгий, без заглядывания в будущее): для каждого тестового месяца T
и горизонта h прогноз строится с точки отсчёта o = T - h. Модель видит только
данные до o включительно — и в признаках, и в обучении.
MAE — обязательная метрика конкурса, R² — дополнительная.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def wape(y_true, y_pred) -> float:
    """Сумма абсолютных ошибок / сумма факта, % — сопоставимо между категориями."""
    y_true = np.asarray(y_true, float)
    return float(np.abs(y_true - np.asarray(y_pred, float)).sum() / np.abs(y_true).sum() * 100)


def r2_change(y_true, y_pred, y_origin) -> float:
    """R² по изменению относительно точки отсчёта: насколько модель объясняет
    *динамику*, а не межмуниципальные различия уровня (обычный R² по уровню
    почти всегда близок к 1, потому что МО сильно различаются по масштабу)."""
    y_true, y_pred, y_origin = (np.asarray(a, float) for a in (y_true, y_pred, y_origin))
    return float(r2_score(y_true / y_origin - 1, y_pred / y_origin - 1))


def origins_for(test_idx: list[int], h: int) -> list[int]:
    return [t - h for t in test_idx]


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    """preds: model, horizon, target, y_true, y_pred, y_origin -> метрики по (model, horizon)."""
    rows = []
    for (m, h), g in preds.groupby(["model", "horizon"]):
        g = g.dropna(subset=["y_pred"])
        if g.empty:
            continue
        rows.append({"model": m, "horizon": h, "n": len(g),
                     "MAE": mae(g.y_true, g.y_pred), "R2": r2(g.y_true, g.y_pred),
                     "R2_change": r2_change(g.y_true, g.y_pred, g.y_origin),
                     "WAPE_%": wape(g.y_true, g.y_pred)})
    return pd.DataFrame(rows).sort_values(["horizon", "MAE"]).reset_index(drop=True)


def win_rate(preds: pd.DataFrame, model: str, ref: str = "prophet") -> pd.DataFrame:
    """Доля прогнозов (ряд × месяц), где |ошибка| модели меньше, чем у ref."""
    key = ["horizon", "series", "target"]
    a = preds[preds.model == model].set_index(key)
    b = preds[preds.model == ref].set_index(key)
    j = a.join(b, lsuffix="_m", rsuffix="_r", how="inner").dropna(subset=["y_pred_m", "y_pred_r"])
    j["win"] = (j.y_pred_m - j.y_true_m).abs() < (j.y_pred_r - j.y_true_r).abs()
    return j.groupby(level="horizon")["win"].mean().rename(f"{model}_beats_{ref}").reset_index()
