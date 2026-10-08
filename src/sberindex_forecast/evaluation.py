"""Метрики качества и схема валидации по времени.

MAE — обязательная метрика конкурса, R² — дополнительная.
Валидация — rolling origin: модель обучается на данных до среза t
и прогнозирует t+h; срез сдвигается вперёд n_folds раз.
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
    """Взвешенная абсолютная ошибка в % — удобна для сравнения МО разного масштаба."""
    y_true = np.asarray(y_true, dtype=float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - np.asarray(y_pred, dtype=float)).sum() / denom * 100) if denom else np.nan


def score(y_true, y_pred) -> dict[str, float]:
    return {"MAE": mae(y_true, y_pred), "R2": r2(y_true, y_pred), "WAPE_%": wape(y_true, y_pred)}


def rolling_origins(dates: pd.Series, n_folds: int, step: int, horizon: int) -> list[pd.Timestamp]:
    """Вернуть даты срезов (последняя известная дата) для rolling-origin валидации.

    Последний срез выбирается так, чтобы для него существовал факт через horizon месяцев.
    """
    uniq = np.sort(pd.Series(dates).unique())
    last_cut_idx = len(uniq) - 1 - horizon
    idx = [last_cut_idx - i * step for i in range(n_folds)]
    idx = sorted(i for i in idx if i >= 0)
    return [pd.Timestamp(uniq[i]) for i in idx]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Свести таблицу прогнозов (model, horizon, y_true, y_pred) в метрики."""
    rows = []
    for (model, h), grp in results.groupby(["model", "horizon"]):
        rows.append({"model": model, "horizon": h, **score(grp["y_true"], grp["y_pred"])})
    return pd.DataFrame(rows).sort_values(["horizon", "MAE"]).reset_index(drop=True)
