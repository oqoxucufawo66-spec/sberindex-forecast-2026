"""Базовые модели. Все функции векторизованы по рядам: Y — матрица (ряды × месяцы),
o — индекс точки отсчёта, h — горизонт. Возвращают прогноз на месяц o + h.

Prophet — базовая модель организаторов, которую нужно превзойти по MAE.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


def naive(Y: np.ndarray, o: int, h: int) -> np.ndarray:
    """Последнее известное значение."""
    return Y[:, o].copy()


def seasonal_naive(Y: np.ndarray, o: int, h: int) -> np.ndarray:
    """Значение того же месяца год назад (T - 12); если его нет в истории — наивный."""
    t = o + h - 12
    return Y[:, t].copy() if 0 <= t <= o else naive(Y, o, h)


def seasonal_naive_growth(Y: np.ndarray, o: int, h: int) -> np.ndarray:
    """Сезонно-наивный с поправкой на годовой рост: Y[T-12] * Y[o] / Y[o-12].
    Если год назад от точки отсчёта данных нет — обычный сезонно-наивный."""
    t = o + h - 12
    if t >= 0 and o - 12 >= 0:
        return Y[:, t] * Y[:, o] / Y[:, o - 12]
    return seasonal_naive(Y, o, h)


# ---------------- Prophet ----------------

def _prophet_one(args):
    y, months, h_max, params = args
    from prophet import Prophet

    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True
    df = pd.DataFrame({"ds": pd.to_datetime(months), "y": y})
    m = Prophet(**params)
    m.fit(df)
    fut = m.make_future_dataframe(periods=h_max, freq="MS", include_history=False)
    return m.predict(fut)["yhat"].to_numpy()


def prophet_paths(Y: np.ndarray, months: list[str], o: int, h_max: int, params: dict,
                  workers: int | None = None) -> np.ndarray:
    """Prophet по каждому ряду на истории [0..o]; возвращает (n_series, h_max)."""
    hist_months = [m + "-01" for m in months[: o + 1]]
    jobs = [(Y[i, : o + 1], hist_months, h_max, params) for i in range(Y.shape[0])]
    workers = workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_prophet_one, jobs, chunksize=16))
    return np.vstack(out)
