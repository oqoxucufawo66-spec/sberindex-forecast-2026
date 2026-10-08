"""Базовые модели: наивная, сезонно-наивная и Prophet.

Prophet — базовая модель, которую по условиям конкурса нужно превзойти
по MAE. Все функции получают историю одного ряда (pd.Series с DatetimeIndex
месячной частоты) и возвращают прогноз на horizon шагов вперёд.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """Последнее известное значение на все шаги вперёд."""
    return np.repeat(float(history.iloc[-1]), horizon)


def seasonal_naive_forecast(history: pd.Series, horizon: int, season_length: int = 12) -> np.ndarray:
    """Значение того же месяца год назад; при короткой истории — наивный прогноз."""
    if len(history) < season_length:
        return naive_forecast(history, horizon)
    last_season = history.iloc[-season_length:].to_numpy(dtype=float)
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon]


def prophet_forecast(history: pd.Series, horizon: int, **params) -> np.ndarray:
    """Прогноз Prophet (месячная частота). Требует пакет `prophet`."""
    from prophet import Prophet  # импорт внутри: пакет тяжёлый и опциональный для тестов

    frame = pd.DataFrame({"ds": history.index, "y": history.to_numpy(dtype=float)})
    model = Prophet(**params)
    model.fit(frame)
    future = model.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
    return model.predict(future)["yhat"].to_numpy()


BASELINES = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "prophet": prophet_forecast,
}
