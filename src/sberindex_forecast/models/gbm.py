"""Глобальная модель LightGBM (одна на все МО и категории) с прямой стратегией.

Целевая переменная — логарифм относительного изменения log(Y[o+h] / Y[o]),
прогноз уровня = Y[o] * exp(prediction). Обучение — L1 с весом Y[o], что
приближает минимизацию MAE в рублях.

Строгость: модель для точки отсчёта o обучается только на парах (o', o'+h)
с o'+h <= o. Поэтому на горизонте 12 мес. при 24 месяцах истории обучить её
честно невозможно (нужны пары, где и признаки, и цель в прошлом) — для h = 12
модель не строится, и это отражено в результатах.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = [
    "cat", "region", "market_access", "log_level", "r1", "r2", "r3", "r6",
    "dev_roll3", "seas_prior", "yoy", "nat_r1", "nat_seas_prior", "target_month", "h",
]


def _safe_log_ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a / b)


def features_at(Y: np.ndarray, meta: pd.DataFrame, months: list[str], o: int, h: int) -> pd.DataFrame:
    """Признаки на точку отсчёта o (используются только Y[:, :o+1])."""
    n = Y.shape[0]
    nan = np.full(n, np.nan)
    lag = lambda k: Y[:, o - k] if o - k >= 0 else nan  # noqa: E731
    f = pd.DataFrame({
        "cat": meta["category"].astype("category").cat.codes.to_numpy(),
        "region": meta["region_code"].fillna(-1).astype(int).to_numpy(),
        "market_access": meta["market_access"].to_numpy(),
        "log_level": np.log(Y[:, o]),
    })
    for k in (1, 2, 3, 6):
        f[f"r{k}"] = _safe_log_ratio(Y[:, o], lag(k))
    f["dev_roll3"] = _safe_log_ratio(Y[:, o], Y[:, max(0, o - 2): o + 1].mean(axis=1))
    # сезонный ориентир: как менялся ряд год назад на том же отрезке (известно на момент o)
    t12, o12 = o + h - 12, o - 12
    f["seas_prior"] = _safe_log_ratio(Y[:, t12], Y[:, o12]) if (o12 >= 0 and t12 <= o) else nan
    f["yoy"] = _safe_log_ratio(Y[:, o], lag(12))
    # общероссийская динамика той же категории (медиана по МО), тоже только прошлое
    cats = meta["category"].to_numpy()
    for col in ("r1", "seas_prior"):
        med = pd.Series(f[col].to_numpy()).groupby(cats).transform("median").to_numpy()
        f["nat_" + col] = med
    f["target_month"] = (int(months[o][5:]) - 1 + h) % 12 + 1
    f["h"] = h
    return f


class StrictDirectGBM:
    def __init__(self, params: dict, seed: int = 42):
        self.params = {**params, "random_state": seed}

    def fit_predict(self, Y: np.ndarray, meta: pd.DataFrame, months: list[str], o: int, h: int):
        """Обучить на парах с целью не позже o и спрогнозировать Y[:, o+h].
        Возвращает (прогноз, модель) или (None, None), если обучающих пар нет."""
        Xs, ys, ws = [], [], []
        for o2 in range(0, o - h + 1):
            X = features_at(Y, meta, months, o2, h)
            Xs.append(X)
            ys.append(np.log(Y[:, o2 + h] / Y[:, o2]))
            ws.append(Y[:, o2])
        if not Xs:
            return None, None
        X, y, w = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)
        model = lgb.LGBMRegressor(**self.params)
        model.fit(X[FEATURES], y, sample_weight=w, categorical_feature=["cat", "region"])
        Xp = features_at(Y, meta, months, o, h)
        return Y[:, o] * np.exp(model.predict(Xp[FEATURES])), model


def shap_importance(model, X: pd.DataFrame, max_rows: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Средний |SHAP| по признакам (на подвыборке строк)."""
    import shap

    Xs = X[FEATURES].sample(min(max_rows, len(X)), random_state=seed)
    vals = shap.TreeExplainer(model).shap_values(Xs)
    return (pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(vals).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
