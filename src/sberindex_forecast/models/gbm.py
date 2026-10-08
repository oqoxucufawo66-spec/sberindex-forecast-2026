"""Глобальная модель LightGBM (одна модель на все МО и категории) по горизонту.

Интерпретация — через SHAP: какие признаки (инерция, сезонность, недавние
сдвиги) определяют прогноз для конкретного муниципалитета.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features import add_horizon_target, feature_columns, inverse_target


class DirectGBM:
    """Прямая стратегия: отдельная LightGBM-модель для каждого горизонта."""

    def __init__(self, horizons: list[int], params: dict, target_transform: str = "log1p"):
        self.horizons = horizons
        self.params = params
        self.target_transform = target_transform
        self.models: dict[int, lgb.LGBMRegressor] = {}
        self.features: list[str] = []

    def fit(self, feat: pd.DataFrame, cutoff: pd.Timestamp) -> "DirectGBM":
        """Обучить модели на строках, у которых дата цели не позже cutoff."""
        self.features = feature_columns(feat)
        for h in self.horizons:
            d = add_horizon_target(feat, h)
            train = d[(d["target_date"] <= cutoff) & d[f"target_{h}"].notna()]
            model = lgb.LGBMRegressor(**self.params)
            model.fit(train[self.features], train[f"target_{h}"])
            self.models[h] = model
        return self

    def predict(self, feat: pd.DataFrame, origin: pd.Timestamp, h: int) -> pd.DataFrame:
        """Прогноз на origin + h месяцев для всех рядов, известных на дату origin."""
        rows = feat[feat["date"] == origin]
        pred = self.models[h].predict(rows[self.features])
        out = rows[["territory_id", "category"]].copy()
        out["target_date"] = origin + pd.DateOffset(months=h)
        out["y_pred"] = inverse_target(pred, self.target_transform)
        return out

    def explain(self, feat: pd.DataFrame, h: int, max_rows: int = 2000) -> pd.DataFrame:
        """Средний |SHAP| по признакам для модели горизонта h."""
        import shap

        sample = feat[self.features].dropna(how="all").tail(max_rows)
        explainer = shap.TreeExplainer(self.models[h])
        values = explainer.shap_values(sample)
        imp = np.abs(values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.features, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
