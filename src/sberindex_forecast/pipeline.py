"""Сквозной бэктест всех моделей по строгому протоколу (см. evaluation.py)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import Panel
from .models.baselines import naive, prophet_paths, seasonal_naive, seasonal_naive_growth
from .models.gbm import StrictDirectGBM, features_at, shap_importance

SIMPLE = {"naive": naive, "seasonal_naive": seasonal_naive, "snaive_growth": seasonal_naive_growth}


def backtest(panel: Panel, cfg: dict, models: list[str], log=print):
    """Возвращает (preds, shap_tables). preds — длинная таблица прогнозов."""
    Y, months = panel.Y, panel.months
    test_idx = [panel.idx(m) for m in cfg["protocol"]["test_months"]]
    horizons = cfg["protocol"]["horizons"]
    h_max = max(horizons)
    jobs: dict[int, list[int]] = {}
    for h in horizons:
        for t in test_idx:
            jobs.setdefault(t - h, []).append(h)

    chronos = None
    if "chronos" in models:
        from .models.foundation import ChronosBolt
        cc = cfg["models"]["chronos"]
        chronos = ChronosBolt(cc["model_id"], cc["device"], cc["batch_size"])

    hybrid = None
    if "chronos2_hybrid" in models:
        from .models.foundation import Chronos2Hybrid
        hc = cfg["models"]["chronos2_hybrid"]
        hybrid = Chronos2Hybrid(hc["model_id"], hc["device"], hc["batch_size"])
    cats = panel.meta["category"].to_numpy()

    gbm = StrictDirectGBM(cfg["models"]["lightgbm"], cfg.get("seed", 42))
    rows, shap_tables = [], {}
    series = np.arange(Y.shape[0])

    def add(name, h, o, pred):
        t = o + h
        rows.append(pd.DataFrame({"model": name, "horizon": h, "series": series,
                                  "origin": months[o], "target": months[t],
                                  "y_true": Y[:, t], "y_pred": pred, "y_origin": Y[:, o]}))

    for o in sorted(jobs):
        t0 = time.time()
        hs = sorted(set(jobs[o]))
        for name, fn in SIMPLE.items():
            if name in models:
                for h in hs:
                    add(name, h, o, fn(Y, o, h))
        if chronos is not None:
            paths = chronos.paths(Y, o, h_max)
            for h in hs:
                add("chronos_bolt", h, o, paths[:, h - 1])
        if hybrid is not None:
            paths = hybrid.paths(Y, cats, o, h_max)
            for h in hs:
                add("chronos2_hybrid", h, o, paths[:, h - 1])
        if "prophet" in models:
            paths = prophet_paths(Y, months, o, h_max, cfg["models"]["prophet"])
            for h in hs:
                add("prophet", h, o, paths[:, h - 1])
        if "lightgbm" in models:
            for h in hs:
                pred, model = gbm.fit_predict(Y, panel.meta, months, o, h)
                if pred is None:
                    continue
                add("lightgbm", h, o, pred)
                if o + h == test_idx[-1]:  # SHAP для последнего тестового месяца
                    shap_tables[h] = shap_importance(model, features_at(Y, panel.meta, months, o, h))
        log(f"origin {months[o]} h={hs} done in {time.time() - t0:.0f}s")

    preds = pd.concat(rows, ignore_index=True)
    preds = add_combos(preds, cfg.get("combos", {}))
    return preds, shap_tables


def add_combos(preds: pd.DataFrame, combos: dict) -> pd.DataFrame:
    """Равновзвешенные комбинации моделей (веса не подбираются на тесте).
    Для каждой комбинации используются те её компоненты, что есть на данном горизонте."""
    if not combos:
        return preds
    key = ["horizon", "series", "target"]
    wide = preds.pivot_table(index=key, columns="model", values="y_pred")
    base = preds.drop_duplicates(key).set_index(key)[["origin", "y_true", "y_origin"]]
    out = [preds]
    for name, members in combos.items():
        cols = [m for m in members if m in wide.columns]
        if not cols:
            continue
        e = wide[cols].mean(axis=1, skipna=True).rename("y_pred")
        out.append(base.join(e).assign(model=name).reset_index())
    return pd.concat(out, ignore_index=True)
