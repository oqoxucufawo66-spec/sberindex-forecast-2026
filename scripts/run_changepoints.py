"""Сравнение методов обнаружения структурных изменений.

1) Полусинтетический бенчмарк: реальные ряды локальной компоненты + вставленный
   ступенчатый сдвиг известного размера и даты (половина рядов — без вставки,
   контроль ложных тревог). Метрики: доля обнаруженных сдвигов (recall), средняя
   задержка, доля ложных тревог на контрольных рядах, precision.
2) Реальные данные: поиск сдвигов во всех МО (офлайн, по всем 24 месяцам),
   распределение найденных точек по месяцам и регионам, крупнейшие сдвиги.
python scripts/run_changepoints.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sberindex_forecast.changepoints import (cusum_alarms, detect_binseg, detect_pelt,  # noqa: E402
                                             noise_sigma, online_first_detection)
from sberindex_forecast.config import load_config  # noqa: E402
from sberindex_forecast.data import load_panel  # noqa: E402
from sberindex_forecast.models.foundation import national_log_median  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml")
    cp = cfg["changepoints"]
    bm = cp["benchmark"]
    out = ROOT / cfg["output"]["dir"] / "results"
    out.mkdir(parents=True, exist_ok=True)
    panel = load_panel(cfg)
    M = panel.months
    rel = np.log(panel.Y) - national_log_median(panel.Y, panel.meta["category"].to_numpy())

    methods = {
        "PELT": (detect_pelt, {"penalty": cp["pelt"]["penalty"], "min_size": cp["pelt"]["min_size"]}),
        "BinSeg": (detect_binseg, {"penalty": cp["binseg"]["penalty"], "min_size": cp["binseg"]["min_size"]}),
        "CUSUM": (cusum_alarms, {"threshold": cp["cusum"]["threshold"], "drift": cp["cusum"]["drift"],
                                 "warmup": cp["cusum"]["warmup"]}),
    }

    # ---------- 1. полусинтетический бенчмарк ----------
    # Два варианта сигнала: rel (локальная компонента) и rel_yoy = rel[t] - rel[t-12]
    # (дополнительно убирает *местную* сезонность, напр. северный завоз, туристический сезон).
    lo, hi = M.index(bm["shift_months"][0]), M.index(bm["shift_months"][1])
    grid = {
        "PELT": [{"penalty": p, "min_size": cp["pelt"]["min_size"]} for p in bm["penalty_grid"]],
        "BinSeg": [{"penalty": p, "min_size": cp["binseg"]["min_size"]} for p in bm["penalty_grid"]],
        "CUSUM": [{"threshold": t, "drift": cp["cusum"]["drift"], "warmup": cp["cusum"]["warmup"]}
                  for t in bm["cusum_threshold_grid"]],
    }
    rows = []
    for signal in ("rel", "rel_yoy"):
        rng = np.random.default_rng(cfg.get("seed", 42))
        idx = rng.choice(len(rel), size=bm["n_series"], replace=False)
        for k, i in enumerate(idx):
            control = k % 2 == 0
            size = float(rng.choice(bm["shift_sizes"]))
            cp_true = None if control else int(rng.integers(lo, hi + 1))
            sign = rng.choice([-1, 1])
            x = rel[i].copy()
            if cp_true is not None:
                x[cp_true:] += np.log1p(size) * sign
            if signal == "rel_yoy":
                x, off = x[12:] - x[:12], 12
                sigma = noise_sigma(rel[i][:12])  * np.sqrt(2)  # шум разности двух лет
            else:
                off = 0
                sigma = noise_sigma(rel[i][:12])  # шум по 2023 г. (до периода мониторинга)
            ct = None if cp_true is None else cp_true - off
            for name, settings in grid.items():
                fn = methods[name][0]
                for kw in settings:
                    det, fa = online_first_detection(x, fn, ct, lo - off, bm["tolerance"], bm["max_delay"],
                                                     sigma=sigma, **kw)
                    rows.append({"signal": signal, "method": name, "param": next(iter(kw.values())),
                                 "control": control, "size": None if control else size,
                                 "true_cp": cp_true, "detected_at": None if det is None else det + off,
                                 "false_alarm": fa, "delay": None if det is None else det - ct})
    b = pd.DataFrame(rows)
    b.to_csv(out / "cpd_benchmark_raw.csv", index=False)
    summ = []
    for (signal, name, param), g in b.groupby(["signal", "method", "param"]):
        sh, ct = g[~g.control], g[g.control]
        tp = sh.detected_at.notna().sum()
        fp = ct.false_alarm.sum() + sh.false_alarm.sum()
        row = {"signal": signal, "method": name, "param": param, "recall": tp / len(sh), "mean_delay_months": sh.delay.mean(),
               "false_alarm_rate_controls": ct.false_alarm.mean(), "precision": tp / max(tp + fp, 1)}
        row["F1"] = 2 * row["precision"] * row["recall"] / max(row["precision"] + row["recall"], 1e-9)
        for s in bm["shift_sizes"]:
            row[f"recall_{int(s * 100)}pct"] = sh[sh["size"] == s].detected_at.notna().mean()
        summ.append(row)
    summ = pd.DataFrame(summ).sort_values(["signal", "F1"], ascending=[True, False])
    summ.to_csv(out / "cpd_benchmark.csv", index=False, float_format="%.3f")
    print(summ.to_string(index=False))

    # ---------- 2. реальные данные: офлайн-разметка всех рядов ----------
    # сигнал rel, офлайн по всем 24 месяцам, параметры из конфига (лучшие по бенчмарку)
    real = []
    for i in range(len(rel)):
        x = rel[i]
        sigma = noise_sigma(x)
        for name in ("PELT", "BinSeg"):
            fn, kw = methods[name]
            for c in fn(x, sigma=sigma, **kw):
                shift = x[c: c + 3].mean() - x[max(0, c - 3): c].mean()
                real.append({"method": name, "series": i, "cp_month": M[c], "shift_log": shift})
    r = pd.DataFrame(real)
    meta = panel.meta.reset_index().rename(columns={"index": "series"})
    r = r.merge(meta[["series", "territory_id", "category", "name", "region_name"]], on="series")
    r.to_csv(out / "cpd_real_all.csv", index=False, float_format="%.4f")
    by_month = r.pivot_table(index="cp_month", columns="method", values="series", aggfunc="count").fillna(0)
    by_month.to_csv(out / "cpd_real_by_month.csv")
    print(by_month.to_string())
    top = (r[(r.method == "PELT") & (r.category == "Все категории")]
           .assign(abs_shift=lambda d: d.shift_log.abs()).sort_values("abs_shift", ascending=False).head(25))
    top.to_csv(out / "cpd_real_top_total.csv", index=False, float_format="%.4f")
    print(top[["name", "region_name", "cp_month", "shift_log"]].to_string(index=False))


if __name__ == "__main__":
    main()
