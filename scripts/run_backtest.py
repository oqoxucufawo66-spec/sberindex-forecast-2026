"""Бэктест моделей прогнозирования.

python scripts/run_backtest.py                       # все модели, все МО
python scripts/run_backtest.py --sample 100 --models naive seasonal_naive lightgbm   # быстрый прогон
Результаты: reports/results/metrics.csv, metrics_by_category.csv, win_rate.csv, shap_h*.csv,
reports/results/predictions.parquet (в git не входит).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sberindex_forecast.config import load_config  # noqa: E402
from sberindex_forecast.data import load_panel, sample_territories  # noqa: E402
from sberindex_forecast.evaluation import summarize, win_rate  # noqa: E402
from sberindex_forecast.pipeline import backtest  # noqa: E402

ALL = ["naive", "seasonal_naive", "snaive_growth", "prophet", "lightgbm", "chronos", "chronos2_hybrid"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="+", default=ALL)
    ap.add_argument("--sample", type=int, default=None, help="число случайных МО (по умолчанию из конфига)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dev", action="store_true", help="прогон только на выборке разработки (60 МО)")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    panel = load_panel(cfg)
    ap_dev = cfg["protocol"].get("dev_territories")
    if args.dev:
        panel = panel.subset(sample_territories(panel, ap_dev, cfg.get("seed", 42)))
    elif ap_dev:
        panel = panel.subset(~sample_territories(panel, ap_dev, cfg.get("seed", 42)))
    n = args.sample or cfg["protocol"].get("sample_territories")
    if n:
        panel = panel.subset(sample_territories(panel, n, cfg.get("seed", 42)))
    print(f"Рядов: {panel.Y.shape[0]} ({panel.meta.territory_id.nunique()} МО × "
          f"{panel.meta.category.nunique()} категорий), {panel.months[0]} — {panel.months[-1]}")

    preds, shap_tables = backtest(panel, cfg, args.models)
    out = Path(args.out or ROOT / cfg["output"]["dir"] / "results")
    out.mkdir(parents=True, exist_ok=True)
    preds = preds.merge(panel.meta[["category"]].reset_index().rename(columns={"index": "series"}), on="series")
    preds.to_parquet(out / "predictions.parquet", index=False)

    metrics = summarize(preds)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    by_cat = pd.concat([summarize(g).assign(category=c) for c, g in preds.groupby("category")])
    by_cat.to_csv(out / "metrics_by_category.csv", index=False, float_format="%.4f")
    if "prophet" in set(preds.model):
        wr = None
        for m in sorted(set(preds.model) - {"prophet"}):
            w = win_rate(preds, m)
            wr = w if wr is None else wr.merge(w, on="horizon", how="outer")
        wr.to_csv(out / "win_rate_vs_prophet.csv", index=False, float_format="%.3f")
    for h, t in shap_tables.items():
        t.to_csv(out / f"shap_h{h}.csv", index=False, float_format="%.5f")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
