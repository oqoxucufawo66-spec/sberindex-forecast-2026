"""Запуск бэктеста: python scripts/run_backtest.py --config configs/default.yaml"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sberindex_forecast.config import load_config  # noqa: E402
from sberindex_forecast.data import filter_short_series, load_spending  # noqa: E402
from sberindex_forecast.pipeline import backtest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--models", nargs="+", default=None,
                    help="naive seasonal_naive prophet lightgbm (по умолчанию без prophet)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    df = load_spending(dcfg["spending_path"], dcfg["columns"], dcfg["freq"])
    df = filter_short_series(df, dcfg["min_history"])
    print(f"Рядов: {df.groupby(['territory_id', 'category']).ngroups}, "
          f"период: {df['date'].min():%Y-%m} — {df['date'].max():%Y-%m}")

    preds, metrics = backtest(df, cfg, args.models)
    out = Path(cfg["output"]["dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "predictions.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"Результаты: {out}")


if __name__ == "__main__":
    main()
