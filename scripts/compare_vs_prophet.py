"""Бутстреп-интервалы для относительной разницы MAE модели и Prophet (ресэмплинг рядов «МО × категория»).
python scripts/compare_vs_prophet.py -> reports/results/bootstrap_vs_prophet.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports/results"
MODELS = ["combo_gbm_fm_prophet", "combo_gbm_fm", "chronos2_hybrid", "lightgbm", "snaive_growth"]


def main(n_boot: int = 1000, seed: int = 42) -> None:
    p = pd.read_parquet(RES / "predictions.parquet")
    p["ae"] = (p.y_pred - p.y_true).abs()
    rng = np.random.default_rng(seed)
    rows = []
    for h, g in p.groupby("horizon"):
        w = g.pivot_table(index=["series", "target"], columns="model", values="ae")
        per_series = w.groupby(level="series").mean()
        ids = per_series.index.to_numpy()
        for m in MODELS:
            if m not in per_series or per_series[m].isna().all():
                continue
            d = (per_series[m] - per_series["prophet"]).to_numpy()
            rel = per_series[m].to_numpy(), per_series["prophet"].to_numpy()
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(ids), len(ids))
                boots.append(rel[0][idx].mean() / rel[1][idx].mean() - 1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"horizon": h, "model": m, "MAE_model": rel[0].mean(), "MAE_prophet": rel[1].mean(),
                         "delta_pct": (rel[0].mean() / rel[1].mean() - 1) * 100,
                         "ci95_low_pct": lo * 100, "ci95_high_pct": hi * 100, "mean_diff_rub": d.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "bootstrap_vs_prophet.csv", index=False, float_format="%.2f")
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
