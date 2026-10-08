"""Первичный разведочный анализ скачанного датасета расходов по МО.

python scripts/eda.py --config configs/default.yaml
Сохраняет сводку и графики в reports/eda/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sberindex_forecast.config import load_config  # noqa: E402
from sberindex_forecast.data import load_spending  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="reports/eda")
    args = ap.parse_args()

    cfg = load_config(args.config)["data"]
    df = load_spending(cfg["spending_path"], cfg["columns"], cfg["freq"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lengths = df.groupby(["territory_id", "category"])["date"].nunique()
    summary = {
        "МО": df["territory_id"].nunique(),
        "категорий": df["category"].nunique(),
        "рядов": len(lengths),
        "период": f"{df['date'].min():%Y-%m} — {df['date'].max():%Y-%m}",
        "медианная длина ряда, мес.": float(lengths.median()),
    }
    (out / "summary.txt").write_text("\n".join(f"{k}: {v}" for k, v in summary.items()), encoding="utf-8")
    print(summary)

    total = df.groupby(["date", "category"])["value"].sum().unstack()
    ax = total.plot(figsize=(11, 5), title="Сумма расходов по категориям, все МО")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(out / "total_by_category.png", dpi=150)
    print(f"Графики: {out}")


if __name__ == "__main__":
    main()
