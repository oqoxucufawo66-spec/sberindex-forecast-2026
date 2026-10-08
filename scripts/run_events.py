"""Событийный анализ: паводок апреля 2024 г. (Оренбургская и Курганская области).

Для МО из реестра событий (data/external/events.csv):
* сдвиг локальной компоненты: среднее rel за апрель–июнь 2024 минус январь–март 2024,
  и его ранг среди всех МО той же категории;
* онлайн-обнаружение: первый месяц, когда PELT / BinSeg / CUSUM (параметры из конфига)
  сообщают о сдвиге в 2024 г., и оценка месяца начала сдвига.
python scripts/run_events.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sberindex_forecast.changepoints import cusum_alarms, detect_binseg, detect_pelt, noise_sigma  # noqa: E402
from sberindex_forecast.config import load_config  # noqa: E402
from sberindex_forecast.data import load_dictionary, load_panel  # noqa: E402
from sberindex_forecast.models.foundation import national_log_median  # noqa: E402
from sberindex_forecast.news import load_events  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml")
    cp = cfg["changepoints"]
    panel = load_panel(cfg)
    M, meta = panel.months, panel.meta
    rel = np.log(panel.Y) - national_log_median(panel.Y, meta["category"].to_numpy())
    ev = load_events(ROOT / cfg["data"]["events"], load_dictionary(cfg["data"]["dictionary"]))
    q1 = [M.index(m) for m in ("2024-01", "2024-02", "2024-03")]
    q2 = [M.index(m) for m in ("2024-04", "2024-05", "2024-06")]
    shift = rel[:, q2].mean(1) - rel[:, q1].mean(1)
    meta = meta.assign(shift=shift)
    meta["shift_pct_rank"] = meta.groupby("category")["shift"].rank(pct=True)
    methods = {
        "PELT": (detect_pelt, {"penalty": cp["pelt"]["penalty"], "min_size": cp["pelt"]["min_size"]}),
        "BinSeg": (detect_binseg, {"penalty": cp["binseg"]["penalty"], "min_size": cp["binseg"]["min_size"]}),
        "CUSUM": (cusum_alarms, {"threshold": cp["cusum"]["threshold"], "drift": cp["cusum"]["drift"],
                                 "warmup": cp["cusum"]["warmup"]}),
    }
    first_t = M.index("2024-01")
    rows = []
    for _, e in ev.iterrows():
        for i in np.where(meta["territory_id"].to_numpy() == e["territory_id"])[0]:
            x = rel[i]
            sigma = noise_sigma(x[:12])
            row = {"event_id": e["event_id"], "mo": meta.at[i, "name"], "category": meta.at[i, "category"],
                   "shift_AprJun_vs_JanMar_pct": (np.exp(shift[i]) - 1) * 100,
                   "shift_percentile": meta.at[i, "shift_pct_rank"] * 100}
            for name, (fn, kw) in methods.items():
                found = None
                for t in range(first_t, len(M)):
                    cps = [c for c in fn(x[: t + 1], sigma=sigma, **kw) if c >= first_t]
                    if cps:
                        found = (M[t], M[cps[0]])
                        break
                row[f"{name}_alarm_month"] = found[0] if found else None
                row[f"{name}_cp_month"] = found[1] if found else None
            rows.append(row)
    out = pd.DataFrame(rows)
    path = ROOT / cfg["output"]["dir"] / "results" / "events_study.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.2f")
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
