"""Графики для отчёта и презентации -> reports/figures/*.png
python scripts/make_figures.py   (после run_backtest.py, run_changepoints.py, run_events.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sberindex_forecast.changepoints import detect_pelt, noise_sigma  # noqa: E402
from sberindex_forecast.config import load_config  # noqa: E402
from sberindex_forecast.data import load_panel  # noqa: E402
from sberindex_forecast.models.foundation import national_log_median  # noqa: E402

RES = ROOT / "reports/results"
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
LABELS = {
    "naive": "Наивная", "seasonal_naive": "Сезонно-наивная", "snaive_growth": "Сезонно-наивная + рост",
    "prophet": "Prophet (база)", "lightgbm": "LightGBM", "chronos_bolt": "Chronos-Bolt (zero-shot)",
    "chronos2_hybrid": "Chronos-2 гибрид", "combo_gbm_fm": "LightGBM + Chronos-2",
    "combo_gbm_fm_prophet": "LightGBM + Chronos-2 + Prophet", "combo_naive_prophet": "Наивная + Prophet",
}
COLORS = {"prophet": "#d62728", "combo_gbm_fm_prophet": "#1a7f37", "combo_gbm_fm": "#2ca02c",
          "lightgbm": "#1f77b4", "chronos2_hybrid": "#9467bd", "chronos_bolt": "#c5b0d5",
          "snaive_growth": "#ff7f0e", "naive": "#7f7f7f", "seasonal_naive": "#bcbd22",
          "combo_naive_prophet": "#e377c2"}


def fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
    ax.tick_params(axis="x", labelsize=8, rotation=0)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_data(panel):
    df = pd.DataFrame(panel.Y.T, index=pd.to_datetime([m + "-01" for m in panel.months]))
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in sorted(panel.meta.category.unique()):
        cols = np.where(panel.meta.category.to_numpy() == c)[0]
        ax.plot(df.index, df[cols].median(axis=1), marker="o", ms=3, label=c)
    ax.set_title("Медиана по МО: средние безналичные расходы жителя в месяц, руб.")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fmt_dates(ax)
    save(fig, "fig_data_overview.png")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    boxes = [
        (0.0, 0.6, 0.19, 0.34, "Данные СберИндекса\nрасходы МО × 6 категорий\n2023-01…2024-12\n+ справочник МО\n+ индекс доступности\nрынков"),
        (0.01, 0.12, 0.17, 0.3, "Реестр событий\n(новости с источниками)\nМО × месяц"),
        (0.23, 0.62, 0.17, 0.3, "Декомпозиция\nlog y = национальный\nфактор категории +\nлокальная компонента МО"),
        (0.45, 0.70, 0.2, 0.25, "Прогноз 1/3/6/12 мес.\nбазовые · Prophet · LightGBM\n· Chronos-Bolt · Chronos-2"),
        (0.45, 0.36, 0.2, 0.25, "Комбинация моделей\n(равные веса)"),
        (0.45, 0.02, 0.2, 0.27, "Обнаружение сдвигов\nPELT · BinSeg · CUSUM\n(онлайн-режим)"),
        (0.71, 0.62, 0.27, 0.3, "Оценка (строгий протокол)\nMAE (обяз.), R², WAPE\nдоля побед над Prophet"),
        (0.71, 0.12, 0.27, 0.36, "Интерпретация\nSHAP · событийный анализ\nпримеры МО · карты"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#eef4fb", ec="#1f4e79", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)
    arrows = [((0.18, 0.77), (0.23, 0.77)), ((0.40, 0.80), (0.45, 0.82)), ((0.40, 0.70), (0.45, 0.16)),
              ((0.55, 0.70), (0.55, 0.61)), ((0.65, 0.48), (0.71, 0.72)), ((0.65, 0.16), (0.71, 0.30)),
              ((0.18, 0.27), (0.45, 0.12)), ((0.18, 0.27), (0.71, 0.24))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#1f4e79"))
    ax.set_title("Архитектура решения", fontsize=12)
    save(fig, "fig_architecture.png")


def fig_mae(metrics):
    show = ["prophet", "naive", "snaive_growth", "lightgbm", "chronos_bolt", "chronos2_hybrid",
            "combo_gbm_fm", "combo_gbm_fm_prophet"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in show:
        d = metrics[metrics.model == m].sort_values("horizon")
        if d.empty:
            continue
        ax.plot(d.horizon, d.MAE, marker="o", lw=2.6 if m in ("prophet", "combo_gbm_fm_prophet") else 1.4,
                color=COLORS.get(m), label=LABELS.get(m, m))
    ax.set_xticks([1, 3, 6, 12])
    ax.set_xlabel("Горизонт прогноза, мес.")
    ax.set_ylabel("MAE, руб.")
    ax.set_title("Ошибка прогноза по горизонтам (тест: июль–декабрь 2024, 1 956 МО × 6 категорий)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    save(fig, "fig_mae_by_horizon.png")


def fig_shap():
    files = sorted(RES.glob("shap_h*.csv"))
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 4))
    axes = np.atleast_1d(axes)
    names = {"seas_prior": "сезонный ориентир (год назад)", "r1": "изменение за 1 мес.", "r2": "за 2 мес.",
             "r3": "за 3 мес.", "r6": "за 6 мес.", "dev_roll3": "откл. от среднего 3 мес.",
             "yoy": "годовой прирост", "nat_r1": "общероссийская динамика", "nat_seas_prior": "общероссийская сезонность",
             "log_level": "уровень расходов", "market_access": "доступность рынков", "region": "регион",
             "cat": "категория", "target_month": "месяц цели", "h": "горизонт"}
    for ax, f in zip(axes, files):
        t = pd.read_csv(f).head(10)[::-1]
        ax.barh([names.get(x, x) for x in t.feature], t.mean_abs_shap, color="#1f77b4")
        ax.set_title(f"LightGBM, h = {f.stem.split('h')[-1]}: средний |SHAP|", fontsize=9)
        ax.tick_params(labelsize=8)
    save(fig, "fig_shap.png")


def fig_cpd_benchmark():
    f = RES / "cpd_benchmark.csv"
    if not f.exists():
        return
    b = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, sig in zip(axes, ["rel", "rel_yoy"]):
        for m, c in zip(["PELT", "BinSeg", "CUSUM"], ["#1f77b4", "#2ca02c", "#d62728"]):
            d = b[(b.signal == sig) & (b.method == m)].sort_values("false_alarm_rate_controls")
            ax.plot(d.false_alarm_rate_controls, d.recall, marker="o", color=c, label=m)
            for _, r in d.iterrows():
                ax.annotate(f"{r.param:g}", (r.false_alarm_rate_controls, r.recall), fontsize=7,
                            xytext=(3, -8), textcoords="offset points")
        ax.set_title({"rel": "Сигнал: локальная компонента", "rel_yoy": "Сигнал: локальная компонента, г/г"}[sig], fontsize=10)
        ax.set_xlabel("Доля ложных тревог на контрольных рядах")
    axes[0].set_ylabel("Доля обнаруженных сдвигов (≤ 3 мес.)")
    axes[0].legend(frameon=False)
    save(fig, "fig_cpd_benchmark.png")


def fig_events(panel, rel):
    f = RES / "events_study.csv"
    if not f.exists():
        return
    M = panel.months
    x_dates = pd.to_datetime([m + "-01" for m in M])
    picks = [("Звериноголовский муниципальный округ", "Курганская обл."), ("городской округ Гайский", "Оренбургская обл."),
             ("городской округ город Орск", "Оренбургская обл.")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, (name, reg) in zip(axes, picks):
        for c in ["Все категории", "Продовольствие", "Здоровье", "Маркетплейсы"]:
            i = np.where((panel.meta.name == name) & (panel.meta.category == c))[0]
            if len(i) == 0:
                continue
            x = rel[i[0]]
            ax.plot(x_dates, (np.exp(x - x[12:15].mean()) - 1) * 100, marker="o", ms=2.5, label=c,
                    lw=2.2 if c == "Все категории" else 1)
        ax.axvline(pd.Timestamp("2024-04-05"), color="red", ls="--", lw=1)
        ax.text(pd.Timestamp("2024-04-10"), ax.get_ylim()[1] * 0.85, "паводок", color="red", fontsize=8)
        ax.set_title(f"{name.replace('муниципальный округ', 'МО').replace('городской округ', 'ГО')}\n({reg})", fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
        fmt_dates(ax)
    axes[0].set_ylabel("Локальная компонента, % к янв–мар 2024")
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "fig_events_flood2024.png")


def fig_cpd_real(panel, rel):
    f = RES / "cpd_real_by_month.csv"
    if not f.exists():
        return
    d = pd.read_csv(f, index_col=0)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(d.index, d["PELT"], color="#1f77b4")
    ax.set_title("Найденные PELT точки сдвига локальной компоненты по месяцам (все ряды)")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    save(fig, "fig_cpd_real_by_month.png")


def fig_examples(panel):
    f = RES / "predictions.parquet"
    if not f.exists():
        return
    p = pd.read_parquet(f)
    p = p[(p.horizon == 1) & p.model.isin(["prophet", "combo_gbm_fm_prophet", "lightgbm"])]
    big = panel.meta[(panel.meta.category == "Все категории")].copy()
    picks = []
    for nm in ["городской округ город Курган", "городской округ город Орск", "Звериноголовский муниципальный округ"]:
        s = big[big.name == nm]
        if len(s):
            picks.append(s.index[0])
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(4.4 * len(picks), 3.6))
    x_dates = pd.to_datetime([m + "-01" for m in panel.months])
    for ax, i in zip(np.atleast_1d(axes), picks):
        ax.plot(x_dates, panel.Y[i], color="k", marker="o", ms=3, label="факт")
        pi = p[p.series == i]
        for m in ["prophet", "lightgbm", "combo_gbm_fm_prophet"]:
            d = pi[pi.model == m].sort_values("target")
            ax.plot(pd.to_datetime(d.target + "-01"), d.y_pred, marker="s", ms=3, color=COLORS[m], label=LABELS[m])
        ax.set_title(panel.meta.at[i, "name"].replace("городской округ ", "ГО ").replace("муниципальный округ", "МО"), fontsize=9)
        fmt_dates(ax)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    np.atleast_1d(axes)[0].set_ylabel("руб. на жителя в месяц")
    fig.suptitle("Прогноз на 1 месяц вперёд, «Все категории» (тест: июль–декабрь 2024)", fontsize=10)
    save(fig, "fig_examples_h1.png")


def fig_map(panel):
    f = RES / "predictions.parquet"
    if not f.exists() or "lat" not in panel.meta:
        return
    p = pd.read_parquet(f)
    p = p[p.model.isin(["prophet", "combo_gbm_fm_prophet"]) & (p.horizon == 3)]
    p["ae"] = (p.y_pred - p.y_true).abs()
    w = p.pivot_table(index="series", columns="model", values="ae", aggfunc="mean")
    w["gain"] = 1 - w["combo_gbm_fm_prophet"] / w["prophet"]
    meta = panel.meta.loc[w.index]
    g = pd.DataFrame({"tid": meta.territory_id.values, "lat": meta.lat.values, "lon": meta.lon.values,
                      "gain": w.gain.values}).groupby("tid").agg(lat=("lat", "first"), lon=("lon", "first"),
                                                                    gain=("gain", "median")).dropna()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sc = ax.scatter(g.lon, g.lat, c=np.clip(g.gain, -0.6, 0.6) * 100, cmap="RdYlGn", s=7, vmin=-60, vmax=60)
    plt.colorbar(sc, ax=ax, label="снижение MAE относительно Prophet, %")
    ax.set_xlim(19, 180)
    ax.set_title(f"Где итоговая модель лучше Prophet (горизонт 3 мес., медиана по категориям; {len(g)} МО с координатами)")
    ax.set_xlabel("долгота"); ax.set_ylabel("широта")
    save(fig, "fig_map_gain_h3.png")


def main():
    cfg = load_config(ROOT / "configs/default.yaml")
    panel = load_panel(cfg)
    rel = np.log(panel.Y) - national_log_median(panel.Y, panel.meta.category.to_numpy())
    fig_data(panel)
    fig_architecture()
    if (RES / "metrics.csv").exists():
        fig_mae(pd.read_csv(RES / "metrics.csv"))
    fig_shap()
    fig_cpd_benchmark()
    fig_cpd_real(panel, rel)
    fig_events(panel, rel)
    # прогнозы: индексы рядов в predictions.parquet соответствуют панели без выборки разработки
    from sberindex_forecast.data import sample_territories
    dev = cfg["protocol"].get("dev_territories")
    ev_panel = panel.subset(~sample_territories(panel, dev, cfg.get("seed", 42))) if dev else panel
    fig_examples(ev_panel)
    fig_map(ev_panel)


if __name__ == "__main__":
    main()
