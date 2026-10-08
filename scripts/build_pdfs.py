"""Сборка PDF: методологический отчёт (docs/methodology.md) и презентация.
Нужен Google Chrome / Chromium (headless печать в PDF) и пакет markdown.
python scripts/build_pdfs.py -> reports/methodology.pdf, reports/presentation.pdf
"""
from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import markdown
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, OUT = ROOT / "reports/results", ROOT / "reports/figures", ROOT / "reports"

CSS = """
body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:10.5pt;line-height:1.42;color:#1d1d1f;margin:0}
h1{font-size:18pt;color:#0b3d6e} h2{font-size:13.5pt;color:#0b3d6e;border-bottom:1px solid #cfd8e3;padding-bottom:2px;margin-top:18px}
h3{font-size:11.5pt;color:#0b3d6e} table{border-collapse:collapse;margin:8px 0;font-size:9pt}
th,td{border:1px solid #c9d3df;padding:3px 6px;text-align:left} th{background:#eef4fb}
code,pre{font-size:8.5pt;background:#f5f7fa} pre{padding:6px} img{max-width:100%}
.fig{text-align:center;margin:8px 0} .fig div{font-size:8.5pt;color:#555}
"""


def img(name: str) -> str:
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}">'


def chrome() -> str:
    for c in ("google-chrome", "chromium", "chromium-browser"):
        if shutil.which(c):
            return c
    raise SystemExit("Нужен Chrome/Chromium для печати в PDF")


def to_pdf(html: str, pdf: Path, landscape: bool = False) -> None:
    tmp = pdf.with_suffix(".html")
    tmp.write_text(html, encoding="utf-8")
    subprocess.run([chrome(), "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", tmp.as_uri()], check=True, capture_output=True)
    print("PDF:", pdf)


FIGS_REPORT = [
    ("## 3. Протокол", "fig_architecture.png", "Архитектура решения"),
    ("## 2. Данные", "fig_data_overview.png", "Медианные расходы по категориям"),
    ("## 6. Обнаружение", "fig_mae_by_horizon.png", "fig_mae_by_horizon: MAE по горизонтам"),
    ("## 6. Обнаружение", "fig_shap.png", "fig_shap: важность признаков LightGBM (SHAP)"),
    ("## 6. Обнаружение", "fig_map_gain_h3.png", "fig_map_gain_h3: выигрыш итоговой модели относительно Prophet по МО"),
    ("## 6. Обнаружение", "fig_examples_h1.png", "Примеры прогнозов на 1 мес."),
    ("## 7. Новости", "fig_cpd_benchmark.png", "fig_cpd_benchmark: полнота и ложные тревоги при разных параметрах"),
    ("## 7. Новости", "fig_cpd_real_by_month.png", "fig_cpd_real_by_month: точки сдвига на реальных данных по месяцам"),
    ("## 8. Ограничения", "fig_events_flood2024.png", "Паводок апреля 2024 г.: локальная компонента расходов в затронутых МО"),
]


def build_report() -> None:
    md = (ROOT / "docs/methodology.md").read_text(encoding="utf-8")
    # вставляем рисунки перед указанными разделами
    for anchor, f, cap in FIGS_REPORT:
        if (FIG / f).exists() and anchor in md:
            md = md.replace(anchor, f'<div class="fig">{img(f)}<div>{cap}</div></div>\n\n{anchor}', 1)
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = f"<html><head><meta charset='utf-8'><style>@page{{size:A4;margin:16mm 15mm}}{CSS}</style></head><body>{body}</body></html>"
    to_pdf(html, OUT / "methodology.pdf")


def table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, escape=False)


def two_cols(left: str, right: str) -> str:
    return (f"<div style='display:flex;gap:6mm;align-items:flex-start'><div style='flex:1'>{left}</div>"
            f"<div style='flex:1.15'>{right}</div></div>")


def build_slides() -> None:
    m = pd.read_csv(RES / "metrics.csv")
    names = {"prophet": "Prophet (база)", "naive": "Наивная", "snaive_growth": "Сезонно-наивная + рост",
             "lightgbm": "LightGBM", "chronos_bolt": "Chronos-Bolt (zero-shot)", "chronos2_hybrid": "Chronos-2 гибрид",
             "combo_gbm_fm": "LightGBM + Chronos-2", "combo_gbm_fm_prophet": "<b>LightGBM + Chronos-2 + Prophet (итог)</b>",
             "combo_naive_prophet": "Наивная + Prophet"}
    t = m[m.model.isin(names)].pivot_table(index="model", columns="horizon", values="MAE").reindex(list(names))
    t = t.map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}".replace(",", " "))
    t.columns = [f"MAE h={c}" for c in t.columns]
    t.insert(0, "Модель", [names[i] for i in t.index])
    b = pd.read_csv(RES / "bootstrap_vs_prophet.csv")
    b = b[b.model == "combo_gbm_fm_prophet"]
    r2 = m[m.model.isin(["prophet", "combo_gbm_fm_prophet"])].pivot_table(index="horizon", columns="model", values="R2")
    vs = pd.DataFrame({"Горизонт": b.horizon.astype(str) + " мес.",
                       "Prophet": b.MAE_prophet.round(0).astype(int), "Итог": b.MAE_model.round(0).astype(int),
                       "Δ MAE": b.delta_pct.map(lambda v: f"{v:+.1f}%"),
                       "95% ДИ": [f"[{lo:+.1f}; {hi:+.1f}]" for lo, hi in zip(b.ci95_low_pct, b.ci95_high_pct)],
                       "R² Prophet / итог": [f"{r2.loc[h, 'prophet']:.3f} / {r2.loc[h, 'combo_gbm_fm_prophet']:.3f}" for h in b.horizon]})
    c = pd.read_csv(RES / "cpd_benchmark.csv")
    c = c[(c.signal == "rel") & (((c.method.isin(["PELT", "BinSeg"])) & (c.param == 8)) | ((c.method == "CUSUM") & c.param.isin([6, 8])))]
    ct = pd.DataFrame({"Метод": c.method + " (" + c.param.map(lambda v: f"{v:g}") + ")",
                       "Обнаружено": (c.recall * 100).map(lambda v: f"{v:.0f}%"),
                       "Задержка, мес.": c.mean_delay_months.map(lambda v: f"{v:.2f}"),
                       "Ложные тревоги": (c.false_alarm_rate_controls * 100).map(lambda v: f"{v:.0f}%"),
                       "F1": c.F1.map(lambda v: f"{v:.2f}")})
    slides = [
        ("Прогноз потребления в муниципалитетах и раннее обнаружение шоков",
         "<p style='font-size:15pt'>Онлайн-конкурс СберИндекса 2026 · направление «Прогнозирование»</p>"
         "<p>Владимир Матусевичус (solo)</p><p>github.com/oqoxucufawo66-spec/sberindex-forecast-2026</p>"
         "<ul><li>Данные СберИндекса: расходы 2 016 МО × 6 категорий, 01.2023–12.2024</li>"
         "<li>Строгий протокол без заглядывания в будущее, тест — июль–декабрь 2024</li>"
         "<li>Итоговая модель лучше Prophet на всех горизонтах: MAE −48% / −36% / −20% / −25% (1/3/6/12 мес.)</li>"
         "<li>Сравнение PELT / BinSeg / CUSUM, проверка на реальном шоке (паводок 04.2024)</li></ul>"),
        ("Архитектура решения", img("fig_architecture.png") +
         "<p>Ключевая идея: log y = национальный фактор категории (сезонность, инфляция) + локальная компонента МО. "
         "Фундаментальная модель Chronos-2 прогнозирует локальную компоненту, глобальный LightGBM учится сразу на всех МО.</p>"),
        ("Протокол оценки",
         "<ul><li>Прогноз на месяц T строится с точки отсчёта o = T − h; модель видит только данные до o — и в признаках, и при обучении</li>"
         "<li>Тест: T = июль…декабрь 2024 → для h = 12 отсчёт в июле–декабре 2023 (7–12 мес. истории)</li>"
         "<li>11 736 рядов (1 956 МО), 70 416 прогнозов на горизонт; 60 МО — выборка разработки, исключена из оценки</li>"
         "<li>MAE (обязательно), R² по уровню и по изменению, WAPE, доля побед над Prophet, бутстреп-ДИ</li>"
         "<li>LightGBM на h = 12 не строится: при 24 мес. данных нет обучающих пар без утечки</li></ul>"),
        ("Сравнение моделей прогнозирования (MAE, руб. на жителя в месяц)", two_cols(table_html(t), img("fig_mae_by_horizon.png"))),
        ("Итоговая модель против Prophet", table_html(vs) +
         "<p>Итоговая модель точнее Prophet в 77% / 75% / 64% / 61% прогнозов (h = 1/3/6/12) и в 99,8% МО на горизонте 3 мес. "
         "Исключение — «Маркетплейсы» на 12 мес.: быстрый рост, линейный тренд Prophet уместен.</p>"),
        ("Фундаментальные модели", "<ul><li>Chronos-Bolt «в лоб» по исходному ряду: MAE 707 / 1 025 / 1 604 / 1 987 — хуже Prophet: по 7–23 точкам "
         "модель не может выучить годовую сезонность</li><li>Гибрид Chronos-2 на локальной компоненте: 294 / 425 / 649 — лучше Prophet на 54% / 41% / 12% (h = 1/3/6)</li>"
         "<li>В паре с LightGBM — лучший результат на 1–3 мес.: 280 / 401</li><li>На 12 мес. гибрид хуже Prophet (1 461): нет года истории для поправки на рост → выручает комбинация с Prophet</li></ul>"
         + img("fig_examples_h1.png")),
        ("Интерпретация прогнозов", img("fig_shap.png") +
         "<p>На 1–3 мес. главные факторы — месяц цели, сезонный ориентир «как год назад» и общероссийская сезонность категории; на 6 мес. — категория и изменение за 6 мес.</p>"),
        ("Где модель лучше Prophet", img("fig_map_gain_h3.png")),
        ("Обнаружение структурных изменений: сравнение методов", two_cols(table_html(ct) + "<p>Бенчмарк: 1 000 реальных рядов, в половину вставлен сдвиг 10–30% (апрель–октябрь 2024), онлайн-режим. "
         "PELT ≈ BinSeg — лучший F1; CUSUM (порог 6) — самое раннее обнаружение (0,8 мес.) ценой большего числа ложных тревог.</p>", img("fig_cpd_benchmark.png"))),
        ("Реальный шок: паводок апреля 2024", img("fig_events_flood2024.png").replace("<img ", "<img style='max-height:70mm' ") +
         "<ul><li>Звериноголовский МО (Курганская обл.): «Все категории» −10% (0,45-й процентиль среди МО), «Здоровье» −29%; PELT/BinSeg датируют сдвиг апрелем, сигнал — в мае–июне (по категориям)</li>"
         "<li>Гайский ГО: −5,5%, сигнал в июле. Орск: падения нет, «Продовольствие» +4%; гипотеза — выплаты пострадавшим, по этим данным не проверяется</li>"
         "<li>Новости привязаны к МО (territory_id) и месяцу; используются только события до точки отсчёта</li></ul>"),
        ("Ограничения и дальнейшее развитие",
         "<ul><li>24 месяца истории, один тестовый период; пик сдвигов в 04.2023 — вероятно, артефакт начала ряда, причины не приписываем</li>"
         "<li>Новости — пока реестр событий с источниками; дальше: автоматический сбор, привязка к МО по топонимам, новостные признаки и абляция</li>"
         "<li>Дообучение Chronos-2 на рядах СберИндекса, ковариаты (календарь, доступность рынков, связи МО)</li>"
         "<li>Вероятностные прогнозы и пороги тревог по квантилям</li></ul>"
         "<p>Воспроизведение: <code>bash scripts/get_data.sh && python scripts/run_backtest.py && python scripts/build_pdfs.py</code>; конфигурация — configs/default.yaml</p>"),
    ]
    css = CSS + """
    @page{size:297mm 167mm;margin:0} body{font-size:12pt}
    .s{width:297mm;height:167mm;box-sizing:border-box;padding:9mm 13mm;page-break-after:always;overflow:hidden;position:relative}
    .s h1{font-size:20pt;margin:0 0 6px 0} .s img{max-height:92mm;display:block;margin:4px auto}
    .s table{font-size:10pt} .s li{margin:3px 0} .n{position:absolute;bottom:5mm;right:10mm;font-size:9pt;color:#888}
    """
    body = "".join(f"<div class='s'><h1>{h}</h1>{c}<div class='n'>{i + 1}/{len(slides)}</div></div>"
                   for i, (h, c) in enumerate(slides))
    to_pdf(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>",
           OUT / "presentation.pdf")


if __name__ == "__main__":
    build_report()
    build_slides()
