# Отчёты

- `methodology.pdf` — методологический отчёт (сборка из `docs/methodology.md`);
- `presentation.pdf` — презентация решения (11 слайдов);
- `figures/` — графики (`python scripts/make_figures.py`);
- `results/` — таблицы метрик и результатов:
  - `metrics.csv`, `metrics_by_category.csv` — MAE, R², R² изменения, WAPE по моделям и горизонтам;
  - `bootstrap_vs_prophet.csv`, `win_rate_vs_prophet.csv` — сравнение с Prophet (бутстреп-ДИ, доля побед);
  - `shap_h{1,3,6}.csv` — важность признаков LightGBM;
  - `cpd_benchmark.csv` — сравнение PELT / BinSeg / CUSUM на полусинтетическом бенчмарке;
  - `cpd_real_by_month.csv`, `cpd_real_top_total.csv` — точки сдвига на реальных данных;
  - `events_study.csv` — разбор паводка апреля 2024 г.

Не коммитятся (пересоздаются скриптами): `predictions.parquet`, `cpd_real_all.csv`, `cpd_benchmark_raw.csv`.
