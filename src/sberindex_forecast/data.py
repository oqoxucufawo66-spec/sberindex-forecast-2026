"""Загрузка открытых данных конкурса и сборка панели «ряд × месяц».

Источник — архив hackathonlicence.zip со страницы конкурса (данные СберИндекса,
лицензия CC BY-SA 4.0):
* consumption.parquet — оценка средних безналичных потребительских расходов
  жителей МО в месяц, руб. (поля territory_id, date, category, value),
  январь 2023 — декабрь 2024, 6 категорий (включая «Все категории»);
* market_access.parquet — индекс доступности рынков МО (2024);
* справочник МО t_dict_municipal_districts.xlsx (id -> название, регион).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_URLS = {
    "hackathon_archive": "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip",
    "mo_dictionary": "https://www.sberbank.com/common/files/t_dict_municipal.rar",
    "dataset_description": (
        "https://sberindex.ru/ru/research/"
        "data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim"
    ),
}


@dataclass
class Panel:
    """Панель рядов: Y[i, t] — расходы ряда i в месяц months[t]."""

    meta: pd.DataFrame          # territory_id, category, name, region_name, region_code, market_access
    Y: np.ndarray               # (n_series, n_months), float
    months: list[str]           # 'YYYY-MM'

    def idx(self, month: str) -> int:
        return self.months.index(month)

    def subset(self, mask: np.ndarray) -> "Panel":
        return Panel(self.meta[mask].reset_index(drop=True), self.Y[mask], self.months)


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_dictionary(path: str | Path) -> pd.DataFrame:
    """Справочник МО: последняя версия записи на каждый territory_id."""
    d = pd.read_excel(_path(path))
    d = d.sort_values("year_to").drop_duplicates("territory_id", keep="last")
    return d[["territory_id", "municipal_district_name", "region_name", "region_code",
              "municipal_district_center_lat", "municipal_district_center_lon"]].rename(
        columns={"municipal_district_name": "name",
                 "municipal_district_center_lat": "lat",
                 "municipal_district_center_lon": "lon"})


def load_panel(cfg: dict) -> Panel:
    """Собрать панель из consumption.parquet + справочник + индекс доступности рынков."""
    dc = cfg["data"]
    c = pd.read_parquet(_path(dc["consumption"]))
    wide = c.pivot_table(index=["territory_id", "category"], columns="date",
                         values="value", aggfunc="first")
    months = sorted(wide.columns)
    wide = wide[months]
    if dc.get("require_complete", True):
        wide = wide[wide.notna().all(axis=1) & (wide > 0).all(axis=1)]
    meta = wide.index.to_frame(index=False)
    meta = meta.merge(load_dictionary(dc["dictionary"]), on="territory_id", how="left")
    ma = pd.read_parquet(_path(dc["market_access"]))
    meta = meta.merge(ma, on="territory_id", how="left")
    return Panel(meta.reset_index(drop=True), wide.to_numpy(dtype=float), list(months))


def sample_territories(panel: Panel, n: int, seed: int) -> np.ndarray:
    """Маска рядов случайной выборки из n МО (все категории каждого МО)."""
    rng = np.random.default_rng(seed)
    ids = np.sort(panel.meta["territory_id"].unique())
    chosen = set(rng.choice(ids, size=min(n, len(ids)), replace=False).tolist())
    return panel.meta["territory_id"].isin(chosen).to_numpy()
