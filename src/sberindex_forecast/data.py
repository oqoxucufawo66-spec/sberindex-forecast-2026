"""Загрузка и приведение данных к единому «длинному» формату.

Внутренний формат — DataFrame с колонками:
    territory_id : идентификатор муниципального образования (МО)
    category     : категория трат (или "total")
    date         : первый день месяца (datetime64)
    value        : значение показателя

Исходные файлы СберИндекса скачиваются вручную (см. data/README.md),
сопоставление колонок задаётся в configs/default.yaml -> data.columns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Страницы открытых данных, на которые ссылается сайт конкурса.
DATA_SOURCES = {
    "consumer_spending_mo": (
        "https://sberindex.ru/ru/dashboards/"
        "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij"
    ),
    "mo_borders_and_changes": (
        "https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities"
    ),
    "mobility_index": "https://sberindex.ru/ru/dashboards/indeks-mobilnosti",
    "all_dashboards": "https://sberindex.ru/ru/dashboards/",
    "rosstat_municipal_db": "https://rosstat.gov.ru/storage/mediabank/Munst.htm",
}

KEY_COLUMNS = ["territory_id", "category", "date"]


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    # Файлы СберИндекса могут быть с разделителем ";" — определяем автоматически.
    return pd.read_csv(path, sep=None, engine="python")


def load_spending(path: str | Path, columns: dict[str, str], freq: str = "MS") -> pd.DataFrame:
    """Загрузить расходы по МО и привести к внутреннему формату.

    Parameters
    ----------
    path : путь к .csv/.parquet
    columns : отображение внутренних имён на имена колонок исходного файла
        (ключи: territory_id, date, value и, опционально, category)
    freq : частота рядов для выравнивания дат
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла {path}. Скачайте данные по инструкции в data/README.md "
            f"(источник: {DATA_SOURCES['consumer_spending_mo']})."
        )
    raw = _read_any(path)
    rename = {src: dst for dst, src in columns.items() if src in raw.columns}
    missing = {"territory_id", "date", "value"} - set(rename.values())
    if missing:
        raise KeyError(
            f"В файле нет колонок для {sorted(missing)}. "
            f"Доступные колонки: {list(raw.columns)}. Поправьте data.columns в конфиге."
        )
    df = raw.rename(columns=rename)
    if "category" not in df.columns:
        df["category"] = "total"
    df["date"] = pd.to_datetime(df["date"]).dt.to_period(freq[0]).dt.to_timestamp()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = (
        df[KEY_COLUMNS + ["value"]]
        .dropna(subset=["value"])
        .groupby(KEY_COLUMNS, as_index=False)["value"]
        .sum()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    return df


def filter_short_series(df: pd.DataFrame, min_history: int) -> pd.DataFrame:
    """Убрать ряды короче min_history наблюдений."""
    sizes = df.groupby(["territory_id", "category"])["value"].transform("size")
    return df[sizes >= min_history].reset_index(drop=True)
