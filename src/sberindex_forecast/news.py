"""Согласование новостных событий с данными СберИндекса.

Минимальная реализация (в рамках конкурса): реестр событий data/external/events.csv —
вручную собранные и проверяемые новости (дата, МО, описание, ссылка на источник).

Правила согласования:
1. Пространство: событие привязано к territory_id (МО в постоянных границах из справочника
   СберИндекса). Если в реестре указано только название МО, id находится по справочнику.
2. Время: событие относится к месяцу даты публикации; для точки отсчёта o используются
   только события с датой не позже конца месяца o (никакой информации из будущего).
3. Использование: (а) проверка методов обнаружения сдвигов на реальных событиях;
   (б) событийный анализ (event study) — как изменилась локальная компонента расходов
   в затронутых МО относительно всех остальных.

Полноценный поток новостей (автоматический сбор, извлечение топонимов, тематическая
разметка, признаки в модель прогноза) — следующий этап, см. README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_events(path, dictionary: pd.DataFrame) -> pd.DataFrame:
    """Прочитать реестр и заполнить territory_id по названию МО из справочника."""
    ev = pd.read_csv(path)
    name2id = dictionary.drop_duplicates("name").set_index("name")["territory_id"]
    ev["territory_id"] = ev["territory_id"].fillna(ev["mo_name"].map(name2id)).astype("Int64")
    ev["month"] = pd.to_datetime(ev["date"]).dt.strftime("%Y-%m")
    return ev.drop_duplicates(["event_id", "territory_id"])


def event_matrix(events: pd.DataFrame, meta: pd.DataFrame, months: list[str]) -> np.ndarray:
    """E[i, t] = 1, если для МО ряда i к концу месяца t уже было событие из реестра."""
    E = np.zeros((len(meta), len(months)), dtype=np.int8)
    tid = meta["territory_id"].to_numpy()
    for _, e in events.iterrows():
        if e["month"] not in months:
            continue
        t0 = months.index(e["month"])
        E[tid == e["territory_id"], t0:] = 1
    return E
