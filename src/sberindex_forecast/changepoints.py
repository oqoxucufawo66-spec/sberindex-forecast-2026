"""Обнаружение точек структурных изменений (сдвигов уровня) в рядах расходов МО.

Ряд для анализа — локальная компонента rel = log(y) - медиана log(y) по всем МО той
же категории. Это убирает общую сезонность и общероссийские тренды: остаются
изменения, характерные именно для данного муниципалитета.

Методы:
* PELT и Binary Segmentation (библиотека ruptures, стоимость L2) — точный и
  жадный поиск точек смены среднего со штрафом BIC-типа;
* CUSUM — классический онлайн-метод: накопленная сумма отклонений от текущего
  уровня, тревога при превышении порога.
Все ряды стандартизуются оценкой шума sigma = MAD(разностей) / (0.6745 * sqrt(2)),
поэтому пороги и штрафы сопоставимы между МО.

Для раннего обнаружения все методы запускаются в «онлайн-режиме»: в каждый месяц t
метод видит только данные до t включительно, фиксируется первый месяц, когда он
сообщает о сдвиге рядом с истинной точкой (задержка обнаружения).
"""
from __future__ import annotations

import numpy as np
import ruptures as rpt


def noise_sigma(x: np.ndarray) -> float:
    d = np.diff(np.asarray(x, float))
    s = np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2)
    return float(s) if s > 1e-9 else float(np.std(d) + 1e-9)


def _standardize(x, sigma=None):
    x = np.asarray(x, float)
    return (x - x.mean()) / (sigma or noise_sigma(x))


def detect_pelt(x, penalty: float = 3.0, min_size: int = 3, sigma=None, model: str = "l2") -> list[int]:
    """Индексы начала новых режимов. penalty — множитель при log(n) (BIC-подобный штраф)."""
    z = _standardize(x, sigma)
    if len(z) < 2 * min_size:
        return []
    bk = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(z.reshape(-1, 1)).predict(pen=penalty * np.log(len(z)))
    return [b for b in bk if b < len(z)]


def detect_binseg(x, penalty: float = 3.0, min_size: int = 3, sigma=None, model: str = "l2") -> list[int]:
    z = _standardize(x, sigma)
    if len(z) < 2 * min_size:
        return []
    bk = rpt.Binseg(model=model, min_size=min_size, jump=1).fit(z.reshape(-1, 1)).predict(pen=penalty * np.log(len(z)))
    return [b for b in bk if b < len(z)]


def cusum_alarms(x, threshold: float = 4.0, drift: float = 0.5, warmup: int = 6, sigma=None) -> list[int]:
    """Онлайн двусторонний CUSUM. Опорный уровень — среднее с последнего сброса
    (первые warmup точек — только накопление). Возвращает индексы тревог; оценка
    начала сдвига — месяц, когда накопленная сумма последний раз была нулевой."""
    x = np.asarray(x, float)
    s = sigma or noise_sigma(x[: max(warmup, 3)] if len(x) >= 3 else x)
    alarms, start = [], 0
    while start + warmup < len(x):
        ref = x[start: start + warmup].mean()
        gp = gn = 0.0
        last0p = last0n = start + warmup
        fired = False
        for t in range(start + warmup, len(x)):
            z = (x[t] - ref) / s
            gp, gn = max(0.0, gp + z - drift), max(0.0, gn - z - drift)
            if gp == 0:
                last0p = t + 1
            if gn == 0:
                last0n = t + 1
            if gp > threshold or gn > threshold:
                cp = last0p if gp > threshold else last0n
                alarms.append(min(cp, t))
                start, fired = min(cp, t), True
                break
        if not fired:
            break
    return alarms


def online_first_detection(x, method, true_cp: int | None, first_t: int, tol: int, max_delay: int, **kw):
    """Прогоняем метод на x[:t+1] для t = first_t..; возвращает (месяц обнаружения или None,
    была ли ложная тревога до/вне окна истинной точки)."""
    false_alarm = False
    for t in range(first_t, len(x)):
        cps = method(x[: t + 1], **kw)
        cps = [c for c in cps if c >= first_t - tol]  # интересуют только сдвиги в периоде мониторинга
        if not cps:
            continue
        if true_cp is not None and any(abs(c - true_cp) <= tol for c in cps) and t - true_cp <= max_delay:
            return t, false_alarm
        if true_cp is None or all(abs(c - true_cp) > tol for c in cps):
            false_alarm = True
            if true_cp is None:
                return None, True
    return None, false_alarm
