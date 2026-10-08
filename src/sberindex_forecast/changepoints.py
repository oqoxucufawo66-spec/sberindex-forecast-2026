"""Обнаружение точек структурных изменений (шоков) в рядах потребления.

Два режима:
* офлайн (ретроспективно, по всему ряду) — PELT и Binary Segmentation из
  библиотеки ruptures; используются для разметки исторических шоков;
* онлайн (по мере поступления данных) — двусторонний CUSUM по остаткам
  прогноза; именно он отвечает за *раннее* выявление.

Качество сравнивается по точности/полноте с допуском ±tolerance месяцев
и по средней задержке обнаружения (detection delay).
"""
from __future__ import annotations

import numpy as np
import ruptures as rpt


def detect_pelt(signal: np.ndarray, model: str = "rbf", penalty: float = 10.0) -> list[int]:
    """Индексы точек изменения (начало нового режима) методом PELT."""
    signal = np.asarray(signal, dtype=float).reshape(-1, 1)
    bkps = rpt.Pelt(model=model, min_size=2).fit(signal).predict(pen=penalty)
    return [b for b in bkps if b < len(signal)]


def detect_binseg(signal: np.ndarray, model: str = "l2", n_bkps: int = 3) -> list[int]:
    """Индексы точек изменения методом Binary Segmentation (фиксированное число точек)."""
    signal = np.asarray(signal, dtype=float).reshape(-1, 1)
    n_bkps = max(0, min(n_bkps, len(signal) // 3))
    if n_bkps == 0:
        return []
    bkps = rpt.Binseg(model=model, min_size=2).fit(signal).predict(n_bkps=n_bkps)
    return [b for b in bkps if b < len(signal)]


def cusum_alarms(residuals: np.ndarray, threshold: float = 5.0, drift: float = 0.5) -> list[int]:
    """Онлайн двусторонний CUSUM по стандартизованным остаткам.

    Возвращает индексы, в которых сработала тревога; после тревоги
    накопленные суммы обнуляются. Остатки стандартизуются по робастной
    оценке масштаба (MAD), чтобы порог был сопоставим между МО.
    """
    r = np.asarray(residuals, dtype=float)
    med = np.nanmedian(r)
    mad = np.nanmedian(np.abs(r - med)) * 1.4826 or 1.0
    z = (r - med) / mad
    s_pos = s_neg = 0.0
    alarms = []
    for i, x in enumerate(z):
        if np.isnan(x):
            continue
        s_pos = max(0.0, s_pos + x - drift)
        s_neg = max(0.0, s_neg - x - drift)
        if s_pos > threshold or s_neg > threshold:
            alarms.append(i)
            s_pos = s_neg = 0.0
    return alarms


def match_changepoints(true_cps: list[int], pred_cps: list[int], tolerance: int = 2) -> dict[str, float]:
    """Precision / recall / F1 с допуском и средняя задержка обнаружения.

    Предсказанная точка засчитывается, если она не раньше чем за tolerance
    и не позже чем через tolerance месяцев от эталонной; каждая эталонная
    точка сопоставляется не более одного раза.
    """
    used, delays, tp = set(), [], 0
    for t in true_cps:
        cands = [p for p in pred_cps if p not in used and -tolerance <= p - t <= tolerance]
        if cands:
            best = min(cands, key=lambda p: abs(p - t))
            used.add(best)
            delays.append(best - t)
            tp += 1
    precision = tp / len(pred_cps) if pred_cps else 0.0
    recall = tp / len(true_cps) if true_cps else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
    }
