"""Фундаментальная модель временных рядов Chronos-Bolt (Amazon), zero-shot.

Модель не обучается на наших данных: на вход подаётся история ряда до точки
отсчёта, на выход — квантили прогноза. Поэтому она применима и там, где
обучаемым моделям не хватает истории (горизонт 12 мес. при 24 месяцах данных).
Зависимости: requirements-fm.txt (torch, chronos-forecasting).
"""
from __future__ import annotations

import numpy as np


class ChronosBolt:
    def __init__(self, model_id: str = "amazon/chronos-bolt-small", device: str = "cpu",
                 batch_size: int = 4096):
        import torch
        from chronos import BaseChronosPipeline

        self.torch = torch
        self.batch_size = batch_size
        self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=device,
                                                        torch_dtype=torch.float32)

    def paths(self, Y: np.ndarray, o: int, h_max: int) -> np.ndarray:
        """Медианный прогноз (n_series, h_max) по истории Y[:, :o+1]."""
        ctx = self.torch.tensor(Y[:, : o + 1], dtype=self.torch.float32)
        out = []
        for i in range(0, len(ctx), self.batch_size):
            q, _ = self.pipe.predict_quantiles(ctx[i: i + self.batch_size], prediction_length=h_max,
                                               quantile_levels=[0.5])
            out.append(q[:, :, 0].cpu().numpy())
        return np.vstack(out)


def national_log_median(Y: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """Медиана log(расходов) по всем МО той же категории в каждом месяце — «общероссийский
    фактор» (сезонность + инфляция + общие тренды). Возвращает матрицу той же формы, что Y.
    В момент t использует только значения месяца t, поэтому утечки будущего нет."""
    import pandas as pd

    return pd.DataFrame(np.log(Y)).groupby(categories).transform("median").to_numpy()


class Chronos2Hybrid:
    """Декомпозиция: log y = национальный фактор категории + локальная компонента МО.

    * локальная компонента rel = log y - national_log_median — гладкий ряд без общей
      сезонности; её прогнозирует Chronos-2 (zero-shot);
    * национальный фактор прогнозируется сезонно-наивно с поправкой на годовой рост
      (nat[T-12] + nat[o] - nat[o-12]); если года истории нет — nat[T-12] или nat[o].
    Прогноз = exp(rel_hat + nat_hat).
    """

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cpu", batch_size: int = 1024):
        import torch
        from chronos import Chronos2Pipeline

        self.torch = torch
        self.batch_size = batch_size
        self.pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=device)

    def paths(self, Y: np.ndarray, categories: np.ndarray, o: int, h_max: int) -> np.ndarray:
        nat = national_log_median(Y[:, : o + 1], categories)
        rel = np.log(Y[:, : o + 1]) - nat
        ctx = self.torch.tensor(rel[:, None, :], dtype=self.torch.float32)
        rel_hat = []
        for i in range(0, len(ctx), self.batch_size):
            q, _ = self.pipe.predict_quantiles(ctx[i: i + self.batch_size], prediction_length=h_max,
                                               quantile_levels=[0.5])
            rel_hat.append(np.stack([x[0, :, 0].cpu().numpy() for x in q]))
        rel_hat = np.vstack(rel_hat)
        out = np.empty_like(rel_hat)
        for h in range(1, h_max + 1):
            t = o + h - 12
            if t >= 0 and o - 12 >= 0:
                nat_h = nat[:, t] + nat[:, o] - nat[:, o - 12]
            elif t >= 0:
                nat_h = nat[:, t]
            else:
                nat_h = nat[:, o]
            out[:, h - 1] = np.exp(rel_hat[:, h - 1] + nat_h)
        return out
