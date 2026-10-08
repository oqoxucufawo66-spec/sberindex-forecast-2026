"""Фундаментальные модели временных рядов (zero-shot прогноз).

Сейчас — обёртка над Chronos-Bolt (Amazon). Требует requirements-fm.txt.
План: сравнить zero-shot и дообученный вариант с LightGBM и Prophet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ChronosForecaster:
    def __init__(self, model_id: str = "amazon/chronos-bolt-small", device: str = "cpu"):
        import torch
        from chronos import BaseChronosPipeline

        self._torch = torch
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.float32
        )

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        """Медианный прогноз на horizon шагов."""
        context = self._torch.tensor(history.to_numpy(dtype=np.float32))
        quantiles, _mean = self.pipeline.predict_quantiles(
            context=context, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
        )
        return quantiles[0, :, 1].cpu().numpy()
