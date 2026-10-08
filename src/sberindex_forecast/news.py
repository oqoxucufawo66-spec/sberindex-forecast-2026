"""Согласование новостных данных с рядами СберИндекса (план реализации).

Принципы, которые закладываются в реализацию:
1. Пространственная привязка: новость -> МО через извлечение топонимов
   (NER) и сопоставление с справочником МО СберИндекса (id + название +
   регион); новости федерального уровня -> признак для всех МО.
2. Временная привязка: новость агрегируется в месяц публикации; в признаки
   для прогноза с точкой отсчёта t попадают только новости с датой <= t
   (никакой информации из будущего).
3. Признаки: число новостей по темам (закрытие/открытие предприятий,
   ЧС и погода, транспорт, социальные выплаты), тональность, всплеск
   упоминаний относительно собственной нормы МО.
4. Проверка пользы: абляция — метрики MAE и качество обнаружения шоков
   с новостными признаками и без них.
"""
from __future__ import annotations

import pandas as pd


def aggregate_news_monthly(news: pd.DataFrame) -> pd.DataFrame:
    """Агрегировать размеченные новости в признаки «МО × месяц».

    Ожидаемые колонки news: territory_id, published_at, topic, sentiment.
    Возвращает: territory_id, date, news_count, news_sentiment_mean и
    счётчики по темам (news_topic_<topic>).
    """
    df = news.copy()
    df["date"] = pd.to_datetime(df["published_at"]).dt.to_period("M").dt.to_timestamp()
    base = df.groupby(["territory_id", "date"]).agg(
        news_count=("topic", "size"), news_sentiment_mean=("sentiment", "mean")
    )
    topics = (
        df.pivot_table(index=["territory_id", "date"], columns="topic", values="sentiment",
                       aggfunc="size", fill_value=0)
        .add_prefix("news_topic_")
    )
    return base.join(topics).reset_index()
