# Данные

Сырые файлы в репозиторий не коммитятся (см. `.gitignore`): их нужно скачать
вручную с сайта СберИндекса и положить в `data/raw/`.

| Набор данных | Где взять | Куда положить |
|---|---|---|
| Безналичные потребительские расходы на уровне МО (по категориям трат) + справочник МО (id → название) | [sberindex.ru → дашборд «Потребительские безналичные расходы на уровне муниципальных образований»](https://sberindex.ru/ru/dashboards/potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij), кнопка скачивания CSV/Parquet | `data/raw/consumer_spending_mo.csv`, `data/raw/mo_dictionary.csv` |
| Границы и изменения муниципальных образований | [sberindex.ru → исследование](https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities) | `data/raw/mo_borders/` |
| Индекс покупательской мобильности (опционально) | [sberindex.ru → дашборд](https://sberindex.ru/ru/dashboards/indeks-mobilnosti) | `data/raw/mobility_mo.csv` |
| Муниципальная статистика Росстата: население, зарплаты (опционально) | [База данных показателей муниципальных образований](https://rosstat.gov.ru/storage/mediabank/Munst.htm) | `data/raw/rosstat/` |

После скачивания проверьте названия колонок и при необходимости поправьте
сопоставление в `configs/default.yaml` → `data.columns`
(внутренние имена: `territory_id`, `date`, `category`, `value`).

Структура:

```
data/
├── raw/         # исходные файлы, как скачаны
├── interim/     # очищенные и объединённые таблицы
└── processed/   # признаки для моделей
```

Использование данных — по условиям конкурса и лицензии СберИндекса.
