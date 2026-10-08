#!/usr/bin/env bash
# Скачать открытые данные конкурса в data/raw/.
# Источники — ссылки со страницы конкурса https://sber.ru/sberindex/konkurs_sberindex:
#   * архив данных СберИндекса hackathonlicence.zip (consumption, market_access, connection + описание/лицензия);
#   * справочник муниципальных образований t_dict_municipal.rar.
# Сайты sberbank.com используют сертификат, которому доверяют не все системы, поэтому curl -k.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/dict
if [ ! -f data/raw/consumption.parquet ]; then
  curl -skL --retry 3 -o data/raw/hackathonlicence.zip \
    https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip
  python - <<'PY'
import zipfile
z = zipfile.ZipFile("data/raw/hackathonlicence.zip")
for i in z.infolist():
    name = i.filename.split("/")[-1]
    if name.endswith(".parquet"):
        open(f"data/raw/{name}", "wb").write(z.read(i))
    elif name.endswith(".pdf"):
        open("data/raw/hackathon_licence.pdf", "wb").write(z.read(i))
PY
fi
if [ ! -f data/raw/dict/t_dict_municipal_districts.xlsx ]; then
  curl -skL --retry 3 -o data/raw/t_dict_municipal.rar https://www.sberbank.com/common/files/t_dict_municipal.rar
  # нужен unar (apt install unar / brew install unar) или 7z
  if command -v unar >/dev/null; then unar -q -f -o data/raw/_dict data/raw/t_dict_municipal.rar
  else 7z x -y -odata/raw/_dict data/raw/t_dict_municipal.rar >/dev/null; fi
  find data/raw/_dict -name '*.xlsx' -exec mv {} data/raw/dict/ \;
  rm -rf data/raw/_dict
fi
ls -la data/raw data/raw/dict
