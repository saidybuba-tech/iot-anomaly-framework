# IoT Anomaly-Based Privacy Protection Framework (prototype)

Прототип фреймворка для обеспечения конфиденциальности данных IoT в умных системах на основе обнаружения аномального поведения устройств методами машинного обучения.

Фреймворк анализирует **только метаданные сетевых потоков** (объём, интенсивность, протоколы, энтропия адресов назначения, временные характеристики) без доступа к содержимому трафика и передаёт за пределы доверенного сегмента только события безопасности.

## Архитектура

```
IoT-устройства -> Шлюз (MQTT, захват пакетов) -> Edge-узел (Raspberry Pi 4) -> Облако
                                                 Isolation Forest + AE       XGBoost + AE
```

Ансамблевая оценка: `s(x) = w_IF·s_IF(x) + w_AE·s_AE(x) + w_XGB·s_XGB(x)`

Адаптивная маршрутизация с порогами θ₁ < θ₂:

| Оценка | Решение |
|---|---|
| s < θ₁ | норма, решение на edge, в облако ничего не передаётся |
| θ₁ ≤ s ≤ θ₂ | в облако передаётся только вектор признаков для верификации |
| s > θ₂ | немедленная реакция на edge (изоляция / уведомление / блокировка) |

## Структура

```
config/config.yaml          параметры: данные, модель, порог, вывод, пороги маршрутизации
src/data_ingestion.py       загрузка данных (CSV; pcap/kafka — расширение)
src/preprocessing.py        очистка, бинарные метки, отбор признаков, масштабирование
src/models/                 IsolationForestModel, AutoencoderModel, XGBoostModel (общий интерфейс BaseModel)
src/detector.py             порог τ (99-й перцентиль на нормальных данных валидации) и решение
src/routing.py              нормализация оценок, ансамбль (3), маршрутизация θ₁/θ₂
src/output.py               JSON-события безопасности без сырых данных
run.py                      точка входа (демо на data/sample_flows.csv)
experiments/compare_models.py  протокол сравнения моделей (раздел 3.1)
experiments/routing_demo.py    демонстрация ансамбля и маршрутизации
tests/                      smoke-тесты на синтетических данных
Dockerfile                  образ python:3.10-slim для Raspberry Pi 4 / облака
```

## Быстрый старт

```bash
pip install -r requirements.txt
python run.py                       # демо на data/sample_flows.csv
python -m pytest -q tests           # smoke-тесты
docker build -t iot-anomaly . && docker run --rm iot-anomaly
```

## Протокол эксперимента

```bash
python experiments/compare_models.py --data path/to/ciciot2023.csv --label-col label
```

- стратифицированное разбиение 70 / 15 / 15 (обучение / валидация / тест);
- StandardScaler, обученный на обучающей выборке;
- Isolation Forest и автоэнкодер обучаются только на нормальном трафике (автоэнкодер — с ранней остановкой по валидации);
- XGBoost: SMOTE только на обучающей выборке, 5-кратная перекрёстная проверка, поиск по сетке `max_depth`, `learning_rate`;
- метрики на общей отложенной выборке: Precision, Recall, F1, ROC-AUC, PR-AUC, время инференса на наблюдение.

Набор данных CIC IoT Dataset 2023 не включён в репозиторий из-за объёма: https://www.unb.ca/cic/datasets/iotdataset-2023.html. Метка нормального трафика — `BenignTraffic`, все остальные классы считаются атаками.

## Лицензия

MIT
