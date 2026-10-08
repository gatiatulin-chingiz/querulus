# Проверка runtime-валидаций и тестов (Querulus)

Перед запуском скопируйте `env_template` в `.env` в корне репозитория и выполните:

```bash
pip install -e .
```

## Как проверить **runtime‑проверки** (валидации в сервисе)

Runtime‑проверки выполняются **при запросе** в `integration/main.py`:

- `preprocessing_and_validate_vector(...)` (B1/B2/B3/A6/C1–C3)
- `validate_exact_columns_order(...)` перед вызовом `mldataworker` (A3/A5)

Запуск сервиса:

```bash
python -m integration
```

## Как проверить **unit‑тесты** (`integration/tests/unit_tests.py`)

```bash
python -m unittest integration.tests.unit_tests -v
```

## Как проверить **shadow 2.0.0** (`integration/tests/test_shadow_v2.py`)

```bash
python -m unittest integration.tests.test_shadow_v2 -v
```

Артефакты для second_ (класть в плоский `integration/results/`; legacy fallback `querulus/2.0.0/`):

- `querulus_ansamble.pickle` — из `example_final` export
- `metadata.json` — `best_threshold`
- `dq_bounds.json` — frozen clip из collect

В запросе ОИСУУ: `main_model` = legacy stem, `second_model` = `querulus_ansamble`, оба вектора признаков.

## Как проверить **интеграционные тесты** (`integration/tests/integration_tests.py`)

Конфиг: `integration/config.py` и корневой `env_template` (скопируйте в `.env`).

Обязательные переменные для интеграционных тестов с моделями:

- `OUTBOXML_MODEL_GROUP` — stem файла `*.pickle` в `integration/results/` (или `prod_models_path`)
- `OUTBOXML_TRAIN_DF_PATH` — путь к обучающему датафрейму

```bash
python -m unittest integration.tests.integration_tests -v
```

## Все тесты разом

```bash
python -m unittest discover -s integration/tests -p "*tests.py" -v
```

## vector_checker — сверка вектора 1С с df_for_service

Пакет: `integration/tests/vector_checker`.

Рабочий цикл:

1. `prepare` — берёт актуальный `df_for_service.parquet`, подставляет все `LOSS_NUMBER` в шаблон `integration/Сутяжность.txt`, пишет запрос для консоли 1С.
2. Вставить `work/Сутяжность_for_1c.txt` в 1С → выгрузить Excel.
3. Положить Excel в `integration/tests/vector_checker/work/excel/`.
4. `compare` — 1:1 сверка фич по `LOSS_NUMBER`.

```bash
# из examples/querulus
python -m integration.tests.vector_checker prepare
# выборка N убытков с фиксированным seed (пишет work/sampled_losses.json)
python -m integration.tests.vector_checker prepare -n 50 --seed 42
python -m integration.tests.vector_checker compare --excel integration/tests/vector_checker/work/excel/export.xlsx

# синтетический smoke (без 1С)
python -m integration.tests.vector_checker demo
python -m integration.tests.vector_checker demo -n 3 --seed 7
```

Отчёты: `integration/tests/vector_checker/work/reports/`.
Шаблон `Сутяжность.txt` не перезаписывается — выходной файл отдельный.

Разбор расхождений в DataFrame (вместо JSON): ноутбук
`integration/tests/vector_checker/vector_checker.ipynb`
(`USE_SYNTHETIC=True` для демо, иначе Excel из `work/excel/`).
