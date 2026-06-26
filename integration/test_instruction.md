# Проверка runtime-валидаций и тестов (Querulus)

## Как проверить **runtime‑проверки** (валидации в сервисе)

Runtime‑проверки у вас сидят в коде сервиса и выполняются **при запросе**:

- `preprocessing_and_validate_vector(...)` (B1/B2/B3/A6/C1–C3)
- `validate_exact_columns_order(...)` перед вызовом `mldataworker` (A3/A5)

Проверить, что они работают, можно так (идея):

- **Сделать запрос** в `/api/predict` с заведомо плохим входом и убедиться, что запрос **падает** *до* `prepare_dataset`:
  - для B1: положить `""` в строковое поле → должно упасть на `validate_preprocessing_invariants`
  - для B2: положить строку в нижнем регистре и отключить upper (не получится без кода), но можно проверить B2 на поле, где upper не применяется (если появится) — сейчас upper применяется везде по логике
  - для C1/C2: `EVENT_DATE="not-a-date"` или `PAYMENT_ORDER_DATE_TIME="not-a-date"` → должно упасть на `validate_derived_dates_strict`
  - для A5: нарушить порядок/состав колонок после `_require_columns` практически можно только если меняли списки/обогащение — это больше “защитная сетка”

Факт “упало до mldataworker” вы увидите по тому, что:
- вернулся HTTP 500 (сейчас исключения пробрасываются в `predict_route` как 500),
- и в логах будет сообщение `main_predict: входной вектор не прошёл строгую проверку`.

## Как проверить **unit‑тесты** (файл `examples/querulus/integration/unit_tests.py`)

Unit‑тесты запускаются **отдельно**, тест‑раннером `unittest`, например:

- Запуск только unit‑тестов (этого файла):

```bash
python -m unittest examples.querulus.integration.unit_tests -v
```

## Как проверить **интеграционные тесты** (файл `examples/querulus/integration/integration_tests.py`)

Интеграционные тесты:
- загружают **два** датафрейма: обучающий (`OUTBOXML_TRAIN_DF_PATH`) и сырой (`OUTBOXML_RAW_DF_PATH`),
- берут из сырого только колонки, которые реально нужны сервису/моделям, прогоняют через `preprocessing_and_validate_vector(...)` и сравнивают с обучающим (только эти же колонки),
- прогоняют сырой вход через `main_predict(...)` и сверяют предсказания с `preds_cf` / `preds_rg`,
- прогоняют `INCOMING_VECTOR` через пайплайн предобработки и сравнивают с `OUTGOING_VECTOR`.

Для запуска интеграционных тестов нужна переменная окружения:
- `OUTBOXML_MODEL_GROUP`: stem файла `*.pickle` с группой моделей (обязательная, иначе тесты упадут).

Для теста train vs raw нужны переменные окружения:
- `OUTBOXML_TRAIN_DF_PATH`: путь к обучающему датафрейму (csv/xlsx/pkl/parquet). Если не задан — интеграционный тест упадёт.
- `OUTBOXML_RAW_DF_PATH`: путь к сырому датафрейму (csv/xlsx/pkl/parquet). Если не задан — интеграционный тест упадёт.

Колонки с предсказаниями в обучающем датафрейме задаются так:
- `OUTBOXML_PREDS_CF_COL` (по умолчанию `preds_cf`) — вероятность классификации
- `OUTBOXML_PREDS_RG_COL` (по умолчанию `preds_rg`) — сырая регрессия

Выравнивание строк между train и raw делается по ключу `LOSS_NUMBER` (если есть в обоих), иначе по `INCIDENT_NUMBER` (если есть в обоих). Если ни одного ключа нет — тест упадёт с понятной ошибкой.

```bash
python -m unittest examples.querulus.integration.integration_tests -v
```

## Как запустить **все тесты разом**

- Запуск двух модулей (unit + integration) одним запуском:

```bash
python -m unittest examples.querulus.integration.unit_tests examples.querulus.integration.integration_tests -v
```

- Автопоиск только в `examples/querulus/integration/` (подхватит `unit_tests.py` и `integration_tests.py`):

```bash
python -m unittest discover -s examples/querulus/integration -p "*tests.py" -v
```

Ожидаемое поведение:
- unit‑тесты, которые `assertRaises(ValueError)`, должны проходить (т.е. реально получать исключение)
- интеграционные тесты должны подтверждать, что:
  - датафрейм после предобработки совпадает с эталоном,
  - `classification_proba` и `regression_predictions` совпадают с `preds_cf/preds_rg` (после округления до 2 знаков).

## Важное различие
- **Runtime‑проверки**: срабатывают при реальном запуске сервиса и обработке запроса.
- **Unit‑тесты**: срабатывают только при запуске `python -m unittest ...`.
- **Интеграционные тесты**: дергают `main_predict(...)` и/или предобработку как в сервисе, но без HTTP.

Если хотите, могу подсказать конкретные примеры JSON для `/api/predict`, которые гарантированно триггерят каждую runtime‑проверку (B1, C1, C3 и т.д.).