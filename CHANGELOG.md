# Changelog — Querulus

Единый журнал изменений проекта: датасет, обучение, HTTP‑сервис, мониторинг, инфраструктура репозитория.

Формат по [Keep a Changelog](https://keepachangelog.com/). Версии HTTP API сервиса совпадают с `oisuu_responce["version"]` в `integration/main.py` и `GET /api/health`.

---

## [Unreleased]

### Changed

- Структура репозитория по [cookiecutter-data-science](https://github.com/drivendataorg/cookiecutter-data-science): код в `src/querulus/`, конфиги в `configs/`, данные в `data/`, ноутбуки в `notebooks/`.
- Единый `env_template` в корне проекта; `CHANGELOG.md` перенесён из `integration/`.

---

## Integration (API)

### [1.2.0] - 2026-06-09

#### Fixed

- **`oisuu_responce.regression_predictions`** — в поле возвращаются сырые значения регрессии (`model.predict` по регрессионным признакам), а не итог с маской по классу. Раньше туда ошибочно попадал тот же список, что и в `predictions` (нули при отрицательном классе), из‑за чего сырая регрессия в ответе ОИСУУ выглядела как нули.
- **Согласованность `oisuu_responce` и `main_response.result`** — `classification_proba`, `classification_predictions`, `regression_predictions` и итоговые поля (`predictions` / `prediction`) собираются из одних и тех же округлённых значений, чтобы интеграции и мониторинг не расходились.

#### Changed

- **Итоговое предсказание (`predictions` / `result.prediction`)** — логика вынесена в `_masked_predictions()`:
  - по умолчанию: регрессия при `classification_predictions == 1`, иначе `0`;
  - при **`IS_LAWYER == 1`** в входном векторе: всегда значение регрессии, даже если классификация дала `0`.
  - Колонка `IS_LAWYER` не подаётся в модели; читается из общего вектора после предобработки. Если колонки нет — считается `0`, старые запросы без поля не ломаются.
  - На `classification_proba`, `classification_predictions` и `regression_predictions` маска не влияет.
- **`GET /api/health`** — кроме `health: true` возвращает `version`. Опционально `git_commit`, если при запуске задана переменная окружения `SERVICE_GIT_COMMIT` (метка сборки, не вызов git из сервиса).

#### Added

- Поддержка входной колонки **`IS_LAWYER`** (см. правила в `_masked_predictions` выше). Значение попадает в `main_response.df.input` вместе с остальным предобработанным вектором.

---

### [1.1.0]

Предыдущий релиз в проде (до правок из 1.2.0). Контракт и поведение совпадают с `1.0.3`, версия в `oisuu_responce["version"]` — `1.1.0`.

---

### [1.0.3]

#### Added

- **SHAP в `oisuu_responce.CPMComment`** — после каждого вызова классификации и регрессии короткая строка с топ‑вкладами признаков (`shap_clf: ... | shap_reg: ...`). Для CatBoost — нативный `ShapValues`, иначе `TreeExplainer`; при ошибке скоринг не падает.
- **Лог при старте** — `version`, путь к каталогу моделей, флаг существования каталога.

#### Changed

- **`main_response.df.input`** — в ответ уходит предобработанный `df_common` (нормализация, derived‑поля, валидации), а не сырой JSON из ОИСУУ. Нужно для корректного DataDrift: сравнение с обучающей выборкой в том же виде, в каком данные проходят пайплайн до `mldataworker`.
- **Сравнение дат** — `EVENT_DATE` и `PAYMENT_ORDER_DATE_TIME` сравниваются только по дате (время не влияет на валидацию и на расчёт `APPLY_DELAY`).
- **`APPLICANT_FORM`** — маппинг устаревших значений из входа («Скрытый юрист», «Представитель (автоюрист)» и т.д.) в канонические категории модели.

#### Removed

- **`prepare_prepared_dataframe()`** — тонкая обёртка над `preprocessing_and_validate_vector()` без дополнительной логики. Вызовы заменены на прямой `preprocessing_and_validate_vector()` в `main_predict` и в интеграционных тестах.
- **`df_raw_enriched`** — отдельный DataFrame «как пришло из ОИСУУ» (`pd.DataFrame(features_values)`), который создавался параллельно с `df_common`. После перехода на `df_common` в `main_response.df.input` сырой дубликат в ответе и в коде не нужен.
- **`df_raw_input`** — необязательный аргумент `run_classification_model` / `run_regression_model`, куда прокидывался `df_raw_enriched`. Внутри функций не использовался (мёртвый параметр от старой отладки). Удалены и параметр, и все места прокидывания.

---

### [1.0.2]

#### Added

- **Интеграционные тесты** — сверка предобработки `INCOMING_VECTOR` с эталоном `OUTGOING_VECTOR`; roundtrip предсказаний с обучающим датасетом (`preds_cf` / `preds_rg`).
- **Документация в коде** — модульные докстринги, комментарии к секциям в `main.py`, тестах, `integration/test_instruction.md`.

#### Changed

- **Fail-fast валидация входа** до вызова `mldataworker`:
  - B1/B2: `""` → NaN, строки в UPPER;
  - B3: `EVENT_YEAR`, `APPLY_DELAY` из дат;
  - A3/A5/A6: точный состав и порядок колонок под модели, запрет дублей имён;
  - C1–C3: валидность дат и derived‑полей.

---

### [1.0.1]

#### Changed

- **Контракт ответа** — в `oisuu_responce` и `main_response.result`: `classification_proba`, `classification_predictions`, `regression_predictions`; итог в `predictions` (JSON‑строка) и `prediction` (список).
- **Порог классификации** — приоритет: `THRESHOLD` во входном векторе → `config.classification_threshold` → дефолт `0.6`.
- **Бизнес-маска** — итог: класс `0` → `0`, класс `1` → сумма регрессии (до появления `IS_LAWYER` в 1.2.0).

---

### [1.0.0]

#### Added

- HTTP‑сервис на FastAPI: `GET /api/health`, `POST /api/predict` (тело `ServiceRequest` / mldataworker).
- Загрузка pickle‑ансамбля: элемент `[0]` — классификация, `[1]` — регрессия.
- Структура ответа: `oisuu_responce`, `main_response` (`usage_model`, `result`, `df`), `second_response` (пустой dict, резерв контракта).
- Базовая предобработка вектора и скоринг по фиксированным спискам признаков классификации и регрессии.
