# Pickle моделей для DataDrift

Train: Hive raw → `train_filter` → `prepare_dataset` (clf+rg), как у сервиса.
Prod: уже закодированный `df.output` / `output_outboxml` из Oracle.

- **PSI/KL/JS** — на закодированной плоскости (train≈prod).
- **`dif_*` / `mode_*`** — человекочитаемые лейблы: code→label по
  `feature.replace` из pickle (`strip` + `casefold`).

## Куда класть

Рядом с этим README, имя = stem из `pipeline/config.py` → `DEFAULT_DRIFT_MODELS`:

```text
monitoring/pipeline/models/
  querulus_ansamble_2026_04_18_v1.pickle
  UU_querulus_2026_11_05_12_44_24.pickle
```

Формат как у сервиса: один файл `{stem}.pickle` = `[clf, rg]`
(тот же артефакт, что в querulus-registry / `integration/results/`).

Datadrift читает pickle **config-only** (без catboost/patsy): нужен
`model_config` для `prepare_dataset` и обратного `replace`.

## Контур (Airflow → GitLab clone)

Pod клонирует `querulus_monitoring.git` в `/tmp`. Положите pickle **в этот
GitLab-репозиторий** в каталог `models/` (корень clone = содержимое `pipeline/`).

Альтернатива: каталог на шаре / PVC и env:

```text
DRIFT_MODELS_DIR=/path/to/pickles
```

Опционально в `DEFAULT_DRIFT_MODELS` у модели ключ `"pickle": "relative/or/abs.pickle"`.

## GitHub monorepo

`*.pickle` в `.gitignore` — в github не коммитятся. Для контура используйте
GitLab `querulus_monitoring` или внешний `DRIFT_MODELS_DIR`.
