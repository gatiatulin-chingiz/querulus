# Тесты integration (Querulus)

## Артефакты (обязательны, без skip)

Единый конфиг путей: `integration/tests/artifacts.py`.

Сгенерировать синтетику:

```bash
python scripts/build_integration_test_artifacts.py
```

Пишет в `integration/results/test_synth/`:
- `querulus_ansamble_legacy.pickle` — main_
- `UU_querulus_2026_10_09_16_51_20.pickle` — second_
- `cf_calibrator.pickle`, `rg_calibrator.pickle`, `metadata.json`
- `df_main.parquet`, `df_second.parquet` (train/test + preds_cf/preds_rg)

Нет файла → тест **падает** (`FileNotFoundError`), не skip.

## Какие тесты за что

| Модуль | Назначение |
|--------|------------|
| `test_common_preprocess` | Общее для main∩second: `""`→NaN, UPPER, даты, IS_LAWYER-маска |
| `test_second_features` | Только second_: CPI, FE_*, τ из meta, калибраторы |
| `test_parity` | CF/RG до 2 знаков vs эталон на train и test; dual `/api/predict` |

## Запуск

Из корня querulus (или monorepo с `PYTHONPATH`):

```bash
python scripts/build_integration_test_artifacts.py
python -m unittest discover -s integration/tests -p "test_*.py" -v
```
