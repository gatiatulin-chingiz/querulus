# Глоссарий таблиц `collect.ipynb`

По названию таблицы / колонки — смысл и как считается. Каталог `FE_*` — в [`../QUERULUS_WIKI.md`](../QUERULUS_WIKI.md) §6.

## C2+. Legacy vs new (`stack_eval`)

Источник: `querulus.training.stack_eval.evaluate_legacy_vs_new`. Holdout по умолчанию после B — `splits.test`.

### Сверка TARGET_2 vs TARGET_FREQ

| Колонка / поле | Смысл |
|----------------|--------|
| (см. `target_compare`) | Совпадение бинарных меток legacy vs new на одних строках holdout |

### Классификация (порог 0.5)

| Строка `stack` | Модель | `y_true` |
|----------------|--------|----------|
| `legacy` | Фичи и CatBoost-hparams блока A (legacy) | `TARGET_2` |
| `legacy feats/hparams @ TARGET_FREQ` | Те же фичи/hparams, **переобучение** на `TARGET_FREQ` (окно train legacy без holdout) | `TARGET_FREQ` |
| `new` | Фичи/hparams блока B | `TARGET_FREQ` |

| Колонка | Смысл |
|---------|--------|
| `pr_auc` | Average precision (PR-AUC) по proba |
| `precision` / `recall` / `f1` | При пороге **0.5** на `pred_freq` |
| `n` | Число строк holdout |
| `threshold` | Всегда 0.5 в этой таблице |

### Покрытие: обе регрессии, классификация new

Обе severity оцениваются при **общей** `pred_freq` модели new и факте `TARGET_FREQ`.  
`amount` — вклад в fin-effect coverage (расход отрицательный), см. `compute_fin_effect_model_coverage`.

| `outcome` | Смысл |
|-----------|--------|
| `0-0` | fact=0, pred=0 |
| `0-1` | fact=0, pred=1 (ложная тревога / ложный иск) |
| `1-0` | fact=1, pred=0 (пропуск) |
| `1-1 хватило` | fact=1, pred=1 и `pred_sev ≥ TARGET_SEV` |
| `1-1 не хватило` | fact=1, pred=1 и `pred_sev < TARGET_SEV` |
| `модель всего` | Сумма по всем строкам |

`stack=legacy` / `new` здесь — какая **severity**-модель дала `pred_sev` (clf одна — new).

### Доля покрытой планки TARGET_SEV (строки 1-1)

Только строки: `TARGET_FREQ=1` и `pred_freq_new=1`.

| Колонка | Смысл |
|---------|--------|
| `n` | Число таких строк |
| `n_covered` | Число строк с `pred_sev ≥ TARGET_SEV` |
| `share_rows_covered` | `n_covered / n` |
| `share_amount_covered` | Σ min(pred_sev, TARGET_SEV) / Σ TARGET_SEV |
| `share_under` | Σ max(TARGET_SEV − pred_sev, 0) / Σ TARGET_SEV |
| `new − legacy` | Построчная **разность** new минус legacy по числовым колонкам |

### Расхождение pred_freq

Сравнение бинарных предсказаний **исходных** clf (legacy на своих фичах vs new), без retrain.

| Показатель | Смысл |
|------------|--------|
| доля несовпадений | Доля строк, где pred разошлись |
| legacy=1, new=0 (n / доли) | Legacy предсказал иск, new — нет |
| legacy=0, new=1 (n / доли) | Наоборот |
| … (доля среди несовпадений) | Условная доля направления среди всех mismatch |

## C3. Fin-effect на holdout Test

Порог подобран на **Val**, отчёт и summary — на **Test** (`fin_effect_b`).

| Поле / колонка summary | Смысл (кратко) |
|------------------------|----------------|
| `fact` / fact_effect | Фактический судебный эффект по формуле FinEffect |
| `model_effect` | Эффект при решениях модели (частота + тяжесть) |
| `net` | Разница model vs fact (см. `print_best_threshold_report`) |

Детали формул — wiki §9 FinEffect.

## График severity: «С / Без выплаты по убытку»

`plot_severity_fact_vs_pred_binned`: бины по fact severity среди фактических исков.

| Панель | Линии |
|--------|--------|
| С выплатой | Медианы таргетов **+** медиана фактических выплат по убытку; отдельно линия выплат |
| Без выплаты | **Sum** чистых таргетов по бину + **sum** фактических выплат по убытку |
