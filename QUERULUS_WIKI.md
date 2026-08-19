# Querulus Wiki (единая база знаний)

Этот файл — “карта местности” по проекту `querulus`, чтобы следующий агент мог быстро понять:
что за сущности внутри пайплайна, какие таргеты считаются и по каким формулам, какие признаки попадают в модели,
как устроено обучение `frequency + severity`, как считается `fin_effect` и как это всё используется в HTTP-сервисе.

Если вы заметите неточность (особенно в бизнес-формулировках) — исправьте/уточните, и мы согласуем трактовку в этом wiki.

---

## 1) Назначение проекта

Цикл в `querulus` (в упрощении):

1. Собрать финальный датасет уровня **инцидента**: `df_final_3.parquet`.
2. Обучить:
   - модель **frequency** (бинарная классификация): был ли факт взыскания по выбранному таргету frequency;
   - модель **severity** (регрессия): сумму взыскания/тяжести.
3. По предсказаниям frequency+severity посчитать **финансовый эффект** (`fin_effect`) и выбрать порог классификации под метрику чистого эффекта.
4. Использовать обученные модели в HTTP-сервисе (FastAPI) для продакшн-скоринга.
5. (Опционально) мониторить дрейф фич.

Ключевые “точки входа”:
- сборка датасета: `src/querulus/dataset/pipeline.py` → `run_pipeline()`
- таргеты: `src/querulus/dataset/steps/targets.py` → `build_targets()`
- признаки FE: `src/querulus/features/pipeline.py` → `run_features()`
- обучение: `src/querulus/training/train_loop.py` (блок B) + `src/querulus/training/pipeline.py` (fit-модели)
- fin_effect: `src/querulus/fin_effect/calculator.py` + `src/querulus/fin_effect/resolve.py`
- сервис: `examples/querulus/integration/main.py`
- мониторинг: `examples/querulus/monitoring/monitoring.py`

---

## 2) Сущности и гранулярность данных

### 2.1 Инцидент (уровень строки в датасете)

**1 строка = 1 инцидент (victim/granularity).**

Как именно фиксируется “victim-строка”:
- берётся **victim: min `LOSS_NUMBER`** на уровне `INCIDENT_NUMBER`;
- и берётся расчётная часть из `AMOUNT_REPAIR` (по calc) как **max `LOSS_NUMBER`** (см. описание в `configs/features_catalog.md`).

В результате обучение идёт по одной строке на инцидент:
- таргеты (`TARGET_*`) тоже агрегируются на уровне `INCIDENT_NUMBER`;
- все `FE_*` признаки “приземлены” на инцидент.

### 2.2 Loss / Claim / Pretensions (как это соотносится с таргетами)

- **Loss**: убыток внутри инцидента (`LOSS_NUMBER`).
- **Claim / Incoming claim**: судебные/исковые взыскания, агрегируются на инцидент для `TARGET_FREQ` / `TARGET_SEV` / компонент.
- **Pretensions**: претензии (часть “сумм” и “доплат” в таргеты и FE).

В коде это отражено в `build_targets()`:
- выбор первичного loss: `select_primary_loss_per_incident()` (`dataset/filters.py`)
- расчёт частоты/тяжести через SQL-парсинг claims и pivot по инстанциям: `dataset/steps/targets.py`

---

## 3) Время и “вызревание” таргетов (T0 + H ≤ S)

Опорные термины из `configs/dataset_filters.json` и `src/querulus/dataset/maturity.py`:

- **T0** (as-of для FE и вызревания): `PAYMENT_ORDER_DATE_TIME`
- **H** (горизонт вызревания): `target_maturity.horizon_months` (по умолчанию `24` месяца)
- **S** (момент “снимка”): либо `target_maturity.snapshot_date` (если задан), либо вычисляется как max дат в `target_3_claims.parquet`

Правило включения строк в обучение:
- в датасете оставить строки, где `T0 + H ≤ S`;
- и **нет незакрытого суда**: инциденты, у которых последняя инстанция иска “void” / “не принята”, выкидываются (см. `incidents_with_open_court()`).

Важная практическая подсказка:
- сплит train/test тоже делается по `PAYMENT_ORDER_DATE_TIME` (не по дате убытка).

Код:
- `src/querulus/dataset/maturity.py`:
  - `apply_target_maturity()`
  - `apply_target_maturity_from_paths()` (resume по `target_3_claims.parquet`)

---

## 4) Пайплайн датасета: от raw → `df_final_3.parquet`

Оператор сборки:
`src/querulus/dataset/pipeline.py` → `run_pipeline(...)`

У него есть параметры:
- `use_sql`: если true, некоторые выгрузки/артефакты строятся SQL-ом
- `save_checkpoint`: сохранять промежуточные parquet
- `include_enrich`: legacy enrich шаги (claims/payments/pretensions + merge person). В обучении по умолчанию **не используется** из‑за утечек ПСР (см. docstring в `run_pipeline`).
- `include_fe_features`: добавлять `derived/incident` FE (`FE_*`)
- `include_person_features`: добавлять `FE_PERSON_*` (тяжёлый по памяти шаг)
- `resume_from_targets`: пропустить `victim → targets` и использовать `df_after_targets.parquet`

Порядок действий по умолчанию:
1. `load_victim()` (`dataset/steps/victim.py`)
   - загрузка victim parquet
   - merge `VictimObjectType` из SQL
   - фильтр на `victim_object_type` из `dataset_filters.json`
2. `build_targets()` (`dataset/steps/targets.py`)
   - выбор первичного loss
   - расчёт `TARGET_2`, `TARGET_3_SEV`, `TARGET_FREQ`, `TARGET_SEV`
   - применение вызревания: `apply_target_maturity()` (по `target_3_claims.parquet`)
3. `run_features()` (`features/pipeline.py`)
   - cleanup/merge колонки: `cleanup_merge_columns()`
   - приведение типов: `cast_integer_like_columns()`
   - derived `FE_*`: `add_derived_features()`
   - incident pretensions FE: `add_incident_pretension_features()`
   - person history FE (опционально): `run_person_features()` + дефляция real (`features/inflation.py`)
   - data quality: `apply_dataset_data_quality()` → clip ≥ 0, winsorize на train-period
4. checkpoint → `data/processed/df_final_3.parquet`

Имена ключевых артефактов:
- `target_3_claims.parquet` (нужно для вызревания)
- `df_after_targets.parquet` (resume)
- `df_final_3.parquet` (финальный датасет)
- `target_maturity_report.json` (отчёт вызревания)

---

## 5) Таргеты (не признаки): что означают и как считаются

Полная “бижутерия” по таргетам и FE — в `configs/features_catalog.md`.
Ниже кратко фиксирую логику именно формул и связь с кодом.

### 5.1 `TARGET_2` (бинарный ПСР, legacy)

В `build_targets()`:
```
TARGET_2 = (
    Сумма_выплат_по_претензиям
  + Сумма_взыскано_по_ФУ
  + Суммы_взыскано_по_иску
)
TARGET_2 = 1 if TARGET_2 > 0 else 0
```

То есть `TARGET_2` отвечает на вопрос: “было ли ПСР (суммарно) > 0”.

### 5.2 `TARGET_3_SEV` (сумма взыскания, legacy)

В `build_targets()`:
- берётся pivot `RECOVEREDMAINDEBT/WEAROUT/LOSSCOMMODYVALUE_{1..5}`
- затем выбирается **последнее ненулевое значение** среди этих компонент для инцидента.

Это сумма “тяжести” в legacy трактовке.

### 5.3 `TARGET_FREQ` (новый frequency: было взыскание по искам)

В `targets.py` через `_build_target_freq_by_incident()`:
- для claims: `RecoveredValueWithSD` на **последней принятой инстанции каждого иска**;
- агрегируется на инцидент + добавляются доплаты претензий (без ограничения “только претензий определённого типа” — берутся `*_all`);
- затем:
  - `TARGET_FREQ = 1`, если `TARGET_FREQ_AMOUNT > 0`, иначе `0`

Практически: frequency — классификация “есть факт взыскания по этому горизонту или нет”.

### 5.4 `TARGET_SEV` (новый severity: сумма тяжести)

В `build_targets()`:
- `TARGET_SEV = TARGET_SEV_CLAIMS_AMOUNT + SurchargeValue_cumsum_by_incident_all + UTSSurchargeValue_cumsum_by_incident_all`

То есть severity — это “OD + износ + УТС” (в трактовке Querulus new), рассчитанное по последним инстанциям + доплаты.

---

## 6) Каталог признаков `FE_*` (включая `FE_INCIDENT_*` и `FE_PERSON_*`)

Ниже приведён **полный** текст каталога признаков из:
`configs/features_catalog.md`

Он считается источником истины по:
- гранулярности (T0, вызревание, as-of)
- назначениям `FE_*`
- смыслу имен
- тому, что кладётся в `TO_DROP`

--- (начало каталога) ---

# Каталог признаков датасета querulus

**Гранулярность:** 1 строка = 1 инцидент (victim: min `LOSS_NUMBER`; `AMOUNT_REPAIR` из calc — max `LOSS_NUMBER`).  
**T0 (as-of FE, сплит, вызревание):** `PAYMENT_ORDER_DATE_TIME`. В датасет — `T0 + H ≤ S` и без незакрытого суда; `H` / `S` — `configs/dataset_filters.json` → `target_maturity`.  
**Итоговый артефакт:** `data/processed/df_final_3.parquet`.

Формат: **фича** — описание — как собирается.

> В обучение попадает не всё: AutoMVP отсекает константы / >95% NaN / >99% одного значения; часть колонок в `TO_DROP` (`training/mvp_types.py`).

---

## 1. Таргеты (не признаки)

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `TARGET_2` | Бинарный ПСР (Litigant, legacy) | `build_targets`: 1 если `Сумма_выплат_по_претензиям + Сумма_взыскано_по_ФУ + Суммы_взыскано_по_иску > 0` |
| `TARGET_3_SEV` | Сумма взыскания ОД/УТС/износ (Litigant, legacy) | pivot `RECOVERED*_{1..5}` → последний ненулевой из 15 колонок (без претензий в сумме) |
| `TARGET_FREQ` | Было взыскание по искам (новый) | `RecoveredValueWithSD` на последней инстанции иска + претензии `*_all` → бинарный флаг |
| `TARGET_SEV` | Сумма тяжести (новый) | сумма ОД+износ+УТС на последней инстанции каждого иска + претензии `*_all` |

---

## 2. Derived `FE_*` (этап 1, `features/derived.py`)

### A. Timeline

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_DAYS_LOSS_TO_PAYMENT_ORDER` | Дней от убытка до поручения на выплату | `PAYMENT_ORDER_DATE_TIME − LOSS_DATE_TIME` |
| `FE_DAYS_EVENT_TO_LOSS` | Дней от ДТП до убытка | `LOSS_DATE_TIME − EVENT_DATE` |
| `FE_DAYS_TO_PH_CONTRACT_END` | Дней от ДТП до конца полиса PH | `POLICYHOLDER_CONTRACT_END_DATE − EVENT_DATE` |
| `FE_DAYS_TO_VICTIM_CONTRACT_END` | Дней от ДТП до конца полиса victim | `VICTIM_CONTRACT_END_DATE − EVENT_DATE` |
| `FE_IS_WEEKEND_EVENT` | ДТП в выходные | `EVENT_DAY ∈ {5,6}` или `dayofweek ≥ 5` |
| `FE_SEASON_EVENT` | Сезон ДТП | winter/spring/summer/autumn по `EVENT_MONTH` |
| `FE_HOUR_BUCKET_EVENT` | Час ДТП | night/morning/day/evening по `EVENT_HOUR` |
| `FE_HIGH_APPLY_DELAY` | Долгая подача заявления | `APPLY_DELAY > 30` |

### B. ДТП

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_PARTICIPANTS_BIN` | Число участников | 2 / 3 / 4+ из `PARTICIPANTS_COUNT` |
| `FE_DELAY_AND_NO_NOTIFY` | Задержка без уведомления | `NOT_NOTIFICATION=1` и `APPLY_DELAY > 7` |

### C. ТС потерпевшего

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_VICTIM_AGE_BIN` | Возраст ТС | 0-3 / 3-7 / 7-15 / 15+ из `VICTIM_VEHICLE_AGE` |
| `FE_VICTIM_POWER_PER_TON` | Мощность на тонну | `VICTIM_CAPACITY_ENGINE / VICTIM_MAX_WEIGHT` |
| `FE_VICTIM_HEAVY` | Тяжёлое ТС | `VICTIM_MAX_WEIGHT > 3500` |
| `FE_VICTIM_DOORS_BIN` | Число дверей | 2/3/4/5+ из `VICTIM_NUM_DOORS` |
| `FE_VICTIM_SEATS_BIN` | Число мест | le_4 / 5-7 / 8+ из `VICTIM_NUM_PLACE` |
| `FE_VICTIM_JAPAN_RF` | Японское ТС произведено в РФ | `VICTIM_VEHICLE_IS_JAPAN` и `VICTIM_VEHICLE_MADE_IN_RF` |
| `FE_VICTIM_ENGINE_BUCKET` | Тип двигателя | копия `VICTIM_TYPE_ENGINE` |
| `FE_VICTIM_BODY_BUCKET` | Тип кузова | копия `VICTIM_TYPE_BODY` |

### D. ТС виновника

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_GUILTY_AGE_BIN` | Возраст ТС виновника | бакеты как у victim |
| `FE_GUILTY_POWER_PER_TON` | Мощность/тонна виновника | `(GUILTY_CAPACITY_ENGINE / GUILTY_MAX_WEIGHT) * 10000` |
| `FE_GUILTY_HEAVY` | Тяжёлое ТС виновника | `GUILTY_MAX_WEIGHT > 3500` |
| `FE_GUILTY_ENGINE_BUCKET` | Тип двигателя виновника | `GUILTY_TYPE_ENGINE` |

### E. Victim vs Guilty

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_DIFF_VEHICLE_POWER` | Разница мощности | victim − guilty |
| `FE_RATIO_VEHICLE_POWER` | Отношение мощности | victim / guilty |
| `FE_DIFF_VEHICLE_WEIGHT` | Разница массы | victim − guilty |
| `FE_SAME_VEHICLE_CATEGORY` | Одна категория ТС | `VICTIM_VEHICLE_CATEGORY == GUILTY_VEHICLE_CATEGORY` |
| `FE_SAME_VEHICLE_COUNTRY` | Одна страна ТС | сравнение country-колонок |
| `FE_SAME_VEHICLE_BRAND` | Один бренд | `VICTIM_VEHICLE_BRAND == GUILTY_VEHICLE_BRAND` |
| `FE_SAME_VEHICLE_BODY` | Один кузов | `VICTIM_TYPE_BODY == GUILTY_TYPE_BODY` |
| `FE_SAME_VEHICLE_DRIVE` | Один привод | `VICTIM_TYPE_PRIVOD == GUILTY_TYPE_PRIVOD` |
| `FE_JAPAN_MISMATCH` | Разный флаг Japan | `VICTIM_VEHICLE_IS_JAPAN != GUILTY_VEHICLE_IS_JAPAN` |
| `FE_EV_MISMATCH` | Разный флаг EV | `VIC_IS_EV_REG != GUIL_IS_EV_REG` |
| `FE_SAME_TS_REGION` | Один регион ТС | `VICTIM_TS_REGION == GUILTY_TS_REGION` |
| `FE_SAME_POLICY_ISSUER_GROUP` | Один эмитент | `VICTIM_POLICY_ISSUER_GROUP == GUILTY_POLICY_ISSUER_GROUP` |

### F. Гео

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_SAME_REGION_EVENT` | Регион убытка = регион ДТП | `REGION == REGION_EVENT` |
| `FE_REGION_CORRECTED` | Регион скорректирован | `REGION_CORRECTED` заполнен и ≠ `REGION` |
| `FE_SAME_ACCEPTED_LOSS_UNIT` | Принявшее = урегулирующее подразделение | `ACCEPTED_UNIT == LOSS_UNIT` |

### G. Полис

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_KBM_BIN` | Бакет КБМ | le_1 / 1_1.17 / gt_1.17 из `RSAPolicyKBM` |
| `FE_COMMERCIAL_USE` | Коммерческое использование | `USED_AS_TAXI` или `USED_AS_CARSH` |
| `FE_HAS_FRANCHISE` | Есть франшиза | `FRANCHISE_VALUE > 0` |
| `FE_PREMIUM_SUM_ALL_REAL_2020` | Сумма премий в руб. 2020 | `PREMIUM_SUM_ALL / CPI` по T0 |
| `FE_PREMIUM_PER_POLICY` | Премия на полис | `FE_PREMIUM_SUM_ALL_REAL_2020 / PREMIUM_COUNT_ALL` |
| `FE_INSURANCE_AMOUNT_BIN` | Бакет страховой суммы | `<400k / 400k-1M / >1M` из `INSURANCE_AMOUNT` |

> Номинал `PREMIUM_SUM_ALL` → `TO_DROP` (как `VALUE_BEFORE_*`).

### H. Возмещение / минимизация

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_REFUND_FORM_MATCH` | Форма ВФ совпадает | `REFUND_FORM_DETAILED == REFUND_FORM_BY_PAYMENT_ORDER` |
| `FE_REFUND_FORM_MISMATCH` | Расхождение форм ВФ | `REFUND_FORM != REFUND_FORM_DETAILED` |
| `FE_REFUND_IS_CASH` | Денежная форма | `REFUND_FORM_DETAILED` содержит «Денежн» |
| `FE_REFUND_IS_REPAIR` | Ремонтная форма | `REFUND_FORM_DETAILED` содержит «Ремонт» |
| `FE_MINIMIZATION_GAP` | Разрыв минимизации | `MINIMIZATION_REC − MINIMIZATION_FACT` |
| `FE_HAS_MINIMIZATION` | Была минимизация | `MINIMIZATION_KIND` not null |

> `FE_REFUND_*` в `TO_DROP` при обучении (post-T0 leakage).

### I. Калькуляция (`VALUE_BEFORE_WITH` / `VALUE_BEFORE_WITHOUT`)

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_SHARE_WORK_TIER` | Доля работ | low/mid/high из `SHARE_WORK` |
| `FE_VALUE_BEFORE_WITHOUT_BIN` | Бакет без износа (руб. 2020) | пороги 100k/300k на `*_REAL_2020` |
| `FE_HIGH_VALUE_BEFORE_WITHOUT` | Дорогая калькуляция без износа | `*_REAL_2020 > 300000` |
| `FE_VALUE_BEFORE_WITH_BIN` | Бакет с износом (руб. 2020) | те же пороги на `*_REAL_2020` |
| `FE_HIGH_VALUE_BEFORE_WITH` | Дорогая калькуляция с износом | `*_REAL_2020 > 300000` |
| `FE_VALUE_BEFORE_WITH_REAL_2020` | VALUE_BEFORE_WITH в руб. 2020 | `nominal / CPI` (Росстат Dec/Dec → дек.2020=1) |
| `FE_VALUE_BEFORE_WITHOUT_REAL_2020` | VALUE_BEFORE_WITHOUT в руб. 2020 | то же |
| `FE_VALUE_BEFORE_DIFF` | Износ в руб. 2020 | `WITHOUT_REAL − WITH_REAL` |
| `FE_VALUE_BEFORE_RATIO` | Отношение with/without | безразмерное (CPI сокращается) |

> В обучение не идут: `AMOUNT_REPAIR`, `REPAIR_VALUE`, `SHARE_WEAROUT`, `LATITUDE`/`LONGITUDE` и старые FE от них (`TO_DROP` в `mvp_types.py`).

### J. История убытков (past only, из victim)

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_VICTIM_LOSS_COUNT_BIN` | Число прошлых убытков victim | 0/1/2/3+ из `VICTIM_LOSS_COUNT` |
| `FE_VICTIM_REPEAT` | Повторный клиент victim | `VICTIM_LOSS_COUNT > 0` |
| `FE_VICTIM_LOSS_SUM_BIN` | Сумма прошлых убытков | бакеты 50k/200k из `VICTIM_LOSS_SUM` |
| `FE_GUILTY_LOSS_COUNT_BIN` | Число прошлых убытков виновника | из `GUILTY_LOSS_COUNT` |
| `FE_GUILTY_REPEAT` | Повторный виновник | `GUILTY_LOSS_COUNT > 0` |

### K. Сигналы ПСР / ДТП (`_add_frequency_risk_features`)

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_DTPOSAGO_TYPE` | Тип ДТП ОСАГО | копия `DTPOSAGOType` / `DTPOSAGO_TYPE` |
| `FE_EVENT_SCHEME` | Схема ДТП | копия `EventSchemeDescription` |
| `FE_REGRESS_FLAG` | Признак регресса | флаг из `Regress` / `REGRESS*` |
| `FE_JOINT_LIABILITY` | Солидарная ответственность | флаг из `JointLiability` |

### Полный список fixed `FE_*` из `derived.py` (67)

`FE_DAYS_LOSS_TO_PAYMENT_ORDER`, `FE_DAYS_EVENT_TO_LOSS`, `FE_DAYS_TO_PH_CONTRACT_END`, `FE_DAYS_TO_VICTIM_CONTRACT_END`, `FE_IS_WEEKEND_EVENT`, `FE_SEASON_EVENT`, `FE_HOUR_BUCKET_EVENT`, `FE_HIGH_APPLY_DELAY`, `FE_PARTICIPANTS_BIN`, `FE_DELAY_AND_NO_NOTIFY`, `FE_VICTIM_AGE_BIN`, `FE_VICTIM_POWER_PER_TON`, `FE_VICTIM_HEAVY`, `FE_VICTIM_DOORS_BIN`, `FE_VICTIM_SEATS_BIN`, `FE_VICTIM_JAPAN_RF`, `FE_VICTIM_ENGINE_BUCKET`, `FE_VICTIM_BODY_BUCKET`, `FE_GUILTY_AGE_BIN`, `FE_GUILTY_POWER_PER_TON`, `FE_GUILTY_HEAVY`, `FE_GUILTY_ENGINE_BUCKET`, `FE_DIFF_VEHICLE_POWER`, `FE_RATIO_VEHICLE_POWER`, `FE_DIFF_VEHICLE_WEIGHT`, `FE_SAME_VEHICLE_CATEGORY`, `FE_SAME_VEHICLE_COUNTRY`, `FE_SAME_VEHICLE_BRAND`, `FE_SAME_VEHICLE_BODY`, `FE_SAME_VEHICLE_DRIVE`, `FE_JAPAN_MISMATCH`, `FE_EV_MISMATCH`, `FE_SAME_TS_REGION`, `FE_SAME_POLICY_ISSUER_GROUP`, `FE_SAME_REGION_EVENT`, `FE_REGION_CORRECTED`, `FE_SAME_ACCEPTED_LOSS_UNIT`, `FE_KBM_BIN`, `FE_COMMERCIAL_USE`, `FE_HAS_FRANCHISE`, `FE_PREMIUM_PER_POLICY`, `FE_PREMIUM_SUM_ALL_REAL_2020`, `FE_INSURANCE_AMOUNT_BIN`, `FE_REFUND_FORM_MATCH`, `FE_REFUND_FORM_MISMATCH`, `FE_REFUND_IS_CASH`, `FE_REFUND_IS_REPAIR`, `FE_MINIMIZATION_GAP`, `FE_HAS_MINIMIZATION`, `FE_SHARE_WORK_TIER`, `FE_VALUE_BEFORE_WITHOUT_BIN`, `FE_HIGH_VALUE_BEFORE_WITHOUT`, `FE_VALUE_BEFORE_WITH_BIN`, `FE_HIGH_VALUE_BEFORE_WITH`, `FE_VALUE_BEFORE_WITH_REAL_2020`, `FE_VALUE_BEFORE_WITHOUT_REAL_2020`, `FE_VALUE_BEFORE_DIFF`, `FE_VALUE_BEFORE_RATIO`, `FE_DTPOSAGO_TYPE`, `FE_EVENT_SCHEME`, `FE_REGRESS_FLAG`, `FE_JOINT_LIABILITY`, `FE_VICTIM_LOSS_COUNT_BIN`, `FE_VICTIM_REPEAT`, `FE_VICTIM_LOSS_SUM_BIN`, `FE_GUILTY_LOSS_COUNT_BIN`, `FE_GUILTY_REPEAT`.

---

## 2b. Incident pretensions `FE_INCIDENT_*` (`features/incident_pretensions.py`)

Агрегаты претензий **текущего** инцидента с `PRETENSION_GET_DATE ≤ T0` (без утечки будущих).

| Фича / шаблон | Описание | Как собирается |
|---------------|----------|----------------|
| `FE_INCIDENT_PRET_COUNT` | Число претензий до T0 | count строк pretensions |
| `FE_INCIDENT_DECLARED_{FIELD}_SUM` | Сумма по Declared_* | sum каждой колонки `DECLARED_*` |
| `FE_INCIDENT_PRETENSION_VALUE_SUM` | Сумма претензии | sum `PRETENSION_VALUE` (если есть) |
| `FE_INCIDENT_UTSVALUE_SUM` | Сумма УТС | sum `UTSVALUE` (если есть) |

Набор `DECLARED_*` зависит от сырых колонок претензий.

---

## 3. Person `FE_PERSON_*` (`features/person/`)

**Роли:** `APPLICANT`, `VICTIM_PH`, `VICTIM`, `GUILTY`, `DRIVER`, `PAYMENT_RECIPIENT`, `VICTIM_OWNER`, `GUILTY_OWNER`, `POLICYHOLDER`, `POLICYHOLDER_OWNER`.

**Фильтр истории:** `event_date < T0` и `INCIDENT_NUMBER ≠ текущий`.

### 3.1 Static (`static.py`)

| Шаблон фичи | Описание | Как собирается |
|-------------|----------|----------------|
| `FE_PERSON_STATIC_EQ_{ROLE_A}_{ROLE_B}` | Один person_id в двух ролях | 1 если `{ROLE_A}_PERSON_ID == {ROLE_B}_PERSON_ID` на строке victim (45 пар ролей) |
| `FE_PERSON_STATIC_DIFF_{R1}_AGE_{R2}_AGE` | Разница возрастов | `APPLICANT/VICTIM/GUILTY/DRIVER/PAYMENT_RECIPIENT_AGE` попарно (10 пар) |

### 3.2 Pretensions history (`history_pretensions.py`)

Источник: `oisuu81_t_Pretensions` + `IncidentToLoss`. Join-ключи по роли:

| Роль | Join по pretensions |
|------|---------------------|
| `APPLICANT` | `ApplicantPersonID` |
| `PAYMENT_RECIPIENT` | `RecipientPersonID` |
| остальные | оба поля |

| Шаблон фичи | Описание | Как собирается |
|-------------|----------|----------------|
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_COUNT` | Число претензий в истории | count после фильтра as-of T0 |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_PRETENSION_NUMBER_NUNIQUE` | Уникальные номера претензий | nunique `PRETENSION_NUMBER` |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_TYPES_NUNIQUE` | Уникальные типы | nunique `PRETENSION_TYPES` |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_GET_METHOD_MODE` | Мода способа подачи | mode `PRETENSION_GET_METHOD` |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_ANSWER_TYPE_MODE` | Мода типа ответа | mode `ANSWER_TYPE` |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_{MONEY}_SUM` | Сумма по статье | sum: `PRETENSION_VALUE`, `SURCHARGE_VALUE`, `UTS_SURCHARGE_VALUE`, `PRETENSION_VALUE_PENALTY`, `SURCHARGE_VALUE_PENALTY` |

> Имена с двойным префиксом — так собирает `add_prefix(FE_PERSON_PRET_{ROLE}_)` поверх колонок `FE_PERSON_PRET_*`.

> ИПЦ: для applicant/payment_recipient сумм `PRETENSION_VALUE` / `SURCHARGE_VALUE` пишутся `*_REAL_2020` (дефляция по T0); номинал в `TO_DROP`.

### 3.3 Court history (`history_court.py`)

Источник: `oisuu81_t_IncomingClaimNewLogicByInst` + `oisuu81_t_Истцы` (join по номеру иска, person = `Лицо`).

| Шаблон фичи | Описание | Как собирается |
|-------------|----------|----------------|
| `FE_PERSON_COURT_{ROLE}_FE_PERSON_COURT_CLAIM_COUNT_ROWS` | Строк иска в истории | count |
| `FE_PERSON_COURT_{ROLE}_FE_PERSON_COURT_INCOMING_CLAIM_NUMBER_NUNIQUE` | Уникальные иски | nunique |
| `FE_PERSON_COURT_{ROLE}_FE_PERSON_COURT_ПРЕДСТАВИТЕЛЬ_MAX/SUM` | Флаг представителя | max/sum |
| `FE_PERSON_COURT_{ROLE}_FE_PERSON_COURT_ЦЕССИОНАРИЙ_MAX/SUM` | Флаг цессионария | max/sum |
| `FE_PERSON_COURT_{ROLE}_FE_PERSON_COURT_{CLAIMED\|RECOVERED}{FIELD}_SUM` | Сумма заявлено/взыскано | sum по колонкам CLAIMED*/RECOVERED* |
| `FE_PERSON_COURT_{ROLE}_FE_PERSON_COURT_{CLAIMED\|RECOVERED}{FIELD}_MEAN` | Среднее заявлено/взыскано | mean |
| `FE_PERSON_COURT_{ROLE}_FE_PERSON_COURT_CLAIMITEM_MODE` | Предмет иска | mode |
| `FE_PERSON_COURT_{ROLE}_FE_PERSON_COURT_CLAIMORIGIN_MODE` | Происхождение иска | mode |

Поля money: `MainDebt`, `PlaintiffExamination`, `CourtExamination`, `RepresentativeExpenses`, `PenaltyFee`, `Fine`, `MoralDamage`, `StateDuty`, `LossCommodyValue`, `Wearout`, `ValueWithSD`, `ValueWithoutSD`, `AmountLoss`, и др. (все `CLAIMED*` / `RECOVERED*` из incoming claim).

> **Не путать** с `RECOVEREDMAINDEBT_1..5` из `build_targets` — это текущий инцидент для `TARGET_SEV`, в обучение **не идут** (`TO_DROP`).

---

## 4. Колонки из `build_targets` (в датасете, часть только для таргетов)

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `SHARE_WORK` | Доля работ в калькуляции | `Работы / СуммаРемонта` из `_InfoRg14746` |
| `AMOUNT_REPAIR` | Сумма ремонта без износа | `_InfoRg14746` → **TO_DROP** |
| `SHARE_WEAROUT` | Процент износа (cap 50) | `_InfoRg14746` → **TO_DROP** |
| `FLAG_APPLICANT_SAME_VICTIM_PH` | Заявитель = PH victim | `APPLICANT_ID == VICTIM_POLICYHOLDER_PERSON_ID` |
| `RECOVERED*_{1..5}` | Взыскания по инстанциям иска | pivot `target_3_claims` → **TO_DROP** |
| `SurchargeValue_cumsum_by_incident` | Доплаты по претензиям (тип) | SQL aggregate pretensions → **TO_DROP** |
| `UTSSurchargeValue_cumsum_by_incident` | УТС-доплаты | SQL aggregate → **TO_DROP** |
| `Сумма_утс`, `Сумма_выплат_по_претензиям`, … | Агрегаты ПСР | `oisuu81_t_ПСР` → **TO_DROP** |

---

## 5. Базовый victim (`df_Victim_final_11.parquet`, `oisuu81_t_Losses`)

Колонки приходят из victim-parquet; ниже — группы по `dataset/sql/querys.sql`. В обучение попадают после AutoMVP (если не в `TO_DROP`).

### 5.1 Идентификаторы и ключи (в обучение не идут)

`LossID`, `LossNumber`, `IncidentNumber`, `ContractNumber`, `PolicyNumber`, `*PersonID`, `*VIN`, `*RegNum`, `*ObjectID` — ключи join; `INCIDENT_NUMBER`, `LOSS_NUMBER` в `other_cols`.

### 5.2 Событие / ДТП

| Колонки | Описание | Источник |
|---------|----------|----------|
| `EventDate`, `EventYear`, `EVENT_*` | Дата/время/описание ДТП | Losses |
| `EventCreatedByGIBDDFlag` | Оформлено ГИБДД | Losses |
| `DTPOSAGOType`, `EventTypeDescription`, `EventSchemeDescription` | Тип/схема ДТП | Losses |
| `EventLocationRegionName`, `LONGITUDE`, `LATITUDE` | Гео ДТП | Losses; `LONGITUDE`/`LATITUDE` → **TO_DROP** |
| `ParticipantsCount` | Участники | Losses → `FE_PARTICIPANTS_BIN` |

### 5.3 Убыток / процесс

| Колонки | Описание | Источник |
|---------|----------|----------|
| `LossDateTime`, `IssueDate`, `PaymentOrderDateTime` | Даты убытка/выпуска/выплаты | Losses |
| `LossStage`, `LossProcess`, `LossStateByIA` | Стадия/процесс | Losses |
| `LossUnit`, `LossUnitDivision`, `LossUnitZone` | Подразделение | Losses |
| `Filial`, `CustomerImportance` | Филиал / важность | Losses |
| `ApplyDelay`, `RecieveMethod`, `NotNotification` | Подача заявления | Losses |

### 5.4 Лица (сырые атрибуты)

| Префикс | Колонки | Описание |
|---------|---------|----------|
| `APPLICANT_*` | age, sex, type, form | Заявитель |
| `VICTIM_*` / `VICTIM_PH_*` | person, age, type | Потерпевший / PH |
| `GUILTY_*`, `DRIVER_*` | person, age | Виновник / водитель |
| `PAYMENT_RECIPIENT_*` | person, birth_date | Получатель выплаты |
| `POLICYHOLDER_*` | person, vehicle | Страхователь |

### 5.5 ТС victim / guilty / policyholder

Бренд, модель, категория, страна, возраст, мощность, масса, Japan/RF, EV, тип кузова/двигателя/привода — колонки `VICTIM_VEHICLE_*`, `GUILTY_*`, `POLICYHOLDER_VEHICLE_*` из Losses. Часть дублируется в `FE_*` блоках C–E.

### 5.6 Полис / премия / PVU / погода

`InsuredSum`, `InsuranceType*`, `Franchise*`, `RefundForm*`, `PREMIUM_*`, `PVU_*`, `SEASON_*`, `total_loss_*` — внешние скоринги и климат; многие PVU/season в `TO_DROP`.

### 5.7 Минимизация / стоимости

`MINIMIZATION_*`, `VALUE_BEFORE_*`, `VALUE_AFTER_*`, `REPAIR_VALUE`, `CPM_*`, `AMOUNT_REPAIR`, `RSA_RE_OUT`, `FL_PHOTO_VIDEO` — калькуляция и проверки.

### 5.8 История убытков (из victim)

`VICTIM_LOSS_COUNT`, `VICTIM_LOSS_SUM`, `GUILTY_LOSS_COUNT`, `*_FUTURE` — прошлые/будущие агрегаты; `*_FUTURE` в `TO_DROP`.

### 5.9 Флаги

`Regress*`, `JointLiability`, `IsTOTAL`, `flPhotoVideo`, `flGrandLoss`, `FL5Percent`, `UsedAsTaxi`, `UsedAsCarsh`, `isRetail` — бинарные/категориальные из Losses.

---

## 6. Служебные / исключённые из обучения

Полный список `TO_DROP`: `training/mvp_types.py` (`DEFAULT_MVP_INPUT_TYPES['TO_DROP']`).

Ключевые группы:
- итоги ПСР и взысканий текущего инцидента;
- `RECOVERED*_{1..5}` (компоненты `TARGET_SEV`);
- post-T0 refund/payment (`REFUND_FORM_BY_PAYMENT_ORDER`, `FACT_AMOUNT_*`, `FE_REFUND_*`);
- PVU/season future-колонки;
- `*_PERSON_ID`, FIO-колонки (рекомендуется держать в denylist).

---

## 7. Оценка объёма

| Блок | Порядок величины |
|------|------------------|
| Victim raw | ~230 колонок |
| `FE_*` derived (`derived.py`) | **67** fixed |
| `FE_INCIDENT_*` | ~3 + N×`DECLARED_*` |
| `FE_PERSON_STATIC` | ~55 (пары ролей + ages) |
| `FE_PERSON_PRET_{ROLE}_*` | ~9 × 10 ролей ≈ 90 |
| `FE_PERSON_COURT_{ROLE}_*` | ~80+ × 10 ролей ≈ 800+ |
| **Итого в df_final_3** | **~1100+ колонок** |
| **В модели (после AutoMVP)** | ~200–400 (зависит от наполненности) |

---

*Файл: `configs/features_catalog.md`. Код-источник: `features/derived.py`, `features/incident_pretensions.py`, `features/person/*`, `dataset/steps/*`, `training/mvp_types.py`.*

--- (конец каталога) ---

---

## 7) Обучение: как строятся пулы и как отбираются фичи

### 7.1 AutoMVP / `mvp_types.py`

Идея: на уровне табличного `df_final_3` модель CatBoost требует корректной типизации категориальных/числовых признаков.

В `training/pipeline.py` (функция `_apply_mvp_types`):
- запускается `querulus.AutoMVP.MVP(df, cutoff_nan=...)`;
- отдельно определяются `BINARY`, `CATEGORIAL`, `NUMERIC` и `TO_DROP`;
- затем `correct_types()` подправляет типы под желаемые входные пулы.

Что важно:
- AutoMVP режет:
  - константы,
  - > `mvp_cutoff_nan` пропусков,
  - и “почти один уникальный” режим;
- в обучение **не идут** колонки из `TO_DROP` (`training/mvp_types.py`).

Практический след:
агенту при вопросах “почему колонка в пулах исчезла” надо смотреть:
- `configs/features_catalog.md` (где прямо сказано “в TO_DROP”),
- `training/mvp_types.py` (где она числится в `DEFAULT_MVP_INPUT_TYPES['TO_DROP']`),
- и логи `train_loop_new`.

### 7.2 `TrainingConfig` (даты, таргеты, параметры CatBoost)

В `src/querulus/training/config.py`:
- `date_column = "PAYMENT_ORDER_DATE_TIME"`
- `train_period = ("2022-01-01", "2024-05-31")`
- `test_period = ("2024-06-01", "2025-06-01")`
- `frequency_target = "TARGET_FREQ"` (по умолчанию)
- `severity_target = "TARGET_SEV"` (по умолчанию)
- `severity_range = None` → severity учится только на `target > 0`

Плюс:
- `severity_target_transform`: `"raw"` или `"log1p"`
- `severity_sample_weight`: `"none" | "sqrt" | "linear"`
- опциональная калибровка frequency (`frequency_calibration_enabled`)

### 7.3 Train-loop new (блок B)

В `src/querulus/training/train_loop.py`:
порядок этапов (если флаги включены):
1. PSI/L1 drift filter (train_core vs val)
2. SHAP feature selection (CatBoost `RecursiveByShapValues`)
3. Noise cut (после SHAP)
4. Backward elimination (с конца → подбор лучшего пула по PR-AUC / MAE)
5. Drop zero-importance
6. Fit частоты и тяжести (CatBoost)
7. optional: HPO (Optuna) + MLflow
8. optional: calibration frequency

---

## 8) Как обучаются конкретно frequency и severity

В `src/querulus/training/pipeline.py`:

### 8.1 Frequency (классификация)

- CatBoost `CatBoostClassifier`
- таргет бинарный: `frequency_target` (обычно `TARGET_FREQ`)
- сплит по датам: train/test = по `TrainingConfig.train_period/test_period`
- калибратор:
  - если включён → берётся отдельный Cal-split и обучается `fit_probability_calibrator`

Функция прогноза вероятностей:
- `frequency_predict_proba()` учитывает calibrator.

### 8.2 Severity (регрессия)

- CatBoost `CatBoostRegressor`
- если `severity_range is None`: обучаем severity только на `target > 0`
- таргет под transform:
  - `"raw"` → 그대로
  - `"log1p"` → `log1p(y)`
- sample weights:
  - `"sqrt"` → `sqrt(y)`,
  - `"linear"` → `y`,
  - `"none"` → без весов

Функции:
- `severity_train_target()` / `severity_sample_weights()` / `severity_predict()` в `training/severity_training.py`

---

## 9) FinEffect: как считается “чистый финансовый эффект”

Опорные компоненты:
- `src/querulus/fin_effect/config.py` (имена колонок + параметры)
- `src/querulus/fin_effect/resolve.py` (выбор fact_mode legacy/new по таргетам)
- `src/querulus/fin_effect/calculator.py` (функции расчёта)

### 9.1 База факта и премии

`FinEffectConfig` задаёт:
- `fact_amount_column` (что считать базой “факта”)
- `premiums_column` (`Взносы`)
- `fu_fee_trigger_column` (когда начислять ФУ-взнос)
- fee суммы:
  - `fu_fee_amount = 100000.0`
  - `court_fee_amount = 15000.0`

`payments_fee(row)`:
- взнос ФУ добавляется, только если:
  - `fu_fee_trigger_column > 0`
  - и база факта (в legacy — сумма претензий+ФУ+суд, в icnl — факт-amount) > 0
- судебный взнос добавляется, если включён `apply_court_fee` и есть claims_amount > 0

### 9.2 `fin_effect_fact`

В `compute_fin_effect_fact()`:
- если `uses_legacy_psr_fact`:
  - `pretension_payments + fu_recovery + court_recovery + premiums`
- иначе:
  - `fact_amount_column + premiums`

### 9.3 Модельный эффект `fin_effect_model` по quadrants

Основная функция:
- `compute_fin_effect_model()`:
  - режим `"legacy"` — старые квадранты
  - режим `"coverage"` — “покрытие” (используется по умолчанию для new)

Режим `"coverage"`:
- предсказание частоты: `pred_freq = (proba_freq >= threshold)` (0/1)
- предсказание severity: `y_pred_sev`
- факт severity: `y_true_sev`
- base_sum/psr + premiums задают “стоимость факта”

Правила (в знаковой конвенции проекта; расходы обычно отрицательные в эффектах):
- `y_true_freq=0, pred_freq=0` → `0`
- `y_true_freq=0, pred_freq=1` → `-pred_sev`
- `y_true_freq=1, pred_freq=0` → `-(psr + premiums)`
- `y_true_freq=1, pred_freq=1`:
  - если `y_pred_sev >= y_true_sev` → `-y_pred_sev`
  - иначе (“не хватило”) → доля покрытия рассчитывается как `share = pred_sev / y_true_sev` (clipped 0..1), а эффект:
    `-(psr*(1-share) + premiums)`

### 9.4 Подбор порога

`search_best_threshold()` перебирает сетку `threshold_start..threshold_stop` с шагом `threshold_step` и выбирает:
- максимум `net_effect` (чистая экономия).

Метрики threshold-зонда:
- precision/recall по classification-смыслу (на `TARGET_FREQ` / frequency_target)
- и net_effect в деньгах.

---

## 10) HTTP-сервис (FastAPI): что ожидает и что возвращает

Файл:
`examples/querulus/integration/main.py`

### 10.1 Endpoints

- `GET /api/health`
  - возвращает `{health: true, version: ...}`
- `POST /api/predict`
  - вход: `mldataworker.core.pydantic_models.ServiceRequest`
  - выход: JSON с блоками `oisuu_responce` и `main_response`

### 10.2 Порог классификации

Приоритет:
1. если во входном векторе есть поле `THRESHOLD` → используем его;
2. иначе `config.classification_threshold`;
3. иначе fallback `DEFAULT_CLASSIFICATION_THRESHOLD = 0.6`.

### 10.3 Препроцессинг

До `prepare_dataset()` из `mldataworker`:
- `prepare_common_vector_dataframe`:
  - '' → NaN
  - строковые значения → `UPPER()`
- `enrich_common_vector_dataframe`:
  - считает `EVENT_YEAR`
  - считает `APPLY_DELAY = (PAYMENT_ORDER_DATE_TIME - EVENT_DATE).days`
  - приводит `APPLICANT_FORM` по бизнес-правилам (замена значений)
- строгие runtime-инварианты:
  - даты не NaN, `EVENT_DATE <= PAYMENT_ORDER_DATE_TIME`, `APPLY_DELAY >= -1`, и т.п.

### 10.4 Какие колонки нужны

Списки фиксированы в сервисе:
- `CLASSIFICATION_FEATURES` (для классификации)
- `REGRESSION_FEATURES` (для регрессии)

Если входной вектор не содержит нужных колонок, запрос завершается с ошибкой (fail-fast).

---

## 11) Мониторинг (дрейф)

Файл:
`examples/querulus/monitoring/monitoring.py`

Содержит:
- `DataDrift` наследник `mldataworker.datadrift.DataDrift`
- `UUEMailMonitoring` — отправка результата по email (опционально)
- PSI/дрейф алерт:
  - в письме: `PSI > 0.3`

Основная бизнес-смысловая цель: сигнализировать, что распределения входных фич “поплыли” относительно датасета обучения.

---

## 12) Синтетический smoke-test (как проверить без remote-данных)

Файл:
`src/querulus/synthetic_dataset.py`

Собирает минимальный `df_final_3_synthetic.parquet` для локальной проверки pipeline.
Логика severity/frequency в синтетике повторяет общий смысл:
- severity > 0 только при frequency=1
- есть segment/срезы по VALUE_BEFORE_*.

---

## 13) Как агенту “искать правду”, если возникнут вопросы

Рекомендованные места для проверки (по сути):
- гранулярность/T0/H/S: `configs/features_catalog.md` + `src/querulus/dataset/maturity.py`
- таргеты и формулы: `src/querulus/dataset/steps/targets.py` + `configs/features_catalog.md`
- состав FE и leakage: `configs/features_catalog.md` + `training/mvp_types.py`
- как FE добавляются: `src/querulus/features/pipeline.py`
- что реально вошло в модель: `training/train_loop.py` + артефакты feature selection
- fin_effect формула quadrants/coverage: `src/querulus/fin_effect/calculator.py`
- выбор режима legacy/new для факта: `src/querulus/fin_effect/resolve.py`
- сервис вход-выход: `examples/querulus/integration/main.py`

Если вы хотите, я могу дополнить wiki ещё и “карточками”:
- `TARGET_FREQ vs TARGET_2` (что меняется в fin_effect),
- `coverage vs legacy` (почему знаки/база выглядят непривычно),
- “какие FE считаются leakage и почему” (но для этого лучше прогнать/посмотреть конкретный TO_DROP в `mvp_types.py`).

