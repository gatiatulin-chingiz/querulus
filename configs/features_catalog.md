# Каталог признаков датасета querulus

**Гранулярность:** 1 строка = 1 инцидент (первичный убыток, max `LOSS_NUMBER`).  
**T0:** `PAYMENT_ORDER_DATE_TIME` — якорь времени для person-history.  
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
| `FE_DAYS_LOSS_TO_T0` | Дней от убытка до выплаты | `PAYMENT_ORDER_DATE_TIME − LOSS_DATE_TIME` |
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
| `FE_GUILTY_POWER_PER_TON` | Мощность/тонна виновника | `GUILTY_CAPACITY_ENGINE / GUILTY_MAX_WEIGHT` |
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
| `FE_PREMIUM_PER_POLICY` | Премия на полис | `PREMIUM_SUM_ALL / PREMIUM_COUNT_ALL` |
| `FE_INSURANCE_AMOUNT_BIN` | Бакет страховой суммы | `<400k / 400k-1M / >1M` из `INSURANCE_AMOUNT` |

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

### I. Калькуляция / ремонт

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_WEAROUT_TIER` | Tier износа | 0-20 / 20-50 / 50+ из `SHARE_WEAROUT` |
| `FE_SHARE_WORK_TIER` | Доля работ | low/mid/high из `SHARE_WORK` |
| `FE_EXPECTED_WEAROUT_RUB` | Ожидаемый износ в руб. | `AMOUNT_REPAIR * SHARE_WEAROUT / 100` |
| `FE_AMOUNT_REPAIR_BIN` | Бакет суммы ремонта | `<100k / 100-300k / >300k` |
| `FE_HIGH_REPAIR` | Дорогой ремонт | `AMOUNT_REPAIR > 300000` |
| `FE_REPAIR_TO_VALUE_RATIO` | Отношение выплаты к калькуляции | `REPAIR_VALUE / AMOUNT_REPAIR` |

### J. История убытков (past only, из victim)

| Фича | Описание | Как собирается |
|------|----------|----------------|
| `FE_VICTIM_LOSS_COUNT_BIN` | Число прошлых убытков victim | 0/1/2/3+ из `VICTIM_LOSS_COUNT` |
| `FE_VICTIM_REPEAT` | Повторный клиент victim | `VICTIM_LOSS_COUNT > 0` |
| `FE_VICTIM_LOSS_SUM_BIN` | Сумма прошлых убытков | бакеты из `VICTIM_LOSS_SUM` |
| `FE_GUILTY_LOSS_COUNT_BIN` | Число прошлых убытков виновника | из `GUILTY_LOSS_COUNT` |
| `FE_GUILTY_REPEAT` | Повторный виновник | `GUILTY_LOSS_COUNT > 0` |

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
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_COUNT` | Число претензий в истории | count строк после фильтра |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_PRETENSION_NUMBER_NUNIQUE` | Уникальные номера претензий | nunique `PRETENSION_NUMBER` |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_TYPES_NUNIQUE` | Уникальные типы | nunique `PRETENSION_TYPES` |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_GET_METHOD_MODE` | Мода способа подачи | mode `PRETENSION_GET_METHOD` |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_ANSWER_TYPE_MODE` | Мода типа ответа | mode `ANSWER_TYPE` |
| `FE_PERSON_PRET_{ROLE}_FE_PERSON_PRET_{MONEY}_SUM` | Сумма по статье | sum для `PRETENSION_VALUE`, `SURCHARGE_VALUE`, `UTS_SURCHARGE_VALUE`, `PRETENSION_VALUE_PENALTY`, `SURCHARGE_VALUE_PENALTY` |

### 3.3 Court history (`history_court.py`)

Источники: `oisuu81_t_IncomingClaimNewLogicByInst` + `oisuu81_t_Истцы` (join по номеру иска, person = `Лицо`).

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
| `AMOUNT_REPAIR` | Сумма ремонта без износа | `_InfoRg14746` |
| `SHARE_WEAROUT` | Процент износа (cap 50) | `_InfoRg14746` |
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
| `EventLocationRegionName`, `LONGITUDE`, `LATITUDE` | Гео ДТП | Losses |
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
- `EMRValue`, `REFUND_FORM_DETAILED`, `REPAIR_VALUE`;
- PVU/season future-колонки;
- `*_PERSON_ID`, FIO-колонки (рекомендуется держать в denylist).

---

## 7. Оценка объёма

| Блок | Порядок величины |
|------|------------------|
| Victim raw | ~230 колонок |
| `FE_*` derived | 59 |
| `FE_PERSON_STATIC` | ~55 |
| `FE_PERSON_PRET_{ROLE}_*` | ~9 × 10 ролей ≈ 90 |
| `FE_PERSON_COURT_{ROLE}_*` | ~80+ × 10 ролей ≈ 800+ |
| **Итого в df_final_3** | **~1100+ колонок** |
| **В модели (после AutoMVP)** | ~200–400 (зависит от наполненности) |

---

*Файл: `configs/features_catalog.md`. Код-источник: `features/derived.py`, `features/person/*`, `dataset/steps/*`, `training/mvp_types.py`.*
