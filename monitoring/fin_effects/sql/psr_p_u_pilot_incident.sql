/* p_U / mean PSR на уровне ИНЦИДЕНТА
   Источник: [OISUU_report].[Datamart].[oisuu81_t_ПСР] (зерно = убыток)

   Фильтры как в сборке датасета (configs/dataset_filters.json + build_targets),
   плюс окно/филиалы как у ретро-priors (compute_terminal_priors):

   Victim / prepare_victim:
   - Форма_возмещения_детальная ∈ {Ремонт, Денежная, Денежная. Отказ от ремонта, Ремонт. Смена СТОА}
     (= REFUND_FORM_DETAILED)
   - Дата_заявления >= 2022-01-01 (= PAYMENT_ORDER_DATE_TIME date_from)
   - Процесс ∈ {Прямое ОСАГО (с 1 марта 2009), Традиционное ОСАГО}
   - Риск = Ущерб имуществу третьих лиц
   - Тип_объекта = Автотранспорт
   - Вид_страхования содержит ОСАГО (= InsuranceTypeGroup ОСАГО)

   Primary loss (select_primary_loss_per_incident):
   - Флаг_первичнозаявленный_убыток = 1
     (fallback: min Номер_убытка на инцидент — см. комментарий в CTE)

   Ретро-окно p_U (как в excel_monitoring):
   - Дата_заявления ∈ [as_of − 2y; as_of], as_of = 2025-06-30
   - Филиал_УУ ∈ PILOT_FILIALS (10 филиалов без Архангельского/Марийского)

   НЕ перенесено (нет колонок / нужна отдельная витрина):
   - VICTIM_POLICYHOLDER_TYPE = «Физ. Лицо»
   - target_maturity (тишина 6/24 мес, cooloff 24 мес, open court)
   - точное определение TARGET_FREQ_AMOUNT (last claim instance + pret surcharge)
     → здесь ПСР ≈ pret + ФУ + иск из datamart
*/

DECLARE @as_of date = '2025-06-30';
DECLARE @lookback_years float = 2.0;
DECLARE @date_from date = '2022-01-01';  /* dataset_filters.victim.date_from */
DECLARE @start date = DATEADD(day, -CAST(365.25 * @lookback_years AS int), @as_of);

;WITH loss_filtered AS (
    SELECT
        [Номер_инциндента] AS incident_id,
        [Номер_убытка]     AS loss_number,
        [Филиал_УУ]        AS filial,
        [Дата_заявления]   AS app_date,
        ISNULL([Сумма_выплат_по_претензиям], 0) AS pret,
        ISNULL([Сумма_взыскано_по_ФУ], 0)       AS fu,
        ISNULL([Суммы_взыскано_по_иску], 0)     AS court,
        ISNULL([Флаг_наличие_ПСР_в_инциденте], 0) AS psr_flag,
        ISNULL([Флаг_первичнозаявленный_убыток], 0) AS is_primary
    FROM [OISUU_report].[Datamart].[oisuu81_t_ПСР]
    WHERE [Номер_инциндента] IS NOT NULL
      AND [Дата_заявления] >= @date_from
      AND [Процесс] IN (
            N'Прямое ОСАГО (с 1 марта 2009)',
            N'Традиционное ОСАГО'
          )
      AND [Риск] = N'Ущерб имуществу третьих лиц'
      AND [Тип_объекта] = N'Автотранспорт'
      AND [Вид_страхования] LIKE N'%ОСАГО%'
      AND [Форма_возмещения_детальная] IN (
            N'Ремонт',
            N'Денежная',
            N'Денежная. Отказ от ремонта',
            N'Ремонт. Смена СТОА'
          )
),
/* один убыток на инцидент: флаг первичного; иначе min Номер_убытка */
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY incident_id
            ORDER BY
                CASE WHEN is_primary = 1 THEN 0 ELSE 1 END,
                loss_number
        ) AS rn
    FROM loss_filtered
),
incident AS (
    SELECT
        incident_id,
        filial,
        app_date,
        pret,
        fu,
        court,
        pret + fu + court AS PSR_amount,
        psr_flag AS has_psr_flag
    FROM ranked
    WHERE rn = 1
),
base AS (
    SELECT *
    FROM incident
    WHERE app_date >= @start
      AND app_date <= @as_of
      AND app_date >= @date_from
      AND filial IN (
            N'Владимирский',
            N'Кемеровский',
            N'Курский',
            N'Магнитогорский',
            N'Мурманский',
            N'Омский',
            N'Пермский',
            N'Петропавловск-Камчатский',
            N'Уфимский',
            N'Ярославский'
          )
)
SELECT
    COUNT(*) AS n_incidents,
    SUM(CASE WHEN PSR_amount > 0 THEN 1 ELSE 0 END) AS n_positive,
    CAST(SUM(CASE WHEN PSR_amount > 0 THEN 1 ELSE 0 END) AS float)
        / NULLIF(COUNT(*), 0) AS p_U,
    AVG(CAST(PSR_amount AS float)) AS mean_PSR_all,
    AVG(CASE WHEN PSR_amount > 0 THEN CAST(PSR_amount AS float) END) AS m_U,
    AVG(CAST(has_psr_flag AS float)) AS p_U_by_flag
FROM base;

/* --- опционально: разбивка по филиалам ---
SELECT
    filial,
    COUNT(*) AS n_incidents,
    CAST(SUM(CASE WHEN PSR_amount > 0 THEN 1 ELSE 0 END) AS float)
        / NULLIF(COUNT(*), 0) AS p_U,
    AVG(CAST(PSR_amount AS float)) AS mean_PSR_all,
    AVG(CASE WHEN PSR_amount > 0 THEN CAST(PSR_amount AS float) END) AS m_U
FROM base
GROUP BY filial
ORDER BY filial;
*/
