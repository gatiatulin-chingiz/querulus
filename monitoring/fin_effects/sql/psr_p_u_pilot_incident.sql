/* p_U / mean PSR на уровне ИНЦИДЕНТА
   Источник: убытки в [OISUU_report].[Datamart].[oisuu81_t_ПСР]
   Фильтры:
   - дата ∈ [as_of - 2y; as_of], as_of = 2025-06-30
   - филиал ∈ PILOT_FILIALS
   - Процесс ∈ {Прямое ОСАГО (с 1 марта 2009), Традиционное ОСАГО}
   - Тип_объекта = Автотранспорт
   Дата: Дата_заявления ≈ PAYMENT_ORDER_DATE_TIME
   ПСР: претензия + ФУ + иск (схлоп сумм с убытков на инцидент)
*/

DECLARE @as_of date = '2025-06-30';
DECLARE @lookback_years float = 2.0;
DECLARE @start date = DATEADD(day, -CAST(365.25 * @lookback_years AS int), @as_of);

;WITH loss AS (
    SELECT
        [Номер_инциндента] AS incident_id,
        [Филиал_УУ]        AS filial,
        [Дата_заявления]   AS app_date,
        ISNULL([Сумма_выплат_по_претензиям], 0) AS pret,
        ISNULL([Сумма_взыскано_по_ФУ], 0)       AS fu,
        ISNULL([Суммы_взыскано_по_иску], 0)     AS court,
        ISNULL([Флаг_наличие_ПСР_в_инциденте], 0) AS psr_flag
    FROM [OISUU_report].[Datamart].[oisuu81_t_ПСР]
    WHERE [Номер_инциндента] IS NOT NULL
      AND [Процесс] IN (
            N'Прямое ОСАГО (с 1 марта 2009)',
            N'Традиционное ОСАГО'
          )
      AND [Тип_объекта] = N'Автотранспорт'
),
incident AS (
    SELECT
        incident_id,
        /* если вдруг разные филиалы внутри инцидента — MAX; см. проверку ниже */
        MAX(filial) AS filial,
        MIN(app_date) AS app_date,
        SUM(pret)  AS pret,
        SUM(fu)    AS fu,
        SUM(court) AS court,
        SUM(pret) + SUM(fu) + SUM(court) AS PSR_amount,
        MAX(psr_flag) AS has_psr_flag
    FROM loss
    GROUP BY incident_id
),
base AS (
    SELECT *
    FROM incident
    WHERE app_date >= @start
      AND app_date <= @as_of
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

/* --- опционально: инциденты с конфликтом филиала ---
SELECT incident_id, COUNT(DISTINCT filial) AS n_filials
FROM loss
GROUP BY incident_id
HAVING COUNT(DISTINCT filial) > 1;
*/
