WITH pret_paid AS (
    SELECT
        itl.[IncidentNumber] AS IncidentNumber,
        p.[PretensionNumber] AS PretensionNumber,
        MAX(p.[SurchargeValue]) AS SurchargeValue,
        MAX(p.[UTSSurchargeValue]) AS UTSSurchargeValue
    FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS p
    LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl
        ON p.[LossID] = itl.[LossID]
    WHERE p.[InsuranceTypeGroups] = 'ОСАГО'
      AND p.[AnswerType] IN ({answer_list}){type_filter}
    GROUP BY itl.[IncidentNumber], p.[PretensionNumber]
)
SELECT
    IncidentNumber,
    SUM(SurchargeValue) AS {surcharge_alias},
    SUM(UTSSurchargeValue) AS {uts_alias}
FROM pret_paid
GROUP BY IncidentNumber
