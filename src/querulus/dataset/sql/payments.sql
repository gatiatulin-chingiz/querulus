SELECT
    ITL.IncidentNumber,
    p.PaymentDateTime,
    p.PaymentValue
FROM [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as ITL
LEFT JOIN [OISUU_report].[dbo].oisuu81_t_payments AS p on p.LOSSID = ITL.LOSSID
