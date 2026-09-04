SELECT *
  FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS P
  LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS ITL ON ITL.LossID=P.LossID
