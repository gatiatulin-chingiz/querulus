with tmp as (
	SELECT 
			itl.IncidentID
		 ,itl.IncidentNumber
		 ,itl.LossID
		 ,itl.LossNumber
		 ,p.[PretensionID]
		 ,p.[IsMarked]
		 ,p.PretensionDate
		 ,p.PretensionNumber
		 ,p.[IsOver]
		 ,p.[PretensionGetDate]
		 ,p.[PretensionType]
		 ,p.[ApplicantPersonID]
		 ,p.[PretensionStage]
		 ,p.[PretensionGetMethod]
		 ,p.[PretensionValue]
		 ,p.[PretensionCurrency]
		 ,p.[UTSValue]
		 ,p.[UTSCurrency]
		 ,p.[LossUnit]
		 ,p.[LossUnitZone]
		 ,p.[AnswerType]
		 ,p.[AnswerDate]
		 ,p.[SurchargeValue]
		 ,p.[SurchargeCurrency]
		 ,p.[UTSSurchargeValue]
		 ,p.[UTSSurchargeCurrency]
		 ,p.[Comment]
		 ,p.[HaveRequisitesOfApplicant]
		 ,p.[RequiredReviewSNEO]
		 ,p.[DateCancellationReviews]
		 ,p.[ExternalOrderSNEO]
		 ,p.[IsFullPretensionAmountsWithBreakdown]
		 ,p.[SendInRSADate]
		 ,p.[DateSentScannedCopies]
		 ,p.[PretensionTypeID]
		 ,p.[PretensionTypes]
		 ,p.[PretensionKinds]
		 ,p.[InsuranceTypes]
		 ,p.[InsuranceTypeGroups]
		 ,p.[Cession]
		 ,p.[LinkedLossID]
		 ,_Fld9244RRef
		 ,_Fld9243 as pretension_value_
		 ,SUM(_Fld9243) OVER(PARTITION BY IncidentNumber, PretensionNumber) as PRETENSION_VALUE_PENALTY
		 ,ROW_NUMBER() OVER(PARTITION BY IncidentNumber, PretensionNumber ORDER BY PretensionNumber) as rn
		 ,0 as SURCHARGE_VALUE_PENALTY
	 FROM [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl
	 left join [OISUU_report].[dbo].[oisuu81_t_Pretensions] as p on p.LossID=itl.LossID
	 left join oisuu81.dbo._Document6169_VT9241 vt on vt._Document6169_IDRRef=p.PretensionID
	 WHERE _Fld9244RRef in (0xB6B5441EA172DD2611E8AC27427E4644, 0xB6B5441EA172DD2611E8AC282FFD5C5A)
 ),
penalty as (
	SELECT 
			itl.IncidentID
		 ,itl.IncidentNumber
		 ,itl.LossID
		 ,itl.LossNumber
		 ,p.[PretensionID]
		 ,p.[IsMarked]
		 ,p.PretensionDate
		 ,p.PretensionNumber
		 ,p.[IsOver]
		 ,p.[PretensionGetDate]
		 ,p.[PretensionType]
		 ,p.[ApplicantPersonID]
		 ,p.[PretensionStage]
		 ,p.[PretensionGetMethod]
		 ,p.[PretensionValue]
		 ,p.[PretensionCurrency]
		 ,p.[UTSValue]
		 ,p.[UTSCurrency]
		 ,p.[LossUnit]
		 ,p.[LossUnitZone]
		 ,p.[AnswerType]
		 ,p.[AnswerDate]
		 ,p.[SurchargeValue]
		 ,p.[SurchargeCurrency]
		 ,p.[UTSSurchargeValue]
		 ,p.[UTSSurchargeCurrency]
		 ,p.[Comment]
		 ,p.[HaveRequisitesOfApplicant]
		 ,p.[RequiredReviewSNEO]
		 ,p.[DateCancellationReviews]
		 ,p.[ExternalOrderSNEO]
		 ,p.[IsFullPretensionAmountsWithBreakdown]
		 ,p.[SendInRSADate]
		 ,p.[DateSentScannedCopies]
		 ,p.[PretensionTypeID]
		 ,p.[PretensionTypes]
		 ,p.[PretensionKinds]
		 ,p.[InsuranceTypes]
		 ,p.[InsuranceTypeGroups]
		 ,p.[Cession]
		 ,p.[LinkedLossID]
		 ,_Fld9244RRef
		 ,_Fld9243 as pretension_value_
		 ,SUM(_Fld9243) OVER(PARTITION BY IncidentNumber, PretensionNumber) as PRETENSION_VALUE_PENALTY
		 ,ROW_NUMBER() OVER(PARTITION BY IncidentNumber, PretensionNumber ORDER BY PretensionNumber) as rn
		 ,pai.Amount AS SURCHARGE_VALUE_PENALTY
	 FROM [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl
	 left join [OISUU_report].[dbo].[oisuu81_t_Pretensions] as p on p.LossID=itl.LossID
	 left join oisuu81.dbo._Document6169_VT9241 vt on vt._Document6169_IDRRef=p.PretensionID
	 left join [OISUU_report].[dbo].[oisuu81_vLossPaimentsTable] as pai on pai.LossID=p.LinkedLossID
	 WHERE _Fld9244RRef in (0xB6B5441EA172DD2611E8AC27427E4644, 0xB6B5441EA172DD2611E8AC282FFD5C5A)
	 and BudgetLine = 'Неустойка/пеня'
)
 SELECT *
 FROM tmp as tmp
 WHERE rn = 1
 union select * from penalty where rn = 1
