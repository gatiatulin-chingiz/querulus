USE [OISUU_report]
GO

/****** Object: StoredProcedure [dbo].[oisuu81_uspUpdateIncomingClaimsNewLogic] Script Date: 09.07.2026 13:15:02 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO











--exec [dbo].[oisuu81_uspUpdateIncomingClaimsNewLogic]

CREATE procedure [dbo].[oisuu81_uspUpdateIncomingClaimsNewLogic] as
Drop table if exists #FirstValue

Select ic1.*
into #FirstValue
From oisuu81_t_IncomingClaims ic1 inner join 
(select IncomingClaimID,InstByOisuu,max(ValuePeriod) MaxVal/*,CaseState MaxCS*/ From oisuu81_t_IncomingClaims where CaseState in (1,2) and MarkedCaseState = 1 group by IncomingClaimID,InstByOisuu/*,CaseState*/) ic2 
on ic1.IncomingClaimID = ic2.IncomingClaimID /*and ic1.CaseState = ic2.MaxCS*/ and ic1.InstByOisuu = ic2.InstByOisuu and ic2.MaxVal = ic1.ValuePeriod
where MarkedCaseState = 1

--(559208 rows affected) Completion time: 2023-06-23T11:51:33.1380279+03:00

Drop table if exists #ResolValue

Select ic1.*
into #ResolValue
From oisuu81_t_IncomingClaims ic1 inner join 
(select IncomingClaimID,InstByOisuu,max(ValuePeriod) MaxVal From oisuu81_t_IncomingClaims where CaseState not in (1,2) and MarkedCaseState = 1 group by IncomingClaimID,InstByOisuu) ic2 
on ic1.IncomingClaimID = ic2.IncomingClaimID and ic2.MaxVal = ic1.ValuePeriod and ic1.InstByOisuu = ic2.InstByOisuu
where MarkedCaseState = 1

--(547751 rows affected) Completion time: 2023-06-23T11:51:33.1380279+03:00

Drop table if exists #group

Select * into #group from #FirstValue
Union
Select * from #ResolValue

--Select * from #group where IncomingClaimNumber = 190640
Drop table if exists #Claimed

Select t1.IncomingClaimID
	,t1.InstByOisuu
	,t1.IncomingClaimNumber
	,t1.ValuePeriod ClaimedValuePeriod
	--,t1.ValueChangeTypeByTime ClaimedValueChangeTypeByTime
	,SUM(t1.MainDebt) ClaimedMainDebt
	,SUM(t1.PlaintiffExamination) ClaimedPlaintiffExamination
	,SUM(t1.CourtExamination) ClaimedCourtExamination
	,SUM(t1.RepresentativeExpenses) ClaimedRepresentativeExpenses
	,SUM(t1.PercentForUses) ClaimedPercentForUses
	,SUM(t1.PenaltyFee) ClaimedPenaltyFee
	,SUM(t1.Fine) ClaimedFine
	,SUM(t1.MoralDamage) ClaimedMoralDamage
	,SUM(t1.OtherExpenses) ClaimedOtherExpenses
	,SUM(t1.StateDuty) ClaimedStateDuty
	,SUM(t1.LossCommodyValue) ClaimedLossCommodyValue
	,SUM(t1.Wearout) ClaimedWearout
	,SUM(t1.ValueWithoutSD) ClaimedValueWithoutSD
	,SUM(t1.ValueWithSD) ClaimedValueWithSD
	,SUM(t1.[AmountLoss]) ClaimedAmountLoss
	,SUM(t1.[CourtExaminationBeforeResolve]) ClaimedCourtExaminationBeforeResolve
	,t1.Instance
	,t1.instID
	,t1.Court
	,t1.Judge
	,t1.Applicant
	,t1.ExpertOrg
	,t1.Decision
into #Claimed
From #group t1 
where t1.CaseState in (1,2) 
Group by t1.IncomingClaimID
	,t1.InstByOisuu
	,t1.IncomingClaimNumber
	,t1.ValuePeriod 
	--,t1.ValueChangeTypeByTime
	,t1.Court
	,t1.Judge
	,t1.Applicant
	,t1.ExpertOrg
	,t1.Decision
	,t1.Instance
	,t1.instID


Drop table if exists #Recovered

Select t1.IncomingClaimID
	,t1.InstByOisuu
	,t1.IncomingClaimNumber
	,t1.ValuePeriod RecoveredValuePeriod
	--,t1.ValueChangeTypeByTime RecoveredValueChangeTypeByTime
	,SUM(t1.MainDebt) RecoveredMainDebt
	,SUM(t1.PlaintiffExamination) RecoveredPlaintiffExamination
	,SUM(t1.CourtExamination) RecoveredCourtExamination
	,SUM(t1.RepresentativeExpenses) RecoveredRepresentativeExpenses
	,SUM(t1.PercentForUses) RecoveredPercentForUses
	,SUM(t1.PenaltyFee) RecoveredPenaltyFee
	,SUM(t1.Fine) RecoveredFine
	,SUM(t1.MoralDamage) RecoveredMoralDamage
	,SUM(t1.OtherExpenses) RecoveredOtherExpenses
	,SUM(t1.StateDuty) RecoveredStateDuty
	,SUM(t1.LossCommodyValue) RecoveredLossCommodyValue
	,SUM(t1.Wearout) RecoveredWearout
	,SUM(t1.ValueWithoutSD) RecoveredValueWithoutSD
	,SUM(t1.ValueWithSD) RecoveredValueWithSD
	,SUM(t1.[AmountLoss]) RecoveredAmountLoss
	,SUM(t1.[CourtExaminationBeforeResolve]) RecoveredCourtExaminationBeforeResolve
into #Recovered
From #group t1 
where t1.CaseState not in (1,2) 
Group by t1.IncomingClaimID
	,t1.InstByOisuu
	,t1.IncomingClaimNumber
	,t1.ValuePeriod 
	--,t1.ValueChangeTypeByTime

Drop table if exists #linkLoss

SELECT distinct [IncomingClaimID]
	,min(LossNumber) LossNumber
into #linkLoss
FROM [OISUU_report].[dbo].[oisuu81_t_ClaimLosses] clloss inner join oisuu81_t_Losses l on clloss.LossID = l.LossID
 where Origin = 'Связанные'
 --	and [IsClaimLoss] = 0
	group by [IncomingClaimID]
	order by 1,2

Drop table if exists #vtPreItog
 
Select distinct cl.IncomingClaimID
	,cl.InstByOisuu
	,cl.IncomingClaimNumber
	,cl.ClaimedValuePeriod
	--,cl.ClaimedValueChangeTypeByTime
	,cl.ClaimedMainDebt
	,cl.ClaimedPlaintiffExamination
	,cl.ClaimedCourtExamination
	,cl.ClaimedRepresentativeExpenses
	,cl.ClaimedPercentForUses
	,cl.ClaimedPenaltyFee
	,cl.ClaimedFine
	,cl.ClaimedMoralDamage
	,cl.ClaimedOtherExpenses
	,cl.ClaimedStateDuty
	,cl.ClaimedLossCommodyValue
	,cl.ClaimedWearout
	,cl.ClaimedValueWithoutSD
	,cl.ClaimedValueWithSD
	,cl.ClaimedAmountLoss
	,cl.ClaimedCourtExaminationBeforeResolve
	,rec.RecoveredValuePeriod
	--,rec.RecoveredValueChangeTypeByTime
	,rec.RecoveredMainDebt
	,rec.RecoveredPlaintiffExamination
	,rec.RecoveredCourtExamination
	,rec.RecoveredRepresentativeExpenses
	,rec.RecoveredPercentForUses
	,rec.RecoveredPenaltyFee
	,rec.RecoveredFine
	,rec.RecoveredMoralDamage
	,rec.RecoveredOtherExpenses
	,rec.RecoveredStateDuty
	,rec.RecoveredLossCommodyValue
	,rec.RecoveredWearout
	,rec.RecoveredValueWithoutSD
	,rec.RecoveredValueWithSD
	,rec.RecoveredAmountLoss
	,rec.RecoveredCourtExaminationBeforeResolve
	,cl.Instance
	,cl.instID
	,cl.Court
	,cl.Judge
	,cl.Applicant
	,cl.ExpertOrg
	,cl.Decision
	,cl2.[IncomingClaimGetDate]
 ,cl2.[CourtWorkOverDate]
 ,cl2.[ClaimItem]
 ,cl2.[ClaimOrigin]
 ,cl2.[CourtWorkUnit]
	,cl2.[EmployeeName]
	,cl2.FLSimpleOrder
	,ll.LossNumber LinkLossNumber
	,ROW_NUMBER() over (partition by cl.IncomingClaimNumber order by cl.InstByOisuu) rn
into #vtPreItog
From #Claimed cl 
left join #Recovered rec 
on cl.IncomingClaimID = rec.IncomingClaimID
and cl.InstByOisuu = rec.InstByOisuu
left join (SELECT distinct [IncomingClaimID]
 ,[IncomingClaimGetDate]
 ,[CourtWorkOverDate]
 ,[ClaimItem]
 ,[ClaimOrigin]
 ,[CourtWorkUnit]
	 ,[EmployeeName]
	 ,Cast(FLSimpleOrder as int) FLSimpleOrder
 FROM [OISUU_report].[dbo].[oisuu81_t_IncomingClaims]) cl2
on cl.IncomingClaimID = cl2.IncomingClaimID
left join #linkLoss ll on ll.IncomingClaimID = cl.IncomingClaimID

--(553646 rows affected) Completion time: 2023-06-26T09:31:39.4091878+03:00

Drop table if exists #general

Select distinct IncomingClaimID
	,IncomingClaimNumber 
	,[IncomingClaimGetDate]
 ,[CourtWorkOverDate]
 ,[ClaimItem]
 ,[ClaimOrigin]
 ,[CourtWorkUnit]
	,[EmployeeName]
	,FLSimpleOrder
	,LinkLossNumber
into #general
from #vtPreItog 

--(466909 rows affected) Completion time: 2023-06-26T10:08:12.2762470+03:00

Drop table if exists #FUValue

Select distinct IncomingClaimID
	,ClaimedValuePeriod FU_ClaimedValuePeriod
	--,ClaimedValueChangeTypeByTime FU_ClaimedValueChangeTypeByTime
	,ClaimedMainDebt FU_ClaimedMainDebt
	,ClaimedPlaintiffExamination FU_ClaimedPlaintiffExamination
	,ClaimedCourtExamination FU_ClaimedCourtExamination
	,ClaimedRepresentativeExpenses FU_ClaimedRepresentativeExpenses
	,ClaimedPercentForUses FU_ClaimedPercentForUses
	,ClaimedPenaltyFee FU_ClaimedPenaltyFee
	,ClaimedFine FU_ClaimedFine
	,ClaimedMoralDamage FU_ClaimedMoralDamage
	,ClaimedOtherExpenses FU_ClaimedOtherExpenses
	,ClaimedStateDuty FU_ClaimedStateDuty
	,ClaimedLossCommodyValue FU_ClaimedLossCommodyValue
	,ClaimedWearout FU_ClaimedWearout
	,ClaimedValueWithoutSD FU_ClaimedValueWithoutSD
	,ClaimedValueWithSD FU_ClaimedValueWithSD
	,RecoveredValuePeriod FU_RecoveredValuePeriod
	--,RecoveredValueChangeTypeByTime FU_RecoveredValueChangeTypeByTime
	,RecoveredMainDebt FU_RecoveredMainDebt
	,RecoveredPlaintiffExaminationFU_RecoveredPlaintiffExamination
	,RecoveredCourtExamination FU_RecoveredCourtExamination
	,RecoveredRepresentativeExpenses FU_RecoveredRepresentativeExpenses
	,RecoveredPercentForUses FU_RecoveredPercentForUses
	,RecoveredPenaltyFee FU_RecoveredPenaltyFee
	,RecoveredFine FU_RecoveredFine
	,RecoveredMoralDamage FU_RecoveredMoralDamage
	,RecoveredOtherExpenses FU_RecoveredOtherExpenses
	,RecoveredStateDuty FU_RecoveredStateDuty
	,RecoveredLossCommodyValue FU_RecoveredLossCommodyValue
	,RecoveredWearout FU_RecoveredWearout
	,RecoveredValueWithoutSD FU_RecoveredValueWithoutSD
	,RecoveredValueWithSD FU_RecoveredValueWithSD
	,Court FU_Court
	,Judge FU_Judge
	,Applicant FU_Applicant
	,ExpertOrg FU_ExpertOrg
	,Decision FU_Decision
into #FUValue
From #vtPreItog
where InstByOisuu = 6

Drop table if exists #FirstInstValue

Select distinct IncomingClaimID
	,ClaimedValuePeriod FirstInst_ClaimedValuePeriod
	--,ClaimedValueChangeTypeByTime FirstInst_ClaimedValueChangeTypeByTime
	,ClaimedMainDebt FirstInst_ClaimedMainDebt
	,ClaimedPlaintiffExamination FirstInst_ClaimedPlaintiffExamination
	,ClaimedCourtExamination FirstInst_ClaimedCourtExamination
	,ClaimedRepresentativeExpenses FirstInst_ClaimedRepresentativeExpenses
	,ClaimedPercentForUses FirstInst_ClaimedPercentForUses
	,ClaimedPenaltyFee FirstInst_ClaimedPenaltyFee
	,ClaimedFine FirstInst_ClaimedFine
	,ClaimedMoralDamage FirstInst_ClaimedMoralDamage
	,ClaimedOtherExpenses FirstInst_ClaimedOtherExpenses
	,ClaimedStateDuty FirstInst_ClaimedStateDuty
	,ClaimedLossCommodyValue FirstInst_ClaimedLossCommodyValue
	,ClaimedWearout FirstInst_ClaimedWearout
	,ClaimedValueWithoutSD FirstInst_ClaimedValueWithoutSD
	,ClaimedValueWithSD FirstInst_ClaimedValueWithSD
	,RecoveredValuePeriod FirstInst_RecoveredValuePeriod
	--,RecoveredValueChangeTypeByTime FirstInst_RecoveredValueChangeTypeByTime
	,RecoveredMainDebt FirstInst_RecoveredMainDebt
	,RecoveredPlaintiffExamination FirstInst_RecoveredPlaintiffExamination
	,RecoveredCourtExamination FirstInst_RecoveredCourtExamination
	,RecoveredPercentForUses FirstInst_RecoveredPercentForUses
	,RecoveredPenaltyFee FirstInst_RecoveredPenaltyFee
	,RecoveredFine FirstInst_RecoveredFine
	,RecoveredMoralDamage FirstInst_RecoveredMoralDamage
	,RecoveredOtherExpenses FirstInst_RecoveredOtherExpenses
	,RecoveredStateDuty FirstInst_RecoveredStateDuty
	,RecoveredLossCommodyValue FirstInst_RecoveredLossCommodyValue
	,RecoveredWearout FirstInst_RecoveredWearout
	,RecoveredValueWithoutSD FirstInst_RecoveredValueWithoutSD
	,RecoveredValueWithSD FirstInst_RecoveredValueWithSD
	,Court FirstInst_Court
	,Judge FirstInst_Judge
	,Applicant FirstInst_Applicant
	,ExpertOrg FirstInst_ExpertOrg
	,Decision FirstInst_Decision
into #FirstInstValue
From #vtPreItog
where InstByOisuu = 1

Drop table if exists #SecondInstValue

Select distinct IncomingClaimID
	,ClaimedValuePeriod SecondInst_ClaimedValuePeriod
	--,ClaimedValueChangeTypeByTime SecondInst_ClaimedValueChangeTypeByTime
	,ClaimedMainDebt SecondInst_ClaimedMainDebt
	,ClaimedPlaintiffExamination SecondInst_ClaimedPlaintiffExamination
	,ClaimedCourtExamination SecondInst_ClaimedCourtExamination
	,ClaimedRepresentativeExpenses SecondInst_ClaimedRepresentativeExpenses
	,ClaimedPercentForUses SecondInst_ClaimedPercentForUses
	,ClaimedPenaltyFee SecondInst_ClaimedPenaltyFee
	,ClaimedFine SecondInst_ClaimedFine
	,ClaimedMoralDamage SecondInst_ClaimedMoralDamage
	,ClaimedOtherExpenses SecondInst_ClaimedOtherExpenses
	,ClaimedStateDuty SecondInst_ClaimedStateDuty
	,ClaimedLossCommodyValue SecondInst_ClaimedLossCommodyValue
	,ClaimedWearout SecondInst_ClaimedWearout
	,ClaimedValueWithoutSD SecondInst_ClaimedValueWithoutSD
	,ClaimedValueWithSD SecondInst_ClaimedValueWithSD
	,RecoveredValuePeriod SecondInst_RecoveredValuePeriod
	--,RecoveredValueChangeTypeByTime SecondInst_RecoveredValueChangeTypeByTime
	,RecoveredMainDebtSecondInst_RecoveredMainDebt
	,RecoveredPlaintiffExamination SecondInst_RecoveredPlaintiffExamination
	,RecoveredCourtExamination SecondInst_RecoveredCourtExamination
	,RecoveredPercentForUses SecondInst_RecoveredPercentForUses
	,RecoveredPenaltyFee SecondInst_RecoveredPenaltyFee
	,RecoveredFine SecondInst_RecoveredFine
	,RecoveredMoralDamage SecondInst_RecoveredMoralDamage
	,RecoveredOtherExpenses SecondInst_RecoveredOtherExpenses
	,RecoveredStateDuty SecondInst_RecoveredStateDuty
	,RecoveredLossCommodyValue SecondInst_RecoveredLossCommodyValue
	,RecoveredWearout SecondInst_RecoveredWearout
	,RecoveredValueWithoutSD SecondInst_RecoveredValueWithoutSD
	,RecoveredValueWithSD SecondInst_RecoveredValueWithSD
	,Court SecondInst_Court
	,Judge SecondInst_Judge
	,Applicant SecondInst_Applicant
	,ExpertOrg SecondInst_ExpertOrg
	,Decision SecondInst_Decision
into #SecondInstValue
From #vtPreItog
where InstByOisuu = 2

Drop table if exists #ThirdInstValue

Select distinct IncomingClaimID
	,ClaimedValuePeriod ThirdInst_ClaimedValuePeriod
	--,ClaimedValueChangeTypeByTime ThirdInst_ClaimedValueChangeTypeByTime
	,ClaimedMainDebt ThirdInst_ClaimedMainDebt
	,ClaimedPlaintiffExamination ThirdInst_ClaimedPlaintiffExamination
	,ClaimedCourtExamination ThirdInst_ClaimedCourtExamination
	,ClaimedRepresentativeExpenses ThirdInst_ClaimedRepresentativeExpenses
	,ClaimedPercentForUses ThirdInst_ClaimedPercentForUses
	,ClaimedPenaltyFee ThirdInst_ClaimedPenaltyFee
	,ClaimedFine ThirdInst_ClaimedFine
	,ClaimedMoralDamage ThirdInst_ClaimedMoralDamage
	,ClaimedOtherExpenses ThirdInst_ClaimedOtherExpenses
	,ClaimedStateDuty ThirdInst_ClaimedStateDuty
	,ClaimedLossCommodyValue ThirdInst_ClaimedLossCommodyValue
	,ClaimedWearout ThirdInst_ClaimedWearout
	,ClaimedValueWithoutSD ThirdInst_ClaimedValueWithoutSD
	,ClaimedValueWithSD ThirdInst_ClaimedValueWithSD
	,RecoveredValuePeriod ThirdInst_RecoveredValuePeriod
	--,RecoveredValueChangeTypeByTime ThirdInst_RecoveredValueChangeTypeByTime
	,RecoveredMainDebt ThirdInst_RecoveredMainDebt
	,RecoveredPlaintiffExamination ThirdInst_RecoveredPlaintiffExamination
	,RecoveredCourtExamination ThirdInst_RecoveredCourtExamination
	,RecoveredPercentForUses ThirdInst_RecoveredPercentForUses
	,RecoveredPenaltyFee ThirdInst_RecoveredPenaltyFee
	,RecoveredFine ThirdInst_RecoveredFine
	,RecoveredMoralDamage ThirdInst_RecoveredMoralDamage
	,RecoveredOtherExpenses ThirdInst_RecoveredOtherExpenses
	,RecoveredStateDuty ThirdInst_RecoveredStateDuty
	,RecoveredLossCommodyValue ThirdInst_RecoveredLossCommodyValue
	,RecoveredWearout ThirdInst_RecoveredWearout
	,RecoveredValueWithoutSD ThirdInst_RecoveredValueWithoutSD
	,RecoveredValueWithSD ThirdInst_RecoveredValueWithSD
	,Court ThirdInst_Court
	,Judge ThirdInst_Judge
	,Applicant ThirdInst_Applicant
	,ExpertOrg ThirdInst_ExpertOrg
	,Decision ThirdInst_Decision
into #ThirdInstValue
From #vtPreItog
where InstByOisuu = 3

Drop table if exists #FourthInstValue

Select distinct IncomingClaimID
	,ClaimedValuePeriod FourthInst_ClaimedValuePeriod
	--,ClaimedValueChangeTypeByTime FourthInst_ClaimedValueChangeTypeByTime
	,ClaimedMainDebt FourthInst_ClaimedMainDebt
	,ClaimedPlaintiffExamination FourthInst_ClaimedPlaintiffExamination
	,ClaimedCourtExamination FourthInst_ClaimedCourtExamination
	,ClaimedRepresentativeExpenses FourthInst_ClaimedRepresentativeExpenses
	,ClaimedPercentForUses FourthInst_ClaimedPercentForUses
	,ClaimedPenaltyFee FourthInst_ClaimedPenaltyFee
	,ClaimedFine FourthInst_ClaimedFine
	,ClaimedMoralDamage FourthInst_ClaimedMoralDamage
	,ClaimedOtherExpenses FourthInst_ClaimedOtherExpenses
	,ClaimedStateDuty FourthInst_ClaimedStateDuty
	,ClaimedLossCommodyValue FourthInst_ClaimedLossCommodyValue
	,ClaimedWearout FourthInst_ClaimedWearout
	,ClaimedValueWithoutSD FourthInst_ClaimedValueWithoutSD
	,ClaimedValueWithSD FourthInst_ClaimedValueWithSD
	,RecoveredValuePeriodFourthInst_RecoveredValuePeriod
	--,RecoveredValueChangeTypeByTime FourthInst_RecoveredValueChangeTypeByTime
	,RecoveredMainDebt FourthInst_RecoveredMainDebt
	,RecoveredPlaintiffExamination FourthInst_RecoveredPlaintiffExamination
	,RecoveredCourtExamination FourthInst_RecoveredCourtExamination
	,RecoveredPercentForUses FourthInst_RecoveredPercentForUses
	,RecoveredPenaltyFee FourthInst_RecoveredPenaltyFee
	,RecoveredFine FourthInst_RecoveredFine
	,RecoveredMoralDamage FourthInst_RecoveredMoralDamage
	,RecoveredOtherExpenses FourthInst_RecoveredOtherExpenses
	,RecoveredStateDuty FourthInst_RecoveredStateDuty
	,RecoveredLossCommodyValue FourthInst_RecoveredLossCommodyValue
	,RecoveredWearout FourthInst_RecoveredWearout
	,RecoveredValueWithoutSD FourthInst_RecoveredValueWithoutSD
	,RecoveredValueWithSD FourthInst_RecoveredValueWithSD
	,Court FourthInst_Court
	,Judge FourthInst_Judge
	,Applicant FourthInst_Applicant
	,ExpertOrg FourthInst_ExpertOrg
	,Decision FourthInst_Decision
into #FourthInstValue
From #vtPreItog
where InstByOisuu = 4

Drop table if exists #FifthInstValue

Select distinct IncomingClaimID
	,ClaimedValuePeriod FifthInst_ClaimedValuePeriod
	--,ClaimedValueChangeTypeByTime FifthInst_ClaimedValueChangeTypeByTime
	,ClaimedMainDebt FifthInst_ClaimedMainDebt
	,ClaimedPlaintiffExamination FifthInst_ClaimedPlaintiffExamination
	,ClaimedCourtExamination FifthInst_ClaimedCourtExamination
	,ClaimedRepresentativeExpenses FifthInst_ClaimedRepresentativeExpenses
	,ClaimedPercentForUses FifthInst_ClaimedPercentForUses
	,ClaimedPenaltyFee FifthInst_ClaimedPenaltyFee
	,ClaimedFine FifthInst_ClaimedFine
	,ClaimedMoralDamage FifthInst_ClaimedMoralDamage
	,ClaimedOtherExpenses FifthInst_ClaimedOtherExpenses
	,ClaimedStateDuty FifthInst_ClaimedStateDuty
	,ClaimedLossCommodyValue FifthInst_ClaimedLossCommodyValue
	,ClaimedWearout FifthInst_ClaimedWearout
	,ClaimedValueWithoutSD FifthInst_ClaimedValueWithoutSD
	,ClaimedValueWithSD FifthInst_ClaimedValueWithSD
	,RecoveredValuePeriod FifthInst_RecoveredValuePeriod
	--,RecoveredValueChangeTypeByTime FifthInst_RecoveredValueChangeTypeByTime
	,RecoveredMainDebt FifthInst_RecoveredMainDebt
	,RecoveredPlaintiffExamination FifthInst_RecoveredPlaintiffExamination
	,RecoveredCourtExamination FifthInst_RecoveredCourtExamination
	,RecoveredPercentForUses FifthInst_RecoveredPercentForUses
	,RecoveredPenaltyFee FifthInst_RecoveredPenaltyFee
	,RecoveredFine FifthInst_RecoveredFine
	,RecoveredMoralDamage FifthInst_RecoveredMoralDamage
	,RecoveredOtherExpenses FifthInst_RecoveredOtherExpenses
	,RecoveredStateDuty FifthInst_RecoveredStateDuty
	,RecoveredLossCommodyValue FifthInst_RecoveredLossCommodyValue
	,RecoveredWearout FifthInst_RecoveredWearout
	,RecoveredValueWithoutSD FifthInst_RecoveredValueWithoutSD
	,RecoveredValueWithSD FifthInst_RecoveredValueWithSD
	,Court FifthInst_Court
	,Judge FifthInst_Judge
	,Applicant FifthInst_Applicant
	,ExpertOrg FifthInst_ExpertOrg
	,Decision FifthInst_Decision
into #FifthInstValue
From #vtPreItog
where InstByOisuu = 5

drop table if exists #itog

Select distinct g.IncomingClaimID
	,g.IncomingClaimNumber 
	,g.[IncomingClaimGetDate]
 ,g.[CourtWorkOverDate]
 ,g.[ClaimItem]
 ,g.[ClaimOrigin]
,g.[CourtWorkUnit]
	,g.[EmployeeName]
	,g.FLSimpleOrder
	,g.LinkLossNumber
	,FU.FU_ClaimedValuePeriod
	--,FU.FU_ClaimedValueChangeTypeByTime
	,FU.FU_ClaimedValueWithoutSD
	,FU.FU_ClaimedValueWithSD
	,FU.FU_RecoveredValuePeriod
	--,FU.FU_RecoveredValueChangeTypeByTime
	,FU.FU_RecoveredValueWithoutSD
	,FU.FU_RecoveredValueWithSD
	,FU.FU_Court
	,FU.FU_Judge
	,FU.FU_Applicant
	,FU.FU_ExpertOrg
	,FU.FU_Decision
	,Fir.FirstInst_ClaimedValuePeriod
--	,Fir.FirstInst_ClaimedValueChangeTypeByTime
	,Fir.FirstInst_ClaimedValueWithoutSD
	,Fir.FirstInst_ClaimedValueWithSD
	,Fir.FirstInst_RecoveredValuePeriod
	--,Fir.FirstInst_RecoveredValueChangeTypeByTime
	,Fir.FirstInst_RecoveredValueWithoutSD
	,Fir.FirstInst_RecoveredValueWithSD
	,Fir.FirstInst_Court
	,Fir.FirstInst_Judge
	,Fir.FirstInst_Applicant
	,Fir.FirstInst_ExpertOrg
	,Fir.FirstInst_Decision
	,sec.SecondInst_ClaimedValuePeriod
	--,sec.SecondInst_ClaimedValueChangeTypeByTime
	,sec.SecondInst_ClaimedValueWithoutSD
	,sec.SecondInst_ClaimedValueWithSD
	,sec.SecondInst_RecoveredValuePeriod
	--,sec.SecondInst_RecoveredValueChangeTypeByTime
	,sec.SecondInst_RecoveredValueWithoutSD
	,sec.SecondInst_RecoveredValueWithSD
	,sec.SecondInst_Court
	,sec.SecondInst_Judge
	,sec.SecondInst_Applicant
	,sec.SecondInst_ExpertOrg
	,sec.SecondInst_Decision
	,th.ThirdInst_ClaimedValuePeriod
	--,th.ThirdInst_ClaimedValueChangeTypeByTime
	,th.ThirdInst_ClaimedValueWithoutSD
	,th.ThirdInst_ClaimedValueWithSD
	,th.ThirdInst_RecoveredValuePeriod
	--,th.ThirdInst_RecoveredValueChangeTypeByTime
	,th.ThirdInst_RecoveredValueWithoutSD
	,th.ThirdInst_RecoveredValueWithSD
	,th.ThirdInst_Court
	,th.ThirdInst_Judge
	,th.ThirdInst_Applicant
	,th.ThirdInst_ExpertOrg
	,th.ThirdInst_Decision
	,Four.FourthInst_ClaimedValuePeriod
	--,Four.FourthInst_ClaimedValueChangeTypeByTime
	,Four.FourthInst_ClaimedValueWithoutSD
	,Four.FourthInst_ClaimedValueWithSD
	,Four.FourthInst_RecoveredValuePeriod
	--,Four.FourthInst_RecoveredValueChangeTypeByTime
	,Four.FourthInst_RecoveredValueWithoutSD
	,Four.FourthInst_RecoveredValueWithSD
	,Four.FourthInst_Court
	,Four.FourthInst_Judge
	,Four.FourthInst_Applicant
	,Four.FourthInst_ExpertOrg
	,Four.FourthInst_Decision
	,fif.FifthInst_ClaimedValuePeriod
	--,fif.FifthInst_ClaimedValueChangeTypeByTime
	,fif.FifthInst_ClaimedValueWithoutSD
	,fif.FifthInst_ClaimedValueWithSD
	,fif.FifthInst_RecoveredValuePeriod
	--,fif.FifthInst_RecoveredValueChangeTypeByTime
	,fif.FifthInst_RecoveredValueWithoutSD
	,fif.FifthInst_RecoveredValueWithSD
	,fif.FifthInst_Court
	,fif.FifthInst_Judge
	,fif.FifthInst_Applicant
	,fif.FifthInst_ExpertOrg
	,fif.FifthInst_Decision
	--,ROW_NUMBER() over (partition by g.IncomingClaimNumber order by g.IncomingClaimNumber) rn
	,docclaim._Fld13252 ContinuingPenalty
into #itog
From #general g 
left join #FUValue FU on g.IncomingClaimID = fu.IncomingClaimID
left join #FirstInstValue fir on g.IncomingClaimID = fir.IncomingClaimID
left join #SecondInstValue sec on g.IncomingClaimID = sec.IncomingClaimID
left join #ThirdInstValue th on g.IncomingClaimID = th.IncomingClaimID
left join #FourthInstValue four on g.IncomingClaimID = four.IncomingClaimID
left join #FifthInstValue fif on g.IncomingClaimID = fif.IncomingClaimID --(466909 rows affected) Completion time: 2023-06-26T10:08:12.2762470+03:00
------------------Zulpikarova 31-07-23
left join oisuu81.dbo._Document5472 docclaim on docclaim._Number = g.IncomingClaimNumber 
----------------------------------------------------------------
Drop table if exists Datamart.oisuu81_t_IncomingClaimNewLogic

Select * into Datamart.oisuu81_t_IncomingClaimNewLogic from #itog

Drop table if exists Datamart.oisuu81_t_IncomingClaimNewLogicByInst

SelectIncomingClaimID
	,InstByOisuu
	,IncomingClaimNumber
	,ClaimedValuePeriod
	,ClaimedMainDebt
	,ClaimedPlaintiffExamination
	,ClaimedCourtExamination
	,ClaimedRepresentativeExpenses
	,ClaimedPercentForUses
	,ClaimedPenaltyFee
	,ClaimedFine
	,ClaimedMoralDamage
	,ClaimedOtherExpenses
	,ClaimedStateDuty
	,ClaimedLossCommodyValue
	,ClaimedWearout
	,ClaimedValueWithoutSD
	,ClaimedValueWithSD
	,ClaimedAmountLoss
	,ClaimedCourtExaminationBeforeResolve
	,RecoveredValuePeriod
	,RecoveredMainDebt
	,RecoveredPlaintiffExamination
	,RecoveredCourtExamination
	,RecoveredRepresentativeExpenses
	,RecoveredPercentForUses
	,RecoveredPenaltyFee
	,RecoveredFine
	,RecoveredMoralDamage
	,RecoveredOtherExpenses
	,RecoveredStateDuty
	,RecoveredLossCommodyValue
	,RecoveredWearout
	,RecoveredValueWithoutSD
	,RecoveredValueWithSD
	,RecoveredAmountLoss
	,RecoveredCourtExaminationBeforeResolve
	,Instance
	,instID
	,Court
	,Judge
	,Applicant
	,ExpertOrg
	,Decision
	,[IncomingClaimGetDate]
 ,[CourtWorkOverDate]
 ,[ClaimItem]
 ,[ClaimOrigin]
 ,[CourtWorkUnit]
	,[EmployeeName]
	,FLSimpleOrder
	,LinkLossNumber
into Datamart.oisuu81_t_IncomingClaimNewLogicByInst from #vtPreItog

Drop table if exists dbo.oisuu81_t_CourtClaimNumber

Select IncomingClaimID, t1.instid, t2._Fld5545 CourtClaimNum, _Period, ROW_NUMBER() over (partition by IncomingClaimID, InstByOISUU order by _Period) RnCCN, InstByOISUU
into dbo.oisuu81_t_CourtClaimNumber
from [OISUU_report].[Datamart].[oisuu81_t_IncomingClaimNewLogicByInst] t1 inner join oisuu81.dbo._InfoRg5518 t2 on t1.IncomingClaimID = t2._Fld5519_RRRef and t1.instid = t2._Fld5520RRef


exec [dbo].[oisuu81_uspUpdateIncomingClaimsNewLogic_forAA]



GO