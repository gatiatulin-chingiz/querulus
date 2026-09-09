USE [OISUU_report]
GO
/****** Object:  StoredProcedure [dbo].[oisuu81_uspUpdate_ВитринаСутяжность]    Script Date: 07.09.2026 15:46:01 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO




---exec [dbo].[oisuu81_uspUpdate_ВитринаСутяжность]

ALTER         procedure [dbo].[oisuu81_uspUpdate_ВитринаСутяжность] as ---OISKBN-6022_Сутяжность

----Претензия в Инциденте-----

drop table if exists #Pret  
SELECT distinct L.LossID,L.LossNumber
  ,case when P.PretensionNumber is null then 0 else 1 end As Претензия
  ,IncidentCharacteristics.IncidentID
  ,IncidentCharacteristics.IncidentNumber As Инцидент
  ,ROW_NUMBER() over (partition by L.LossNumber order by P.PretensionNumber) rn
into #Pret
FROM oisuu81_t_Pretensions As P
join oisuu81_t_Losses As L
on P.LossID = L.LossID
left join dbo.oisuu81_t_IncidentCharacteristics as IncidentCharacteristics
  on IncidentCharacteristics.LossID = P.LossID
where IsMarked = 0x00

drop table if exists #Pretensions
select LossID,LossNumber,Инцидент,max(rn) as Претензия 
into #Pretensions
from #Pret
group by LossID,LossNumber,Инцидент--,Претензия


drop table if exists #PretensionsИнцидент
select distinct Инцидент,max(rn) as Претензия
into #PretensionsИнцидент
from #Pret
group by Инцидент

------ Тип претензии по претензионного убытку--------03/06/2026 Болотина OISKBN-6181
drop table if exists #PretensionTypes
select distinct l.LossNumber,PretensionTypes
into #PretensionTypes
from oisuu81_t_Losses as l  
join oisuu81_t_Pretensions as P on P.LinkedLossID = l.LossID
where 1=1
and l.PaymentOrderDateTime between '2026-04-20 00:00:00' and GETDATE()
and l.LossProcess in ('Прямое ОСАГО (с 1 марта 2009)','Традиционное ОСАГО')
and l.Risk = 'Ущерб имуществу третьих лиц'
and RefundFormDetailed in ('Претензия','Соглашение. Претензия')
and  l.Filial in ('Архангельский','Владимирский','Кемеровский','Курский','Магнитогорский','Марийский','Мурманский'
,'Омский','Пермский','Уфимский','Петропавловск-Камчатский','Ярославский')
order by l.LossNumber



drop table if exists #Сутяжность 
select distinct _Fld11689_RRRef as LossID--,  _Fld13228RRef as Модель ,_Description
,_Period as [Дата вызова модели сутяжности]
,Case when _Fld11690 = 0 then 0 
      when _Fld11690 > 0 then _Fld11690 
	  when _Fld11690 < 0 then 0 end as [Сумма рекомендованная к доплате по модели]
,Case when _Fld11690 = 0 then 0 when _Fld11690 > 0 then 1 when _Fld11690 < 0 then -100 end as РезультатПроверки
into #Сутяжность
FROM [oisuu81].[dbo].[_InfoRg11688] m
left join  [oisuu81].[dbo]._Reference11793 s 
on m._Fld13228RRef = s._IDRRef
where 1 =1 and _Fld13228RRef =0xA50C005056A8E5E511F13448E7333739 -- 'Сутяжность'

drop table if exists #РасчетВыплаты
select _Fld18874RRef as LossID
,_Date_Time as ДатаДокумента
,_Fld18875 as СуммаВыплаты
,_Fld18876 as СуммаОсновногоДолгаКВыплате
,_Fld18877 as СуммаОсновногоДолгаЗаявлено--ОбщаяСуммаУщербаПоЕМР
,_Fld18888 as СуммаЗаявлено
,_Fld18889 as СуммаКВыплате
into #РасчетВыплаты
from [oisuu81].[dbo].[_Document18774] D--РасчетВыплаты
left join [oisuu81].[dbo].[_Document18774_VT18885] DV on D._IDRRef = DV._Document18774_IDRRef-- Документ.РасчетВыплаты.ВыплатыДополнительные
where 1 = 1
and _Marked = 0x00
and _Fld18887RRef=0xA507005056A8E5E511F118D754B5E4BD --Категория 'Иные затраты'

drop table if exists #IncidentСутяжность
select distinct I.IncidentID
,max(case when SUT.LossID is not null and RV.СуммаКВыплате > 0 then 1 
      else 0 end) as [Выплата по модели]
into #IncidentСутяжность
from oisuu81_t_IncidentToLoss as I
left join oisuu81_t_Losses as l on l.LossID  = I.LossID
left join #Сутяжность as SUT on SUT.LossID = I.LossID
left join #РасчетВыплаты as RV on RV.LossID = I.LossID
where l.InsuranceTypeName in ('ОСАГО ФЛ')
and l.LossProcess in ('Прямое ОСАГО (с 1 марта 2009)','Традиционное ОСАГО')
and l.Risk = 'Ущерб имуществу третьих лиц'
and Year(l.EventDate)>=2020 
group by I.IncidentID

--- РегистрСведений.СуммыКалькуляцийAudanet

drop table if exists #СуммыКалькуляцийAudanet
select _Fld14747RRef as LossID
,_Fld15038 as ДатаРасчета
,_Fld14748 as СуммаРемонтаБезУчётаИзноса
,_Fld14754 as СтоимостьСУчетомИзноса
,_Period
,ROW_NUMBER() over (partition by _Fld14747RRef order by _Period asc) rn
into #СуммыКалькуляцийAudanet
from [oisuu81].[dbo].[_InfoRg14746]

drop table if exists #СуммыКалькуляций
select * into #СуммыКалькуляций from #СуммыКалькуляцийAudanet where rn = 1

drop table if exists #ИсторияМинимизации
select _Fld11960RRef as LossID,_Fld11964 as ИтоговыйПроцентМинимизации,_Period
,ROW_NUMBER() over (partition by _Fld11960RRef order by _Period desc) rn
into #ИсторияМинимизации
from [oisuu81].[dbo].[_InfoRg11959]
drop table if exists #ИсторияМинимизацииИтог
select * into #ИсторияМинимизацииИтог from #ИсторияМинимизации where rn = 1

Drop table if exists [OISUU_report].[dbo].[ВитринаСутяжность]

select l.LossNumber as Убыток 
,INC.IncidentNumber as НомерИнцидент
,l.LossUnitZone as ЗонаУрегулирования
,l.Filial as Филиал
,cast(l.PaymentOrderDateTime as date) as ДатаЗаявления
,cast(l.EventDate as date) as ДатаСобытия
,cast(l.IssueDate as date) as ДатаУрегулирования
,l.LossProcess as Процесс
,PO.ApplicantType as [Тип заявителя]
,DATEDIFF(year, cast(Per.PersonBirthDate as date), cast(l.PaymentOrderDateTime as Date)) as ВозрастЗаявителя
,PO.ApplicationGetMethod as СпособПолученияЗаявления
,case when l.EventCreatedByGIBDDFlag = 0x00 then 0 else 1 end as ОформленоГИБДД
,case when VictimObjectType = 'Автотранспорт' then 1 else 0 end as [ТипОбъектаАвтотранспорт]---03/06/2026 Болотина OISKBN-6181
,VictimVehicleTypeByClassificator as [Категория ТС потерпевшего]
,VictimVehicleOwnerType as [Тип владельца транспортного средства] ---03/06/2026 Болотина OISKBN-6181
,VictimVehicleAge as [Возраст ТС потерпевшего]
,case when l.OffsDate is not null then 'Списано' else l.RefundForm end as ФормаВозмещения--,l.RefundFormDetailed as ФормаВозмещенияДетализированная ---03/06/2026 Болотина OISKBN-6181
,case when l.OffsReasonName = 'Не предоставление ТС на осмотр' then 1 else 0 end as [УбытокСписан_НеПредоставлениеТСнаОсмотр]---03/06/2026 Болотина OISKBN-6181
,case when SUT.LossID is not null then 1 else 0 end as ВызовМодельСутяжность
,SUT.[Дата вызова модели сутяжности]
,SUT.РезультатПроверки
,SUT.[Сумма рекомендованная к доплате по модели]
,RV.СуммаОсновногоДолгаЗаявлено
,RV.СуммаОсновногоДолгаКВыплате
,СуммыКалькуляций.СуммаРемонтаБезУчётаИзноса
,СуммыКалькуляций.СтоимостьСУчетомИзноса
,isnull(RV.СуммаКВыплате,0.00) as [Иные затраты]--[Cумма к доплате по модели] --13.05.2026
,isnull(RV.СуммаВыплаты,0.00) as СуммаКВыплате
,ИсторияМинимизацииИтог.ИтоговыйПроцентМинимизации
,PO.ValueZU as СуммаЗУ
,isnull(Pay.PaymentValue,0.00) as СуммаПлатежа--фактическая сумма выплаты по убытку
,case when l.RefundForm = 'Соглашение' then 1 else 0 end as [Заключено соглашение]
,case when SUT.LossID is not null and RV.СуммаКВыплате > 0 then 1 
      else 0 end as [Выплата по модели]
,case when Pret.Претензия is null then 0 else 1 end as Претензия
,isnull(case when l.RefundFormDetailed in ('Претензия','Соглашение. Претензия') then Pay.PaymentValue else 0 end,0.00) as [Cумма выплаты по претензии]
,isnull(Pret.Претензия,0) as КолВоПретензий
,isnull(case when PretensionsИнцидент.Претензия is not null then 1 else 0 end,0) as ЕстьПретензияВИнциденте
,isnull(case when l.RefundFormDetailed in ('Судебная ФУ') then 1 else 0 end,0) as [Обращение к ФУ]
,isnull(case when l.RefundFormDetailed in ('Судебная ФУ') then Pay.PaymentValue else 0 end,0.00) as [Cумма выплат по ФУ]
,isnull(case when l.RefundFormDetailed in ('Судебная ИСК') then 1 else 0 end,0) as [Обращение к суду]
,isnull(case when l.RefundFormDetailed in ('Судебная ИСК') then Pay.PaymentValue else 0 end,0.00) as [Cумма выплаты по суду]
,case when l.RefundFormDetailed in ('Судебная ФУ','Судебная ИСК') then 'Судебный'
when l.RefundFormDetailed in ('Претензия','Соглашение. Претензия') then 'Претензионный'
      else 'Первичный'  end as УбытокСтатус
,PT.PretensionTypes as ТипПретензии --03/06/2026 Болотина OISKBN-6181
,IncidentСутяжность.[Выплата по модели] as [Выплата по модели в Инциденте]
	,PC.Main_DeclareAmount as ОсновныеВыплатыСуммаОсновногоДолгаЗаявлено                                                               
	,PC.Main_PaymentAmount as ОсновныеВыплатыСуммаОсновногоДолгаКВыплате
	,PC.Declare_Main_MainDebtAmount as [ОсновныеВыплатыЗаявлено_Основной долг]
	,PC.Declare_Main_UTSAmount as [ОсновныеВыплатыЗаявлено_Утрата товарной стоимости] 
	,PC.Declare_Main_TSEvacuationCostsAmount as [ОсновныеВыплатыЗаявлено_Затраты на эвакуацию ТС] 
	,PC.Declare_Main_TSStorageCostsAmount as [ОсновныеВыплатыЗаявлено_Затраты на хранение ТС] 
	,PC.Declare_Main_MailCostsAmount as [ОсновныеВыплатыЗаявлено_Затраты на почту]
	,PC.Declare_Main_NotatyFeesAmount as [ОсновныеВыплатыЗаявлено_Затраты на нотариуса] 
	,PC.Declare_Main_ReviewCostsAmount as [ОсновныеВыплатыЗаявлено_Затраты на независимую экспертизу] 
	,PC.Payment_Main_MainDebtAmount as [ОсновныеВыплатыКВыплате_Основной долг] 
	,PC.Payment_Main_UTSAmount as [ОсновныеВыплатыКВыплате_Утрата товарной стоимости] 
	,PC.Payment_Main_TSEvacuationCostsAmount as [ОсновныеВыплатыКВыплате_Затраты на эвакуацию ТС] 
	,PC.Payment_Main_TSStorageCostsAmount as [ОсновныеВыплатыКВыплате_Затраты на хранение ТС] 
	,PC.Payment_Main_MailCostsAmount as [ОсновныеВыплатыКВыплате_Затраты на почту]
	,PC.Payment_Main_NotatyFeesAmount as [ОсновныеВыплатыКВыплате_Затраты на нотариуса]
	,PC.Payment_Main_ReviewCostsAmount as [ОсновныеВыплатыКВыплате_Затраты на независимую экспертизу]
    ,PC.Declare_ADD_WearAmount as ДополнительныеВыплатыЗаявлено_Износ
    ,PC.Declare_ADD_CostsEmergencyCommAmount as [ДополнительныеВыплатыЗаявлено_Затраты на аваркома]
	,PC.Declare_ADD_DefectCostsAmount as [ДополнительныеВыплатыЗаявлено_Затраты на дефектовку ТС]
	,PC.Declare_ADD_DamegesAmount as [ДополнительныеВыплатыЗаявлено_Заявленная сумма убытка]
	,PC.Declare_ADD_OtherCostsAmount as [ДополнительныеВыплатыЗаявлено_Иные затраты]
	,PC.Declare_ADD_PenaltyFeeAmount as [ДополнительныеВыплатыЗаявлено_Неустойка/пеня]
	,PC.Payment_ADD_WearAmount [ДополнительныеВыплатыКВыплате_Износ]
	,PC.Payment_ADD_CostsEmergencyCommAmount [ДополнительныеВыплатыКВыплате_Затраты на аваркома] 
	,PC.Payment_ADD_DefectCostsAmount [ДополнительныеВыплатыКВыплате_Затраты на дефектовку ТС]
	,PC.Payment_ADD_DamegesAmount [ДополнительныеВыплатыКВыплате_Сумма к выплате]  
	,PC.Payment_ADD_OtherCostsAmount [ДополнительныеВыплатыКВыплате_Иные затраты]
	,PC.Payment_ADD_PenaltyFeeAmount [ДополнительныеВыплатыКВыплате_Неустойка/пеня]
	,(PC.Payment_Main_UTSAmount+	PC.Payment_Main_TSEvacuationCostsAmount+PC.Payment_Main_TSStorageCostsAmount+PC.Payment_Main_MailCostsAmount +PC.Payment_Main_NotatyFeesAmount +
	PC.Payment_Main_ReviewCostsAmount) ОбщаяСуммаОсновнойВыплатыДоп
	,(PC.Payment_ADD_WearAmount +PC.Payment_ADD_CostsEmergencyCommAmount +PC.Payment_ADD_DefectCostsAmount +PC.Payment_ADD_DamegesAmount + 
	PC.Payment_ADD_OtherCostsAmount + PC.Payment_ADD_PenaltyFeeAmount) ОбщаяСуммаДополнительныхВыплат
into [OISUU_report].[dbo].[ВитринаСутяжность]
from oisuu81_t_Losses as l
left join #Сутяжность as SUT on SUT.LossID = l.LossID
left join #РасчетВыплаты as RV on RV.LossID = l.LossID
left join oisuu81_t_PaymentOrders as PO on PO.LossID = l.LossID
left join (select LossID,sum(PaymentValue) as PaymentValue from oisuu81_t_Payments --where ReasonForPayment !='ВПРС' 
group by LossID) as Pay on Pay.LossID = l.LossID
left join #Pretensions as Pret on Pret.LossID = l.LossID
left join oisuu81_t_IncidentToLoss as INC on INC.LossID = l.LossID
left join #PretensionsИнцидент as PretensionsИнцидент on PretensionsИнцидент.Инцидент = INC.IncidentNumber
left join oisuu81_t_Persons as PeronPer.PersonID=PO.Applicant
left join #IncidentСутяжность as IncidentСутяжность on IncidentСутяжность.IncidentID = INC.IncidentID
left join #СуммыКалькуляций as СуммыКалькуляций on СуммыКалькуляций.LossID = l.LossID
left join #ИсторияМинимизацииИтог as ИсторияМинимизацииИтог on ИсторияМинимизацииИтог.LossID = l.LossID
left join dbo.oisuu81_t_PaymentCalc as PC on PC.LossID = l.LossID
left join #PretensionTypes as PT on PT.LossNumber = l.LossNumber --03/06/2026 Болотина OISKBN-6181
where 1=1
and l.PaymentOrderDateTime between '2026-04-20 00:00:00' and GETDATE()
and l.LossProcess in ('Прямое ОСАГО (с 1 марта 2009)','Традиционное ОСАГО')
and l.Risk = 'Ущерб имуществу третьих лиц'
and  l.Filial in ('Архангельский','Владимирский','Кемеровский','Курский','Магнитогорский','Марийский','Мурманский'
,'Омский','Пермский','Уфимский','Петропавловск-Камчатский','Ярославский')