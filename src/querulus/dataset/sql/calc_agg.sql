with tmp as (
	SELECT
		_Period         Период	,
		itl.IncidentNumber AS INCIDENT_NUMBER,
		l.LossNumber 	AS LOSS_NUMBER	,
		_Fld14748	СуммаРемонтаБезУчётаИзноса	,
		_Fld14749	СтоимостьЗапчастей	,
		_Fld14750	Работы	,
		_Fld14751	Материалы	,
		_Fld14752	ДеталиРазовогоМонтажа	,
		_Fld14753	ПроцентИзноса	,
		_Fld14754	СтоимостьСУчетомИзноса	,
		_Fld16561	Пробег	,
		_Fld16562	ДатаНачалаЭксплуатации	,
		_Fld14787	НомерРасчета	,
		cast(_Fld14788 as INT)	СканыКалькуляцииОбработаны	,
		_Fld15038	ДатаРасчета,
		ROW_NUMBER() over (
            partition by itl.IncidentNumber
            order by l.LossNumber desc, _Period desc
        ) as rn
	from oisuu81.dbo._InfoRg14746 i
	left join oisuu81_t_losses l on l.LossID = _Fld14747RRef
	LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl on l.LossID=itl.LossID
	where year(l.IssueDate) is not null
	and l.LossProcess in ('Прямое ОСАГО (с 1 марта 2009)','Традиционное ОСАГО')
	and Risk = 'Ущерб имуществу третьих лиц'
)
SELECT *
FROM tmp
WHERE rn = 1
