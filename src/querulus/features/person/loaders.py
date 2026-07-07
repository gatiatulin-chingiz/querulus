"""Loaders for pretensions and court history.

Цель: использовать existing SQL (как в legacy steps) без включения include_enrich.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from querulus.dataset.io import LazyOisuuConnection, load_sql_artifact
from querulus.dataset.paths import DataPaths
from querulus.dataset.pretension_utils import dedupe_pretension_rows


def _require_conn(conn: LazyOisuuConnection | None, use_sql: bool) -> LazyOisuuConnection:
    if use_sql and conn is None:
        raise ValueError("conn обязателен при use_sql=True")
    return conn or LazyOisuuConnection()


def load_pretensions_base(
    paths: DataPaths,
    conn: LazyOisuuConnection | None,
    *,
    use_sql: bool,
    save_checkpoint: bool,
) -> pd.DataFrame:
    """Загрузить претензии с INCIDENT_NUMBER (без enrich/cumsum)."""
    query = """
    SELECT *
      FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS P
      LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS ITL ON ITL.LossID=P.LossID
    """
    _conn = _require_conn(conn, use_sql)
    df = load_sql_artifact(
        paths,
        _conn,
        paths.raw_dir,
        "df_pretensions.parquet",
        query,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )
    df.columns = df.columns.str.upper()
    return dedupe_pretension_rows(df)


def load_pretensions_penalty_surcharge(
    paths: DataPaths,
    conn: LazyOisuuConnection | None,
    *,
    use_sql: bool,
    save_checkpoint: bool,
) -> pd.DataFrame:
    """Загрузить доплаты/неустойку по претензиям (df_pretensions_3.parquet)."""
    # Запрос взят из dataset/steps/pretensions.py (без изменений).
    query = """
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
     FROM tmp
     WHERE rn = 1
     union select * from penalty where rn = 1
    """
    _conn = _require_conn(conn, use_sql)
    df = load_sql_artifact(
        paths,
        _conn,
        paths.raw_dir,
        "df_pretensions_3.parquet",
        query,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        sql_reader=pd.read_sql_query,
    )
    df.columns = df.columns.str.upper()
    return df


def load_claims_persons(
    paths: DataPaths,
    conn: LazyOisuuConnection | None,
    *,
    use_sql: bool,
    save_checkpoint: bool,
) -> pd.DataFrame:
    """Загрузить истцов (person_id -> INCOMING_CLAIM_NUMBER)."""
    query = """
    SELECT 
        Лицо,
        CAST ( Представитель as int ) Представитель,
        CAST ( Цессионарий as int ) Цессионарий,
        ПолноеФИОЛица,
        ТипЛица,
        ПолЛица,
        НомерИск,
        ОбращениеКФУОтЗаявителяПоступилоПосредством
    FROM [OISUU_report].[Datamart].[oisuu81_t_Истцы]
    WHERE ПометкаУдаленияИск=0x00
    """
    _conn = _require_conn(conn, use_sql)
    df = load_sql_artifact(
        paths,
        _conn,
        paths.raw_dir,
        "df_claims_persons.parquet",
        query,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        sql_reader=pd.read_sql_query,
    )
    return df


def load_claims_incoming(
    paths: DataPaths,
    conn: LazyOisuuConnection | None,
    *,
    use_sql: bool,
    save_checkpoint: bool,
    claims_where_sql: str,
) -> pd.DataFrame:
    """Загрузить суды (incoming claim) с INCIDENT_NUMBER и claimed/recovered колонками."""
    query = f"""
    WITH doc as (
        SELECT [_Document5472_IDRRef]
              ,[_KeyField]
              ,[_LineNo14756]
              ,[_Fld14757]
              ,[_Fld14758]
              ,[_Fld14759]
              ,[_Fld14760]
              ,[_Fld14761]
              ,[_Fld14762RRef]
              ,[_Fld14763]
              ,ROW_NUMBER() OVER (PARTITION BY [_Document5472_IDRRef] ORDER BY [_LineNo14756] DESC) as rn
          FROM [oisuu81].[dbo].[_Document5472_VT14755]
    ),
    cessia as (
        SELECT 
                НомерИск
                ,max([Представитель]) as [Представитель]
                ,max([Цессионарий]) as [Цессионарий]
          FROM [OISUU_report].[Datamart].[oisuu81_t_Истцы]
          group by НомерИск
    )
    SELECT itl.[LossID]
          ,itl.[LossNumber]
          ,itl.[IncidentID]
          ,itl.[IncidentNumber]
          ,icnl.[IncomingClaimID]
          ,icnl.[InstByOisuu]
          ,icnl.[IncomingClaimNumber]
          ,icnl.[ClaimedValuePeriod]
          ,icnl.[ClaimedMainDebt]
          ,icnl.[ClaimedPlaintiffExamination]
          ,icnl.[ClaimedCourtExamination]
          ,icnl.[ClaimedRepresentativeExpenses]
          ,icnl.[ClaimedPercentForUses]
          ,icnl.[ClaimedPenaltyFee]
          ,icnl.[ClaimedFine]
          ,icnl.[ClaimedMoralDamage]
          ,icnl.[ClaimedOtherExpenses]
          ,icnl.[ClaimedStateDuty]
          ,icnl.[ClaimedLossCommodyValue]
          ,icnl.[ClaimedWearout]
          ,icnl.[ClaimedValueWithoutSD]
          ,icnl.[ClaimedValueWithSD]
          ,icnl.[ClaimedAmountLoss]
          ,icnl.[ClaimedCourtExaminationBeforeResolve]
          ,icnl.[RecoveredValuePeriod]
          ,icnl.[RecoveredMainDebt]
          ,icnl.[RecoveredPlaintiffExamination]
          ,icnl.[RecoveredCourtExamination]
          ,icnl.[RecoveredRepresentativeExpenses]
          ,icnl.[RecoveredPercentForUses]
          ,icnl.[RecoveredPenaltyFee]
          ,icnl.[RecoveredFine]
          ,icnl.[RecoveredMoralDamage]
          ,icnl.[RecoveredOtherExpenses]
          ,icnl.[RecoveredStateDuty]
          ,icnl.[RecoveredLossCommodyValue]
          ,icnl.[RecoveredWearout]
          ,icnl.[RecoveredValueWithoutSD]
          ,icnl.[RecoveredValueWithSD]
          ,icnl.[RecoveredAmountLoss]
          ,icnl.[RecoveredCourtExaminationBeforeResolve]
          ,icnl.[Instance]
          ,icnl.[instID]
          ,icnl.[Court]
          ,icnl.[Judge]
          ,icnl.[Applicant]
          ,icnl.[ExpertOrg]
          ,icnl.[Decision]
          ,icnl.[IncomingClaimGetDate]
          ,icnl.[CourtWorkOverDate]
          ,icnl.[ClaimItem]
          ,icnl.[ClaimOrigin]
          ,icnl.[CourtWorkUnit]
          ,icnl.[EmployeeName]
          ,icnl.[FLSimpleOrder]
          ,icnl.[LinkLossNumber]
          ,docs.[_Fld14757] as [Payment_fee_fu]
          ,cs.[Представитель]
          ,cs.[Цессионарий]
    FROM [OISUU_report].[Datamart].[oisuu81_t_IncomingClaimNewLogicByInst] as icnl
    LEFT JOIN cessia as cs on cs.НомерИск=icnl.IncomingClaimNumber
    LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl on itl.LossNumber=icnl.LinkLossNumber
    LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Losses] as l on l.LossNumber=itl.LossNumber
    LEFT JOIN doc as docs on docs.[_Document5472_IDRRef]=icnl.[IncomingClaimID]
    WHERE (docs.rn = 1 or docs.rn is null)
        AND {claims_where_sql}
    """
    _conn = _require_conn(conn, use_sql)
    df = load_sql_artifact(
        paths,
        _conn,
        paths.raw_dir,
        "df_claims_incoming.parquet",
        query,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        sql_reader=pd.read_sql_query,
    )
    df.columns = df.columns.str.upper()
    return df


def normalize_hex_person_id(value: Any) -> str | None:
    """Привести person_id из SQL к строке hex upper (как в legacy)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex().upper()
    raw = str(value).strip()
    return raw.upper() if raw else None


def normalize_person_id_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_hex_person_id)

