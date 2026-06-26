%load_ext autoreload
%autoreload 2

from AutoMVP import *

import pandas as pd
# import seaborn as sns
import numpy as np
# import pyodbc 
import pickle

from sklearn.model_selection import train_test_split
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import json
import itertools
import sys
from copy import deepcopy
import traceback
# import matplotlib.pyplot as plt
from itertools import cycle, combinations
import re
from tqdm import tqdm
from tqdm import tqdm_notebook
tqdm.pandas()
# from pyspark.sql import SparkSession
# from pyspark.sql.types import TimestampType
# from pyspark.sql.functions import date_format
import math 
import gc
sys.path.append("/home/jovyan/share/1libs/Functions/") # go to parent dir
from oskar_functions import *
import subprocess
import logging
import shutil
from pyarrow.parquet import ParquetFile
import pyarrow as pa
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from itertools import chain


sys.path.append("/home/jovyan/old_home")

from modeldiagnostics.src.tuning import TuningHyperparameters
from modeldiagnostics.src.categorical_features_processor import CategoricalFeatureProcessor
from modeldiagnostics.src.modeldiagnostics import ModelDiagnostics

import pymssql
conn_oisuu = pymssql.connect(server='', user='', password='', database='', port=)


# function vizualisation

expsoure = 'expos'
damage_count = 'TARGET_2'
damage_sum =  'TARGET_3_SEV'

       
def plot_cat_vs_target(data, x_min, x_max, figsize, feature, save, model_type, rotation):
    # Функция нижнего уровня для двух функций ниже
    if x_min:
        data = data[data['ratio'] > x_min]
    if x_max:
        data = data[data['ratio'] < x_max]
    n = data.shape[0]
    ind = np.arange(n)
    
    fig, ax = plt.subplots(dpi=100, figsize=figsize)

    ax.bar(ind, data[expsoure])
    # ax.set_yticks(fontsize=20)
    # ax.ylabel()
    ax.set_ylabel(expsoure, fontsize=20)
    ax.set_xticks(ind, data.index.tolist(), fontsize=20, rotation=rotation) #  rotation='vertical'
    
    ax.tick_params(axis='both', labelsize=20)
    
    axes2 = ax.twinx()
    axes2.plot(ind, data['ratio'], color='r', marker='o')
    
    if model_type == 'frequency':
        axes2.set_ylabel('Частота', fontsize=20)
    elif model_type == 'severity':
        axes2.set_ylabel('Severity', fontsize=20)
        
    axes2.tick_params(axis='both', labelsize=20)
    plt.grid(False)
    plt.title(f"""{feature}_{str(model_type).upper()}""", fontsize=25)
    
    if save:
        plt.savefig(f"""plots/{feature}_{str(model_type).upper()}.png""", bbox_inches='tight', dpi=1200)
    plt.show()

def research_continous(data, feature, quantiles, model_type='frequency', figsize:tuple=(55, 10), save=False, rotation=90):
    if model_type == 'frequency':
        data = data[[feature, expsoure, damage_count]]
        quantiles, bins = pd.qcut(data[feature], quantiles, duplicates='drop', retbins=True)
        data.drop(feature, axis=1, inplace=True)
        data = pd.concat([data, quantiles], axis=1, join='outer')
        grouped = data.groupby(feature).agg(sum)
        grouped['ratio'] = grouped[damage_count] / grouped[expsoure]
    elif model_type == 'severity':
        data = data[[feature, damage_sum, expsoure, damage_count]]
        quantiles, bins = pd.qcut(data[feature], quantiles, duplicates='drop', retbins=True)
        data.drop(feature, axis=1, inplace=True)
        data = pd.concat([data, quantiles], axis=1, join='outer')
        grouped = data.groupby(feature).agg(sum)
        grouped['ratio'] = grouped[damage_sum] / grouped[damage_count]
        
    # grouped[expsoure] = grouped[expsoure] / sum(grouped[expsoure])
    grouped[damage_count] = grouped[damage_count] / sum(grouped[damage_count])

    plot_cat_vs_target(grouped, None, None, figsize, feature, save,  model_type, rotation)
    return grouped
    
def research_feature(data, feature, bounds=None, sort_by=None, x_min: float=None, x_max: float=None, 
                     figsize: tuple=(55, 10), max_limit=None, min_limit=None, model_type='frequency', 
                     save=False, rotation=90):

    if model_type == 'frequency':
        data = data[[feature, expsoure, damage_count]]
        grouped = data.groupby(feature, dropna=False).agg(sum)
    
        grouped['ratio'] = grouped[damage_count] / grouped[expsoure]
    elif model_type == 'severity':
        data = data[[feature, expsoure, damage_count, damage_sum]]
        grouped = data.groupby(feature, dropna=False).agg(sum)
        
        grouped['ratio'] = grouped[damage_sum] / grouped[damage_count]
        # grouped['ratio'] = grouped['ratio'] / sum(grouped['ratio'])
        
    # grouped[expsoure] = grouped[expsoure] / sum(grouped[expsoure])
    # grouped[damage_count] = grouped[damage_count] / sum(grouped[damage_count])
    # list(quarter.index)[0].split()[1:]
    if sort_by == 'index':
        grouped = grouped.sort_index()
    elif sort_by == 'index_d':
        
        def quarter(q):
            res = []
            for quarter in q:
                quar, year = str(quarter).split()[1:]
                quar = quar[0]
                res.append(int(year + quar))
            return res
        
        grouped = grouped.sort_index(key=quarter)
        
    elif sort_by is None:
        grouped = grouped.sort_values(by='ratio', ascending=False)
    else:
        grouped = grouped.sort_values(by=sort_by, ascending=False)
        
    
    if max_limit is not None:
        grouped = grouped[grouped[expsoure] < max_limit]
        
    if min_limit is not None:
        grouped = grouped[grouped[expsoure] > min_limit]
        
        
    if bounds is not None:
        groups = concat_group(grouped[['ratio']], bounds)
        grouped = pd.concat([grouped, groups], axis=1)
    plot_cat_vs_target(grouped, x_min, x_max, figsize, feature, save, model_type, rotation)
    return grouped




def value_type(df: pd.DataFrame, isprint=True, count_numeric=100):
    """
    Функция для разделения признаков по количеству значений данных в них

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм, из которого будут получены данные
    isprint : str
        Флаг, отвечающий за то, будет ли выводиться строка после распределения по каждому признаку

    Returns
    -------
    (bin_list, cat_list, num_list, drop_list, obj_list): Cortage of 5 [list of str]
        (
        Список бинарных признаков (2 значения),
        Список категориальных признаков (от 3 до 20 уникальных значений в столбцах),
        Список числовых признаков (всё, что не object с большим количеством значений),
        Список признаков на удаление (1 значение), 
        Список признаков типа object (обязательны к рассмотрению),
        ) 

    Examples
    --------
    >>> (bin_list, cat_list, num_list, drop_list, obj_list) = value_type(df, isprint=False)
        BINARY: ['EventCreatedByGIBDDFlag', 'E-Garant', <...> ]
        CATEGORIAL: ['CustomerImportance', 'DTPOSAGOType', <...>]
        NUMERIC: ['LossNumber', 'InsuredSum', 'LossDateTime', <...>]
        TO_DROP: ['EventTypeDescription', 'InsuranceTypeName', <...>]
        OBJECT: ['ContractNumber', 'VictimContractNumber', <...>]
    """
    # Инициализация списков
    bin_list = []
    cat_list = []
    num_list = []
    drop_list = []
    date_list = []
    obj_list = []
    # Цикл по колонкам датафрейма
    for col in tqdm(df.columns):
        try:
            VC = df[col].nunique(dropna=False)
        except:
            print(col, ' не хэшируемый тип')
            continue
        # Если только 1 значение
        if VC ==1:
            if isprint:
                print('DROP:', col )
            drop_list.append(col)
        # Если только 2 значения
        if VC ==2:
            if isprint:
                print('binary:', col )
            bin_list.append(col)
        # Если значений в столбце от 3 до 100
        if 2 < VC <= count_numeric:
            if isprint:
                print('categorial:', col )
            cat_list.append(col)
    for col in tqdm(df.columns):
        #VC = df[col].value_counts(dropna=False)
        # Теперь рассмотрим колонки, которые не вошли в предыдущие списки
        if col not in bin_list and col not in cat_list:
            # Для строкового типа, например
            if df[col].dtype == object:
                if isprint:
                    print('object:', col )
                obj_list.append(col)
            elif df[col].dtype == '<M8[ns]':
                if isprint:
                    print('date:', col )
                date_list.append(col)
            # Для всего остального
            else:                
                if isprint:
                    print('numeric:', col )
                num_list.append(col)
    print("BINARY:", bin_list)
    print("CATEGORIAL:", cat_list)
    print("NUMERIC:", num_list)
    print("TO_DROP:", drop_list)
    print("OBJECT:", obj_list)
    print("DATE:", date_list)
    #return (bin_list, cat_list, num_list, drop_list, obj_list, date_list)
    return ({"BINARY": bin_list,
             "CATEGORIAL": cat_list,             
             "NUMERIC": num_list,
             "TO_DROP": drop_list,
             "OBJECT": obj_list, 
             "DATE": date_list
            })


def convert_to_pandas(spark_df):
    """
    This function will safely convert a spark DataFrame to pandas.
    """
    # Iterate over columns and convert each timestamp column to a string
    timestamp_cols = []
    decimal_cols = []
    for column in spark_df.schema:
        if isinstance(column.dataType, TimestampType):
            timestamp_cols.append(column.name)
            spark_df = spark_df.withColumn(
                column.name,
                date_format(column.name, "yyyy-MM-dd HH:mm:ss"))
        if isinstance(column.dataType, DecimalType):
            decimal_cols.append(column.name)
    # Convert to a pandas DataFrame and reset timestamp columns
    pandas_df = spark_df.toPandas()
    for column_header in timestamp_cols:
        pandas_df[column_header] =   pd.to_datetime(pandas_df[column_header], errors = 'coerce') # .astype("datetime64[ns]")
    for column_header in decimal_cols:
        pandas_df[column_header] =   pandas_df[column_header].astype(float) # .astype("datetime64[ns]")
    return pandas_df



import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def spark_to_pandas_parquet(
        spark,
        sql_query,
        source,
        to_df: str = "tmp_dataset.parquet"
):
    if os.path.exists(to_df):
        if os.path.isdir(to_df):
            shutil.rmtree(to_df)
        else:
            os.remove(to_df)

    if not sql_query:
        logger.info("Spark dataset loading...")
        df = spark.table(source)
    else:
        logger.info("Spark dataset loading using sql query...")
        df = spark.sql(sql_query)

    df.write.mode("overwrite").parquet(to_df)
    subprocess.run(f"hdfs dfs -get {to_df}", shell=True)

    return pd.read_parquet(to_df)



rename_dict = {'IncidentNumber': 'INCIDENT_NUMBER',
 'LossID': 'LOSS_ID',
 'LOSSNUMBER': 'LOSS_NUMBER',
 'PRETENSIONID': 'PRETENSION_ID',
 'PAYMENTNUMBER': 'PAYMENT_NUMBER',
 'ISMARKED': 'IS_MARKED',
 'PRETENSIONDATE': 'PRETENSION_DATE',
 'PRETENSIONNUMBER': 'PRETENSION_NUMBER',
 'ISOVER': 'IS_OVER',
 'PRETENSIONGETDATE': 'PRETENSION_GET_DATE',
 'APPLICANTPERSONID': 'APPLICANT_PERSON_ID',
 'PRETENSIONSTAGE': 'PRETENSION_STAGE',
 'PRETENSIONGETMETHOD': 'PRETENSION_GET_METHOD',
 'PRETENSIONVALUE': 'PRETENSION_VALUE',
 'LOSSUNIT': 'LOSS_UNIT',
 'LOSSUNITZONE': 'LOSS_UNIT_ZONE',
 'LOSSID': 'LOSS_ID',
 'HAVEREQUISITESOFAPPLICANT': 'HAVE_REQUISITES_OF_APPLICANT',
 'REQUIREDREVIEWSNEO': 'REQUIRED_REVIEWS_NEO',
 'EXTERNALORDERSNEO': 'EXTERNAL_ORDERS_NEO',
 'ISFULLPRETENSIONAMOUNTSWITHBREAKDOWN': 'IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN',
 'PRETENSIONTYPEID': 'PRETENSION_TYPE_ID',
 'PRETENSIONTYPES': 'PRETENSION_TYPES',
 'INSURANCETYPEGROUPS': 'INSURANCE_TYPE_GROUPS',
 'CESSION': 'CESSION',
 'LINKEDLOSSID': 'LINKED_LOSS_ID',
 'INCOMINGCLAIMGETDATE': 'INCOMING_CLAIM_GET_DATE',
 'INCIDENTNUMBER': 'INCIDENT_NUMBER',
 'DriverPersonID': 'DRIVER_PERSON_ID',
 'ApplicantID': 'APPLICANT_ID',
 'GuiltyObjectOwnerPersonID': 'GUILTY_OBJECT_OWNER_PERSON_ID',
 'GuiltyPersonID': 'GUILTY_PERSON_ID',
 'PaymentRecipientPersonID': 'PAYMENT_RECIPIENT_PERSON_ID',
 'PolicyholderObjectOwnerPersonID': 'POLICYHOLDER_OBJECT_OWNER_PERSON_ID',
 'PolicyholderPersonID': 'POLICYHOLDER_PERSON_ID',
 'VictimObjectOwnerPersonID': 'VICTIM_OBJECT_OWNER_PERSON_ID',
 'VictimPersonID': 'VICTIM_PERSON_ID',
 'VIctimPolicyholderPersonID': 'VICTIM_POLICYHOLDER_PERSON_ID',
 'RSAPolicyKBM': 'RSA_POLICY_KBM',
 'ReinsurancePool': 'REINSURANCE_POOL',
 'Drivers_cnt': 'DRIVERS_CNT',
 'SALES_CHANNEL': 'SALES_CHANNEL',
 'DATECANCELLATIONREVIEWS': 'DATE_CANCELLATION_REVIEWS',
 'SURCHARGEVALUE': 'SURCHARGE_VALUE',
 'IssueDate': 'ISSUE_DATE',
 'PRETENSION_TYPES_Запрос документов по делу': 'PRETENSION_TYPES_0',
 'PRETENSION_TYPES_Несогласие с суммой выплаты': 'PRETENSION_TYPES_1',
 'PRETENSION_TYPES_Отказ от ремонта (смена формы возмещения)': 'PRETENSION_TYPES_2',
 'PRETENSION_TYPES_Отказ по ТРЭ': 'PRETENSION_TYPES_3',
 'PRETENSION_TYPES_По расторгнутому договору': 'PRETENSION_TYPES_4',
 'PRETENSION_TYPES_Претензия на принятое решение': 'PRETENSION_TYPES_5',
 'PRETENSION_TYPES_Претензия на сроки ремонта и согласования': 'PRETENSION_TYPES_6',
 'PRETENSION_TYPES_Претензия по ДТП с 01.12.2014 по 01.05.2015': 'PRETENSION_TYPES_7',
 'PRETENSION_TYPES_Претензия по жизни': 'PRETENSION_TYPES_8',
 'PRETENSION_TYPES_Претензия по здоровью': 'PRETENSION_TYPES_9',
 'PRETENSION_TYPES_Претензия по качеству ремонта': 'PRETENSION_TYPES_10',
 'PRETENSION_TYPES_Требование по выплате только неустойки': 'PRETENSION_TYPES_11',
 'PRETENSION_TYPES_Частичный отказ по ТРЭ': 'PRETENSION_TYPES_12',
 'PRETENSION_TYPES_Электронное соглашение': 'PRETENSION_TYPES_13',
 'PAYMENTVALUE': 'PAYMENT_VALUE',
 'PRETENSIONVALUE_CUMSUM': 'PRETENSION_VALUE_CUMSUM',
 'SURCHARGE_VALUE_CUMSUM': 'SURCHARGE_VALUE_CUMSUM',
 'PAYMENT_VALUE_CUMSUM': 'PAYMENT_VALUE_CUMSUM',
 'LINKLOSSNUMBER': 'LINK_LOSS_NUMBER',
 'INCOMINGCLAIMNUMBER': 'INCOMING_CLAIM_NUMBER',
 'CLAIMEDPENALTYFEE_MAX': 'CLAIMED_PENALTY_FEE_MAX',
 'CLAIMEDVALUEWITHSD_MAX': 'CLAIMED_VALUE_WITH_SD_MAX',
 'RECOVEREDPENALTYFEE_MAX': 'RECOVERED_PENALTY_FEE_MAX',
 'RECOVEREDVALUEWITHSD_MAX': 'RECOVERED_VALUE_WITH_SD_MAX',
 'PENALTYFEE': 'PENALTY_FEE',
 'PAYMENTDATETIME': 'PAYMENT_DATETIME',
 'IncidentNumber': 'INCIDENT_NUMBER',
 'LossID': 'LOSS_ID',
 'LossNumber': 'LOSS_NUMBER',
 'PretensionID': 'PRETENSION_ID',
 'IsMarked': 'IS_MARKED',
 'PretensionDate': 'PRETENSION_DATE',
 'PretensionNumber': 'PRETENSION_NUMBER',
 'IsOver': 'IS_OVER',
 'PretensionGetDate': 'PRETENSION_GET_DATE',
 'PretensionType': 'PRETENSIONTYPE',
 'ApplicantPersonID': 'APPLICANT_PERSON_ID',
 'PretensionStage':'PRETENSION_STAGE',
 'PretensionGetMethod': 'PRETENSION_GET_METHOD',
 'PretensionValue': 'PRETENSION_VALUE',
 'PretensionCurrency': 'PRETENSIONCURRENCY',
 'UTSValue': 'UTSVALUE',
 'UTSCurrency': 'UTSCURRENCY',
 'LossUnit': 'LOSS_UNIT',
 'LossUnitZone': 'LOSS_UNIT_ZONE',
 'AnswerType': 'ANSWERTYPE',
 'AnswerDate': 'ANSWERDATE',
 'SurchargeValue': 'SURCHARGE_VALUE',
 'SurchargeCurrency': 'SURCHARGECURRENCY',
 'UTSSurchargeValue': 'UTSSURCHARGEVALUE',
 'UTSSurchargeCurrency': 'UTSSURCHARGECURRENCY',
 'Comment': 'COMMENT',
 'HaveRequisitesOfApplicant': 'HAVE_REQUISITES_OF_APPLICANT',
 'RequiredReviewSNEO': 'REQUIRED_REVIEWS_NEO',
 'DateCancellationReviews': 'DATE_CANCELLATION_REVIEWS',
 'ExternalOrderSNEO': 'EXTERNAL_ORDERS_NEO',
 'IsFullPretensionAmountsWithBreakdown': 'IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN',
 'SendInRSADate': 'SENDINRSADATE',
 'DateSentScannedCopies': 'DATESENTSCANNEDCOPIES',
 'PretensionTypeID': 'PRETENSION_TYPE_ID',
 'PretensionTypes': 'PRETENSION_TYPES',
 'PretensionKinds': 'PRETENSIONKINDS',
 'InsuranceTypes': 'INSURANCETYPES',
 'InsuranceTypeGroups': 'INSURANCE_TYPE_GROUPS',
 'Cession': 'CESSION',
 'LinkedLossID': 'LINKED_LOSS_ID',
 'PRETENSION_VALUE_2': 'PRETENSION_VALUE_2',
 'PRETENSIONKINDS': 'PRETENSION_KINDS',
 'ANSWERTYPE': 'ANSWER_TYPE',
 'UTSVALUE': 'UTS_VALUE',
 'UTSSURCHARGEVALUE': 'UTS_SURCHARGE_VALUE',
 'CLAIMITEM': 'CLAIM_ITEM'
 }



df_victim = pd.read_parquet('/home/jovyan/share/Matsera/df_Victim_final_11.parquet')
df_victim


df_victim = df_victim.query(
    'REFUND_FORM_DETAILED in ["Ремонт","Денежная","Денежная. Отказ от ремонта","Ремонт. Смена СТОА"]'
    'and LOSS_DATE_TIME >= "2022-01-01"'
    'and LOSS_DATE_TIME <= "2025-06-30"'
    'and LOSS_PROCESS in ["Прямое ОСАГО (с 1 марта 2009)","Традиционное ОСАГО"]'
    'and RISK == "Ущерб имуществу третьих лиц"')



df_victim = df_victim[['LOSS_NUMBER','LOSS_DATE_TIME', 'EVENT_DATE','PAYMENT_ORDER_DATE_TIME','ISSUE_DATE','LOSS_STATE_BY_IA','RSA_RE_OUT','CUSTOMER_IMPORTANCE','INSURANCE_TYPE_NAME'
 ,'REGISTRATING_SYSTEM','EVENT_LOCATION_DESCRIPTION','LONGITUDE','LATITUDE','POST_INDEX','FILIAL','GUILTY_CONTRACT_NUMBER','VICTIM_CONTRACT_NUMBER'
 ,'GUILTY_VEHICLE_IS_JAPAN','VICTIM_VEHICLE_IS_JAPAN','GUILTY_VEHICLE_BRAND','GUILTY_VEHICLE_MODEL','VICTIM_VEHICLE_BRAND','VICTIM_VEHICLE_MODEL',
 'GUILTY_VEHICLE_CATEGORY','VICTIM_VEHICLE_CATEGORY','GUILTY_VEHICLE_MADE_IN_RF','VICTIM_VEHICLE_MADE_IN_RF','GUILTY_POLICY_ISSUER',
 'VICTIM_POLICY_ISSUER','EVENT_CREATED_BY_GIBDD_FLAG','FL_PHOTO_VIDEO','NOTIFICATION_METHOD','LOSS_PROCESS','VICTIM_VEHICLE_TYPE_BY_CLASSIFICATOR'
 ,'GUILTY_VEHICLE_TYPE_BY_CLASSIFICATOR','REFUND_FORM','REFUND_FORM_DETAILED','PAYMENTS_SUM_RUR','RISK','LOSS_UNIT','LOSS_UNIT_ZONE','LOSS_UNIT_TYPE'
 ,'LOSS_UNIT_DIVISION','PARTICIPANTS_COUNT','GUILTY_OBJECT_VIN','VICTIM_OBJECT_VIN','TER_OSAGO','USED_AS_CARSH','USED_AS_TAXI','VICTIM_OBJECT_MULTIDRIVE',
 'PROFITCENTER_CODE','FRANCHISE_VALUE','APPLICANT_FORM','RECIEVE_METHOD','POLICY_ADACTA_ID','SALE_CHANEL','SALE_CHANEL1','INSURANCE_AMOUNT','CONTACT_DATE_TIME',
 'VICTIM_CONTRACT_START_DATE','VICTIM_CONTRACT_END_DATE','REGION','INS_OBJECT_DRIVERS_LIST','VICTIM_INSURER_PERSON_BIRTH_DATE','INCIDENT_NUMBER',
 'VICTIM_OBJECT_YEAR','VICTIM_OBJECT_POWER','GUILTY_OBJECT_YEAR','GUILTY_OBJECT_POWER','GUILTY_VEHICLE_AGE','VICTIM_VEHICLE_AGE','VICTIM_TYPE_KPP',
 'VICTIM_TYPE_ENGINE','VICTIM_CAPACITY_ENGINE','VICTIM_POWER_ENGINE','VICTIM_TYPE_BODY','VICTIM_TYPE_PRIVOD','VICTIM_MAX_WEIGHT','VICTIM_NUM_PLACE'
 ,'VICTIM_NUM_DOORS','GUILTY_TYPE_KPP','GUILTY_TYPE_ENGINE','GUILTY_CAPACITY_ENGINE','GUILTY_POWER_ENGINE','GUILTY_TYPE_BODY','GUILTY_TYPE_PRIVOD'
 ,'GUILTY_MAX_WEIGHT','DRIVER','DRIVER_SEX','DRIVER_BIRTH_DATE','GUILTY_TYPE','GUILTY_SEX','GUILTY_BIRTH_DATE','VICTIM','VICTIM_TYPE','VICTIM_SEX'
 ,'VICTIM_BIRTH_DATE','APPLICANT','APPLICANT_TYPE','APPLICANT_SEX','APPLICANT_BIRTH_DATE','PAYMENT_RECIPIENT','PAYMENT_RECIPIENT_TYPE','PAYMENT_RECIPIENT_SEX'
 ,'PAYMENT_RECIPIENT_BIRTH_DATE','GUILTY_VEHICLE_OWNER_TYPE','VICTIM_VEHICLE_OWNER_TYPE','VICTIM_VEHICLE_OWNER_SEX','VICTIM_VEHICLE_OWNER_BIRTH_DATE',
 'GUILTY_VEHICLE_OWNER','VICTIM_VEHICLE_OWNER','GUILTY_VEHICLE_OWNER_SEX','GUILTY_VEHICLE_OWNER_BIRTH_DATE','VICTIM_POLICYHOLDER','VICTIM_POLICYHOLDER_BIRTH_DATE'
 ,'VICTIM_POLICYHOLDER_TYPE','ACCEPTED_UNIT','PAYMENT_TYPE','VEHICLE_CONDITION','FEDERAL_DISTRICT','VICTIM_VEHICLE_COUNTRY','GUILTY_VEHICLE_COUNTRY','PAYMENT_VALUE',
 'GUILTY_AGE','DRIVER_AGE','VICTIM_AGE','APPLICANT_AGE','GUILTY_VEHICLE_AGE_BAD','VICTIM_VEHICLE_AGE_BAD','DIFF_TS_AGE','EVENT_HOUR','EVENT_DAY','EVENT_MONTH'
 ,'EVENT_YEAR','GUILTY_TS_REGION','VICTIM_TS_REGION','GUILTY_POLICY_ISSUER_GROUP','VICTIM_POLICY_ISSUER_GROUP','MULTIDRIVE','VIC_TS_COUNTRY','GUIL_TS_COUNTRY'
 ,'APPLY_DELAY','EVENT_FROM_START_POLICYHOLDER','EVENT_FROM_START_VICTIM','NOT_NOTIFICATION','REGION_EVENT','REGION_CORRECTED','VIC_IS_EV_REG','GUIL_IS_EV_REG'
 ,'GROUPS_PERSON_TYPE','DRIVER_AGE_GROUP','VICTIM_VEHICLE_TYPE','GUILTY_VEHICLE_TYPE','VICTIM_GOS','GUILTY_GOS','EVENT_NUMBER','HEALTH_FLAG','LOSS_AMOUNT','FIX'
 ,'FIN','EVENT_DATE_DAY','REFUND_FORM_INCIDENT','VICTIM_LOSS_SUM','VICTIM_LOSS_COUNT','VICTIM_LOSS_SUM_FUTURE','VICTIM_LOSS_COUNT_FUTURE','begin_calc',
 'GUILTY_LOSS_SUM','GUILTY_LOSS_COUNT','GUILTY_LOSS_SUM_FUTURE','GUILTY_LOSS_COUNT_FUTURE','RSAPolicyKBM','ReinsurancePool','Drivers_cnt','SALES_CHANNEL','DTP_COUNT'
 ,'TYPE_LAST','CARDS_CNT','OWNERS_CNT','OWNERS_PERIOD','RESTRICT_CNT','RESTRICT_PERIOD','PREMIUM_CALC','PREMIUM_SUM_ALL'
 ,'PREMIUM_COUNT_ALL','PREMIUM_SUM_FUTURE_ALL','PREMIUM_COUNT_FUTURE_ALL','PREMIUM_SUM_OSAGO','PREMIUM_COUNT_OSAGO','VICTIM_POLICYHOLDER_PERSON_ID','APPLICANT_ID'
 ]]
df_victim



query = \
"""
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
#df_claims_persons = pd.read_sql_query(query, conn_oisuu) 
#df_claims_persons.to_parquet('/home/jovyan/Litigant/parquet/df_claims_persons.parquet')
df_claims_persons = pd.read_parquet('/home/jovyan/old_home/Litigant/parquet/df_claims_persons.parquet')
df_claims_persons



query_pay = \
"""
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
)
SELECT
       itl.[IncidentNumber]
      ,icnl.[IncomingClaimNumber]
      ,RecoveredValueWithSD
      ,docs.[_Fld14757] as [Payment_fee_fu]
      ,ROW_NUMBER() OVER (PARTITION BY IncomingClaimNumber ORDER BY RecoveredValuePeriod desc) rn_inst
FROM [OISUU_report].[Datamart].[oisuu81_t_IncomingClaimNewLogicByInst] as icnl
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl on itl.LossNumber=icnl.LinkLossNumber
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Losses] as l on l.LossNumber=itl.LossNumber
LEFT JOIN doc as docs on docs.[_Document5472_IDRRef]=icnl.[IncomingClaimID]
WHERE (ClaimOrigin not in ('Исходящий иск') or ClaimOrigin is null )
    AND (docs.rn = 1 or docs.rn is null)
    and icnl.Instance != 'Суд кассационной инстанции'
    and icnl.RecoveredValuePeriod is not null
"""

df_claims_pay = pd.read_sql_query(query_pay, conn_oisuu)
df_claims_pay


# Группируем все взыскания по инциденты 
df_claims_pay['RecoveredValueWithSD'] = df_claims_pay['RecoveredValueWithSD'].fillna(0)
df_claims_pay['Payment_fee_fu'] = df_claims_pay['Payment_fee_fu'].fillna(0)
df_claims_pay = df_claims_pay[df_claims_pay['rn_inst']==1].groupby('IncidentNumber')[['RecoveredValueWithSD','Payment_fee_fu']].agg('sum').reset_index()
df_claims_pay[:2]


conn_oisuu.close()


query = \
"""
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
WHERE (ClaimOrigin not in ('Исходящий иск') or ClaimOrigin is null )
    AND (docs.rn = 1 or docs.rn is null)
"""


df_claims = pd.read_sql_query(query, conn_oisuu)


df_claims.to_parquet('/home/jovyan/old_home/Litigant/parquet/df_claims.parquet')

df_claims = pd.read_parquet('/home/jovyan/old_home/Litigant/parquet/df_claims.parquet')
df_claims

df_claims.columns = df_claims.columns.str.upper()
df_claims = df_claims.rename(columns=rename_dict)

def convert_to_binary(value):
    if value == '00' or value == b'\x00':
        return 0
    elif value == '01' or value == b'\x01':
        return 1
    else:
        return 0


# Применение функции к колонке DataFrame
df_claims['ПРЕДСТАВИТЕЛЬ'] = df_claims['ПРЕДСТАВИТЕЛЬ'].apply(convert_to_binary)
df_claims['ЦЕССИОНАРИЙ'] = df_claims['ЦЕССИОНАРИЙ'].apply(convert_to_binary)


df_base = df_claims[['INCIDENT_NUMBER', 'LOSS_ID', 'LOSS_NUMBER',
                     'INCOMINGCLAIMID', 'INCOMING_CLAIM_NUMBER',
                     'LINK_LOSS_NUMBER']].drop_duplicates(['INCOMING_CLAIM_NUMBER'])
df_base = df_base.sort_values(by=['INCOMING_CLAIM_NUMBER']).reset_index(drop=True)



# Отбираем только необходимые колонки (относящиеся к судам)
df_sorted = df_claims.loc[:, ['LOSS_NUMBER'] + list(df_claims.iloc[:, 6:].columns)]
# Удаляем ненужные колонки
df_sorted = df_sorted.drop(['INSTID', 'LINK_LOSS_NUMBER'], axis=1)
#df_sorted = df_sorted.drop(['INSTID'], axis=1)
# Сортируем датафрейм по номеру иска и дате выставления, чтобы инстанции шли по временному порядку
df_sorted = df_sorted.sort_values(by=['INCOMING_CLAIM_NUMBER', 'CLAIMEDVALUEPERIOD']).reset_index(drop=True)
# Первое вхождение в какую-либо инстанцию (будь то 6, или 1 инстанции) в отдельно взятом номере иска == перва инстанция
# для данного иска (cumcount)
df_sorted['Instance'] = df_sorted.groupby(['INCOMING_CLAIM_NUMBER']).cumcount() + 1
# Сводная таблица по искам и инстанциям в строку
df_pivot = df_sorted.pivot(index=['INCOMING_CLAIM_NUMBER'], columns='Instance')
# Колонки переименовываем в соответствии с номером инстанции (ClaimedValueWithSD_1, ClaimedValueWithSD_2 И т.д.)
df_pivot.columns = [f'{col[0]}_{int(col[1])}' for col in df_pivot.columns]
df_pivot = df_pivot.reset_index()
df_pivot[df_pivot['INCOMING_CLAIM_NUMBER']==434485]


df_claims = df_base.merge(df_pivot, how='left', on=['INCOMING_CLAIM_NUMBER'])



group_map_item = {
 
 'ВРЕД ЗДОРОВЬЮ':'ЖиЗ',
 'ВРЕД ЖИЗНИ':'ЖиЗ',
 'ВРЕД ЖИЗНИ И/ИЛИ ЗДОРОВЬЮ':'ЖиЗ',
 'ВОЗМЕЩЕНИЕ ВРЕДА, ПРИЧИНЕННОГО ЖИЗНИ И ЗДОРОВЬЮ':'ЖиЗ',
 'ВОЗМЕЩЕНИЕ ПО ПОТЕРЕ КОРМИЛЬЦА':'ЖиЗ',
 
 'МОРАЛЬНЫЙ ВРЕД':'МОРАЛЬНЫЙ ВРЕД',
 'О ВЗЫСКАНИИ МОРАЛЬНОГО ВРЕДА':'МОРАЛЬНЫЙ ВРЕД',
 
 'НЕДОСТАТКИ РЕМОНТА':'НЕДОСТАТКИ РЕМОНТА',
 'УСТРАНЕНИЕ НЕДОСТАТКОВ РЕМОНТА':'НЕДОСТАТКИ РЕМОНТА',
 'КОМПЕНСАЦИЯ НЕДОСТАТКОВ РЕМОНТА':'НЕДОСТАТКИ РЕМОНТА',
 
 'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ Е-ОСАГО':'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ',
 'НИЧТОЖНЫЙ ИЛИ ПОДДЕЛЬНЫЙ ПОЛИС ОСАГО':'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ',
 'СОБЩЕНИЕ ЗАВЕДОМО ЛОЖНЫХ СВЕДЕНИЙ':'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ',
 'СООБЩЕНИЕ ЗАВЕДОМО ЛОЖНЫХ СВЕДЕНИЙ':'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ',
 
 'НЕ ПРЕДОСТАВЛЕНИЕ ПОВРЕЖДЕННОГО ОБЪЕКТА СТРАХОВАНИЯ НА ОСМОТР':'НЕ ПРЕДОСТАВЛЕНИЕ НА ОСМОТР',
 'НЕ ПРЕДОСТАВЛЕНИЕ ПОВРЕЖДЕННОГО ИМУЩЕСТВА НА ОСМОТР':'НЕ ПРЕДОСТАВЛЕНИЕ НА ОСМОТР',
 
 'ДОПЛАТА':'СПОР ПО СУММЕ',
 'СПОР ПО СУММЕ':'СПОР ПО СУММЕ',
 'ДОП. РАСХОДЫ':'СПОР ПО СУММЕ',

 
 'НЕПРИЗНАНИЕ СОБЫТИЯ СТРАХОВЫМ СЛУЧАЕМ ПО ЗАКОНУ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
 'НЕПРИЗНАНИЕ СОБЫТИЯ СТРАХОВЫМ СЛУЧАЕМ ПО ПРАВИЛАМ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
 'НЕПРИЗНАНИЕ\xa0 СОБЫТИЯ СТРАХОВЫМ СЛУЧАЕМ ПО ЗАКОНУ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
 'СОБЫТИЕ НЕ В СТРАХОВОЙ\xa0 ПЕРИОД': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
 'НЕПРИЗНАНИЕ СОБЫТИЯ СТРАХОВЫМ СЛУЧАЕМ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
 'СОБЫТИЕ ПРОИЗОШЛО НЕ В ПЕРИОД ДЕЙСТВИЯ ДОГОВОРА СТРАХОВАНИЯ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
 
 'НЕУСТОЙКА/ШТРАФ ПО ЗОЗПП ОТДЕЛЬНЫМ ИСКОМ': 'НЕУСТОЙКА',
 'НЕУСТОЙКА ПО ОСАГО ОТДЕЛЬНЫМ ИСКОМ': 'НЕУСТОЙКА',
 'ВЗЫСКАНИЕ НЕУСТОЙКИ': 'НЕУСТОЙКА',
 'ПРОЦЕНТЫ ЗА ПОЛЬЗОВАНИЕ ЧУЖИМИ ДЕНЕЖНЫМИ СРЕДСТВАМИ':'НЕУСТОЙКА',
 

 'ИНДЕКСАЦИЯ ПРИСУЖДЕННЫХ СУММ':'НЕУСТОЙКА',
 
 'ИЗМЕНЕНИЕ УСЛОВИЙ СТРАХОВАНИЯ': 'УСЛОВИЯ СТРАХОВАНИЯ',
 'ПО УСЛОВИЯМ ДОГОВОРА': 'УСЛОВИЯ СТРАХОВАНИЯ',
 
 'ВОЗВРАТ СТОИМОСТИ ТУРИСТИЧЕСКОГО ПРОДУКТА': 'ВОЗВРАТ СТОИМОСТИ ТУРИСТИЧЕСКОГО ПРОДУКТА',
 'ГО ПО ДОГОВОРУ О РЕАЛИЗАЦИИ ТУРИСТИЧЕСКОГО ПРОДУКТА': 'ВОЗВРАТ СТОИМОСТИ ТУРИСТИЧЕСКОГО ПРОДУКТА',
 
 'ПРАВО НА ПВУ': 'ПВУ',
 'ПО ПВУ': 'ПВУ', 
 'СПОР О ПВУ': 'ПВУ', 
 
 'СПОР О ФОРМЕ СТРАХОВОГО ВОЗМЕЩЕНИЯ (РЕМОНТ/ВЫПЛАТА)': 'ФОРМА ВОЗМЕЩЕНИЯ',
 'СПОР О ФОРМЕ ВЫПЛАТЫ (РЕМОНТ/ДЕНЬГИ)': 'ФОРМА ВОЗМЕЩЕНИЯ', 
 'УСЛОВИЕ "РЕМОНТ"': 'ФОРМА ВОЗМЕЩЕНИЯ',
 'ВЫДАТЬ НАПРАВЛЕНИЕ НА РЕМОНТ': 'ФОРМА ВОЗМЕЩЕНИЯ',
 'ОРГАНИЗАЦИЯ РЕМОНТА НА СТОА': 'ФОРМА ВОЗМЕЩЕНИЯ',
 'ОБЯЗАТЕЛЬНЫЙ РЕМОНТ ОСАГО.ЦЕССИЯ ЮЛ': 'ФОРМА ВОЗМЕЩЕНИЯ',
 'ЦЕССИЯ ОСАГО.ОБЯЗАТЕЛЬНЫЙ РЕМОНТ': 'ФОРМА ВОЗМЕЩЕНИЯ',
 
 'ВОЗВРАТ СТРАХОВОЙ ПРЕМИИ': 'ВОЗВРАТ СТРАХОВОЙ ПРЕМИИ',
 'ВОЗВРАТ ПРЕМИИ (КБМ)': 'ВОЗВРАТ СТРАХОВОЙ ПРЕМИИ', 
 'ОСПАРИВАНИЕ КБМ (ОСАГО)': 'ВОЗВРАТ СТРАХОВОЙ ПРЕМИИ',
 
 'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА': 'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА',
 'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА (КАСКО)': 'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА', 
 'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА (НС)': 'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА',

 
 
 'ПРЕКРАЩЕНИЕ ДОГОВОРА ОСАГО': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ', 
 'О ПРИЗНАНИИ НЕДЕЙСТВИТЕЛЬНЫМ ДОГОВОРА СТРАХОВАНИЯ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 'ПЕРИОД ОХЛАЖДЕНИЯ (КАСКО)': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ', 
 'ПЕРИОД ОХЛАЖДЕНИЯ (НС)': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 'ПЕРИОД ОХЛАЖДЕНИЯ (ИФЛ)': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 'ДОСРОЧНОЕ ПРЕКРАЩЕНИЕ ДОГОВОРА КАСКО': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 'ПЕРИОД ОХЛАЖДЕНИЯ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 'РАСТОРЖЕНИЕ, ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 'ПРИЗНАНИЕ УСЛОВИЙ КРЕДИТНОГО ДОГОВОРА НЕДЕЙСТВИТЕЛЬНЫМИ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 'ПРЕКРАЩЕНИЕ ДОГОВОРА ОСАГО': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 'ПОНУЖДЕНИЕ К ЗАКЛЮЧЕНИЮ ДОГОВОРА': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 
 'НЕВРУЧЕНИЕ КИД (НС)': 'КИД', 
 'НЕВРУЧЕНИЕ КИД ': 'КИД',
 'НЕКОРРЕКТНЫЙ КИД': 'КИД',

 'ПРЕВЫШЕНИЕ СТРАХОВОЙ СУММЫ НАД СТОИМОСТЬЮ': 'ПРЕВЫШЕНИЕ СТРАХОВОЙ СУММЫ НАД СТОИМОСТЬЮ', 
 'ПРЕВЫШЕНИЕ СТРАХОВОЙ СУММЫ НАД СТРАХОВОЙ СТОИМОСТЬЮ': 'ПРЕВЫШЕНИЕ СТРАХОВОЙ СУММЫ НАД СТОИМОСТЬЮ',


 'НЕДОКАЗАННОСТЬ ОБСТОЯТЕЛЬСТВ НАСТУПЛЕНИЯ СТРАХОВОГО СЛУЧАЯ (ТРАСОЛОГИЯ)': 'ТРАСОЛОГИЯ', 
 'НЕДОКАЗАННОСТЬ РАЗМЕРА УЩЕРБА (ТРАСОЛОГИЯ)': 'ТРАСОЛОГИЯ',
 'ЧАСТИЧНЫЙ ОТКАЗ ПО ТРАСОЛОГИИ': 'ТРАСОЛОГИЯ',
 'ПОЛНЫЙ ОТКАЗ ПО ТРЭ': 'ТРАСОЛОГИЯ',
 'ЧАСТИЧНЫЙ ОТКАЗ ПО ТРЭ': 'ТРАСОЛОГИЯ',
 
 'ИНОЕ – РСА 3-Е ЛИЦО':'3_Е_ЛИЦО',
 'ВСК - 3 ЛИЦО':'3_Е_ЛИЦО',
 'ИНОЕ – ВСК 3-Е ЛИЦО':'3_Е_ЛИЦО',
 
 
 'ОСПАРИВАНИЕ СОГЛАШЕНИЯ ОБ УРЕГУЛИРОВАНИИ ОСАГО':'ОСПАРИВАНИЕ СОГЛАШЕНИЯ ОБ УРЕГУЛИРОВАНИЕ' ,
 
 'ЖИЗ СОЛИДАРНАЯ ОТВЕТСТВЕННОСТЬ':'ОБОЮДНАЯ ВИНА',
 
 'СУДЕБНЫЕ РАСХОДЫ':'СУДЕБНЫЕ РАСХОДЫ И САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА', 
 'САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА':'СУДЕБНЫЕ РАСХОДЫ И САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА',
 'САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ':'СУДЕБНЫЕ РАСХОДЫ И САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА',
 'ТРЕБОВАНИЕ ПО ВЫПЛАТЕ ШТРАФНЫХ САНКЦИЙ':'СУДЕБНЫЕ РАСХОДЫ И САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА'
}
df_claims['CLAIM_ITEM_1'] = df_claims['CLAIM_ITEM_1'].str.upper()
df_claims['CLAIM_ITEM'] = df_claims['CLAIM_ITEM_1'].str.upper().replace(group_map_item).fillna(df_claims['CLAIM_ITEM_1'])


cat_claim_item = list(df_claims['CLAIM_ITEM'].value_counts().reset_index()[0:9]['CLAIM_ITEM'].unique())
df_claims['CLAIM_ITEM'] = df_claims['CLAIM_ITEM'].apply(lambda x : x if x in cat_claim_item else 'ПРОЧЕЕ')
df_claims['CLAIM_ITEM'].value_counts()


#ohe_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
ohe_enc = OneHotEncoder( handle_unknown='ignore')
cat_features = ['CLAIM_ITEM']
ohe_enc.fit(df_claims[cat_features])
enc = ohe_enc.transform(df_claims[cat_features]).toarray()
enc = pd.DataFrame(enc, columns=ohe_enc.get_feature_names_out())
df_claims_ = pd.concat([df_claims.reset_index(drop=True), enc.reset_index(drop=True)], axis=1)
df_claims_[:2]


df_claims_.to_parquet('/home/jovyan/old_home/Litigant/data/processed/df_claims_.parquet')
df_claims.to_parquet('/home/jovyan/old_home/Litigant/data/processed/df_claims.parquet')

df_claims_ = pd.read_parquet('/home/jovyan/old_home/Litigant/data/processed/df_claims_.parquet')
df_claims = pd.read_parquet('/home/jovyan/old_home/Litigant/data/processed/df_claims.parquet')


df_payments_q = ("""
    SELECT *
    FROM [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as ITL
    LEFT JOIN [OISUU_report].[dbo].oisuu81_t_payments AS p on p.LOSSID = ITL.LOSSID
""")

df_payments = pd.read_sql(df_payments_q, conn_oisuu)

print(df_payments.columns[df_payments.columns.duplicated()])
df_payments = df_payments.loc[:, ~df_payments.columns.duplicated()].copy()
df_payments[:1]


df_payments = pd.read_parquet('/home/jovyan/old_home/Litigant/data/raw/df_payments.parquet')
df_payments


column_mapping = {
    'LossID': 'LOSSID',
    'LossNumber': 'LOSSNUMBER',
    'IncidentID': 'INCIDENTID',
    'IncidentNumber': 'INCIDENTNUMBER',
    'PaymentDateTime': 'PAYMENTDATETIME',
    'PaymentNumber': 'PAYMENTNUMBER',
    'PaymentID': 'PAYMENTID',
    'PaymentValue': 'PAYMENTVALUE',
    'DocumentNumber': 'DOCUMENTNUMBER',
    'ValueRub': 'VALUERUB',
    'Comment': 'COL_COMMENT',
    'DocumentOperation': 'DOCUMENTOPERATION',
    'CurrencyCode': 'CURRENCYCODE',
    'Currency': 'CURRENCY',
    'PaymentRecipientPersonName': 'PAYMENTRECIPIENTPERSONNAME',
    'PaymentRecipientPersonCode': 'PAYMENTRECIPIENTPERSONCODE',
    'PaymentRecipientPersonID': 'PAYMENTRECIPIENTPERSONID',
    'IsReasonForPaymentExternalOrder': 'ISREASONFORPAYMENTEXTERNALORDER',
    'IsReasonForPaymentVPRS': 'ISREASONFORPAYMENTVPRS',
    'IsReasonForPaymentInsuranceAct': 'ISREASONFORPAYMENTINSURANCEACT',
    'IsReasonForPaymentCompletedWorkGetAct': 'ISREASONFORPAYMENTCOMPLETEDWORKGETACT',
    'ReasonForPayment': 'REASONFORPAYMENT'
}

df_payments.rename(columns=column_mapping, inplace=True)


df_payments = df_payments.rename(columns=rename_dict)
df_payments = df_payments.sort_values('PAYMENT_DATETIME')


df_payments['PAYMENT_VALUE_CUMSUM'] = df_payments.groupby(['INCIDENT_NUMBER'])['PAYMENT_VALUE'].cumsum()


df_claims_payments = df_claims.merge(df_payments[['INCIDENT_NUMBER',
                                                  'PAYMENT_DATETIME',
                                                  'PAYMENT_VALUE_CUMSUM']],
                                     left_on='INCIDENT_NUMBER',
                                     right_on='INCIDENT_NUMBER',
                                     how='left')
df_claims_payments

df_claims_payments.to_parquet('/home/jovyan/old_home/Litigant/data/processed/df_claims_payments.parquet')

df_claims_payments = pd.read_parquet('/home/jovyan/old_home/Litigant/data/processed/df_claims_payments.parquet')
df_claims_payments.shape

# Шаг 1: Фильтрация — делаем ОДИН раз, без копирования лишнего
mask = (df_claims_payments['PAYMENT_DATETIME'] <= df_claims_payments['RECOVEREDVALUEPERIOD_1']) | \
       df_claims_payments['PAYMENT_DATETIME'].isnull()

# Применяем маску и сразу освобождаем исходный df, если он больше не нужен
df_claims_payments = df_claims_payments[mask]

# Шаг 2: Сортировка — делаем только по необходимым столбцам
# Убедитесь, что столбцы datetime имеют правильный тип (datetime64)
df_claims_payments['RECOVEREDVALUEPERIOD_1'] = pd.to_datetime(df_claims_payments['RECOVEREDVALUEPERIOD_1'])
df_claims_payments['PAYMENT_DATETIME'] = pd.to_datetime(df_claims_payments['PAYMENT_DATETIME'])

df_claims_payments = df_claims_payments.sort_values(
    ['RECOVEREDVALUEPERIOD_1', 'PAYMENT_DATETIME'],
    na_position='first',
#    kind='mergesort'  # стабильная сортировка, иногда эффективнее по памяти
)

# Шаг 3: Удаление дубликатов — используем keep='last'
df_claims_payments = df_claims_payments.drop_duplicates('INCOMING_CLAIM_NUMBER', keep='last')

df_claims_payments.to_parquet('/home/jovyan/old_home/Litigant/data/processed/df_claims_payments.parquet')

# ОТПРАВНАЯ ТОЧКА 1 по судам + выплатам
df_claims_payments = pd.read_parquet('/home/jovyan/old_home/Litigant/data/processed/df_claims_payments.parquet')

gc.collect()

df_pretensions_ = ("""
SELECT *
  FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS P
  LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS ITL ON ITL.LossID=P.LossID
""")
df_pretensions = pd.read_sql(df_pretensions_, conn_oisuu)

print(df_pretensions.columns[df_pretensions.columns.duplicated()])
df_pretensions = df_pretensions.loc[:, ~df_pretensions.columns.duplicated()].copy()
df_pretensions[:1]

df_pretensions.to_parquet('/home/jovyan/old_home/Litigant/data/raw/df_pretensions.parquet')

df_pretensions = pd.read_parquet('/home/jovyan/old_home/Litigant/data/raw/df_pretensions.parquet')

# хадуп не работает, поэтому берем нужные id фио из sql
sql_pret_id = ( """
select 
    pr.PretensionNumber PRETENSION_NUMBER
    ,L.LossNumber LOSS_NUMBER
    ,(L.PolicyholderPersonID) as POLICYHOLDER_PERSON_ID
    ,(L.VictimPersonID) as VICTIM_PERSON_ID
    ,(L.VIctimPolicyholderPersonID) as VICTIM_POLICYHOLDER_PERSON_ID
    , L.VictimObjectOwnerPersonID as VICTIM_OBJECT_OWNER_PERSON_ID
 from OISUU_REPORT.DBO.oisuu81_t_Pretensions pr
 left join OISUU_REPORT.DBO.oisuu81_t_Losses L on l.LossID=pr.LossID

""")

pretension_fio_id = pd.read_sql(sql_pret_id,conn_oisuu)
pretension_fio_id.shape

pretension_fio_id['POLICYHOLDER_PERSON_ID'] = pretension_fio_id['POLICYHOLDER_PERSON_ID'].apply(lambda x: x.hex().upper() if x is not None  else np.nan)
pretension_fio_id['VICTIM_PERSON_ID'] = pretension_fio_id['VICTIM_PERSON_ID'].apply(lambda x: x.hex().upper() if x is not None  else np.nan)
pretension_fio_id['VICTIM_POLICYHOLDER_PERSON_ID'] = pretension_fio_id['VICTIM_POLICYHOLDER_PERSON_ID'].apply(lambda x: x.hex().upper() if x is not None  else np.nan)
pretension_fio_id['VICTIM_OBJECT_OWNER_PERSON_ID'] = pretension_fio_id['VICTIM_OBJECT_OWNER_PERSON_ID'].apply(lambda x: x.hex().upper() if x is not None  else np.nan)
pretension_fio_id[:2]

df_pretensions.columns = df_pretensions.columns.str.upper()
df_pretensions = df_pretensions.rename(columns=rename_dict)

#Добавим id в основной датасет по претензиям
print(df_pretensions.shape)
df_pretensions = df_pretensions.merge(pretension_fio_id[['PRETENSION_NUMBER','VICTIM_POLICYHOLDER_PERSON_ID','VICTIM_OBJECT_OWNER_PERSON_ID']],how='left',on='PRETENSION_NUMBER')
print(df_pretensions.shape)

query = \
"""
-- ОТСЮДА БЕРУТСЯ ПРЕТЕНЗИИ С СУММОЙ ПО НЕУСТОЙКЕ
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
	-- left join [oisuu81].[dbo].[_Document6169_VT9237] dcv on dcv._Document6169_IDRRef = p.PretensionID
	 WHERE _Fld9244RRef in (0xB6B5441EA172DD2611E8AC27427E4644, 0xB6B5441EA172DD2611E8AC282FFD5C5A)
 ),
 -- ОТСЮДА БЕРУТСЯ ТЕ ЖЕ ПРЕТЕНЗИИ, КОТОРЫЕ В ЗАПРОСЕ ВЫШЕ, НО ВЫТАСКИВАЮТСЯ СУММЫ ДОПЛАТ
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
 --left join tmp_1 as tmp_1 on tmp.PretensionNumber=tmp_1.PretensionNumber
 WHERE rn = 1
 union select * from penalty where rn = 1
"""
df_pretensions_3 = pd.read_sql_query(query, conn_oisuu)

df_pretensions_3.to_parquet('/home/jovyan/old_home/Litigant/data/raw/df_pretensions_3.parquet')

df_pretensions_3 = pd.read_parquet('/home/jovyan/old_home/Litigant/data/raw/df_pretensions_3.parquet')
df_pretensions_3.shape


df_pretensions_3.columns = df_pretensions_3.columns.str.upper()
df_pretensions_3 = df_pretensions_3.rename(columns=rename_dict)
df_pretensions_3 = df_pretensions_3.sort_values(['PRETENSION_NUMBER', 'SURCHARGE_VALUE_PENALTY'])

df_pretensions_3 = df_pretensions_3.drop_duplicates(['PRETENSION_NUMBER'], keep='last')
df_pretensions_3.duplicated('PRETENSION_NUMBER').sum()

df_pretensions = df_pretensions.merge(df_pretensions_3[['PRETENSION_NUMBER',
                                                        'PRETENSION_VALUE_PENALTY',
                                                        'SURCHARGE_VALUE_PENALTY']], how='left', on='PRETENSION_NUMBER')
df_pretensions['PRETENSION_VALUE_PENALTY'] = df_pretensions['PRETENSION_VALUE_PENALTY'].fillna(0)
df_pretensions['SURCHARGE_VALUE_PENALTY'] = df_pretensions['SURCHARGE_VALUE_PENALTY'].fillna(0)


# Применение функции к колонке DataFrame
df_pretensions['IS_MARKED'] = df_pretensions['IS_MARKED'].apply(convert_to_binary)
df_pretensions['IS_OVER'] = df_pretensions['IS_OVER'].apply(convert_to_binary)
df_pretensions['CESSION'] = df_pretensions['CESSION'].apply(convert_to_binary)
df_pretensions['HAVE_REQUISITES_OF_APPLICANT'] = df_pretensions['HAVE_REQUISITES_OF_APPLICANT'].apply(convert_to_binary)
df_pretensions['REQUIRED_REVIEWS_NEO'] = df_pretensions['REQUIRED_REVIEWS_NEO'].apply(convert_to_binary)
df_pretensions['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN'] = df_pretensions['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN'].apply(convert_to_binary)

df_pretensions = df_pretensions[df_pretensions['IS_MARKED'] == 0]

df_pretensions = df_pretensions.sort_values(by=['INCIDENT_NUMBER', 'PRETENSION_GET_DATE'])

list_PRETENSION_TYPES = list(df_pretensions['PRETENSION_TYPES'].value_counts().reset_index()[:9]['PRETENSION_TYPES'].unique())
df_pretensions['PRETENSION_TYPES_'] = df_pretensions['PRETENSION_TYPES'].apply(lambda x: x if x in list_PRETENSION_TYPES else 'ПРОЧЕЕ')
df_pretensions['INSURANCE_TYPE_GROUPS_'] = df_pretensions['INSURANCE_TYPE_GROUPS'].apply(lambda x: x if x in ['ОСАГО','КАСКО+ГО'] else 'ПРОЧЕЕ')

map_ANSWER_TYPE = {
'Уведомление о доплате':'Частичная выплата',
'Согласован дополнительный объем ремонта' :'Выплата',
'Уведомление о проделанной работе': 'Направлен ответ',
'Приглашение в офис'  : 'Направлен ответ',
'Отказ повторный': 'Отказ в удовлетворении претензии',
'Отказ, нет реквизитов': 'Отказ в удовлетворении претензии', 
'Направлен на ремонт' :'Выплата',
'Приглашение на ремонт' :'Выплата'
    
}
df_pretensions['PRETENSION_GET_METHOD_'] = df_pretensions['PRETENSION_GET_METHOD'].apply(lambda x: 'ПРОЧЕЕ' if x in ['Иное','Партнер'] else x)
df_pretensions['ANSWER_TYPE_'] = df_pretensions['ANSWER_TYPE'].replace(map_ANSWER_TYPE)

#ohe_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
ohe_enc = OneHotEncoder( handle_unknown='ignore')
cat_features = ['INSURANCE_TYPE_GROUPS_','PRETENSION_TYPES_', 'PRETENSION_GET_METHOD_', 'ANSWER_TYPE_']
ohe_enc.fit(df_pretensions[cat_features])
enc = ohe_enc.transform(df_pretensions[cat_features]).toarray()
enc = pd.DataFrame(enc, columns=ohe_enc.get_feature_names_out())

df_pretensions['PRETENSION_CUMCOUNT'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['PRETENSION_NUMBER'].cumcount() + 1

df_pretensions['PRETENSION_VALUE_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['PRETENSION_VALUE'].cumsum()

df_pretensions['SURCHARGE_VALUE_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['SURCHARGE_VALUE'].cumsum()

df_pretensions['PRETENSION_VALUE_PENALTY_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['PRETENSION_VALUE_PENALTY'].cumsum()

df_pretensions['SURCHARGE_VALUE_PENALTY_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['SURCHARGE_VALUE_PENALTY'].cumsum()

df_pretensions['UTS_VALUE_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['UTS_VALUE'].cumsum()

df_pretensions['UTS_SURCHARGE_VALUE_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['UTS_SURCHARGE_VALUE'].cumsum()

df_pretensions['HAVE_REQUISITES_OF_APPLICANT_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['HAVE_REQUISITES_OF_APPLICANT'].transform(max)
df_pretensions['REQUIRED_REVIEWS_NEO_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['REQUIRED_REVIEWS_NEO'].transform(max)
df_pretensions['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN'].transform(max)
df_pretensions['CESSION_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['CESSION'].transform(max)

df_pretensions = pd.concat([df_pretensions.reset_index(drop=True), enc.reset_index(drop=True)], axis=1)

df_pretensions.loc[(df_pretensions['PRETENSION_TYPES'] == 'Требование по выплате только неустойки') &
                   (df_pretensions['PRETENSION_VALUE_PENALTY'] == 0) &
                   (df_pretensions['PRETENSION_VALUE_PENALTY'] != df_pretensions['PRETENSION_VALUE']), \
                   'PRETENSION_VALUE_PENALTY'] = df_pretensions.loc[:, 'PRETENSION_VALUE']

df_pretensions.to_parquet('/home/jovyan/old_home/Litigant/data/processed/df_pretensions_enriched.parquet')

# ОТПРАВНАЯ ТОЧКА 1 по претензиям (merge df_claims и df_claims_3 (скрытые претензии) уже был) 
df_pretensions = pd.read_parquet('/home/jovyan/old_home/Litigant/parquet/df_pretensions_enriched.parquet')  
df_pretensions.shape

# ОТПРАВНАЯ ТОЧКА 1 по претензиям (merge df_claims и df_claims_3 (скрытые претензии) уже был) 
df_pretensions = pd.read_parquet('/home/jovyan/old_home/Litigant/data/processed/df_pretensions_enriched.parquet')  
df_pretensions.shape


df = df_victim.query(
'REFUND_FORM_DETAILED in ["Ремонт","Денежная","Денежная. Отказ от ремонта","Ремонт. Смена СТОА"]' 
'and LOSS_DATE_TIME >="2022-01-01"'
'and LOSS_DATE_TIME <="2025-06-30"'
'and LOSS_PROCESS in ["Прямое ОСАГО (с 1 марта 2009)","Традиционное ОСАГО"]'
'and RISK=="Ущерб имуществу третьих лиц"'
        )
df.shape

df.to_parquet('df_pre_final.parquet')
# df_claims.to_parquet('df_claims_pre_final.parquet')
df_claims_payments.to_parquet('df_claims_pre_final.parquet')
df_pretensions.to_parquet('df_pretensions_pre_final.parquet')

df = pd.read_parquet('df_pre_final.parquet')
display(df.shape)
df_claims = pd.read_parquet('df_claims_pre_final.parquet')
display(df_claims.shape)
df_pretensions = pd.read_parquet('df_pretensions_pre_final.parquet')
display(df_pretensions.shape)

#агрегаты по заявителю убытка 
#присоединяем претензии где ФИО Заявитель убытка  выступал в качестве заявителя претензии (Applicant)
df_applicant_agg = df.sort_values(['INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME'])[['APPLICANT_ID','INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME']]\
                     .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                     .merge(df_pretensions[df_pretensions['APPLICANT_PERSON_ID']!='00000000000000000000000000000000']\
                     .drop(columns=['INCIDENT_NUMBER'],axis=1),how='left',left_on='APPLICANT_ID',right_on='APPLICANT_PERSON_ID')

df_applicant_agg = df_applicant_agg[df_applicant_agg['PAYMENT_ORDER_DATE_TIME']>=df_applicant_agg['PRETENSION_GET_DATE']]


sum_cols = ['PRETENSION_VALUE','UTS_VALUE','SURCHARGE_VALUE','UTS_SURCHARGE_VALUE','CESSION','PRETENSION_VALUE_PENALTY','SURCHARGE_VALUE_PENALTY',
            "INSURANCE_TYPE_GROUPS__КАСКО+ГО",
            "INSURANCE_TYPE_GROUPS__ОСАГО",
            "INSURANCE_TYPE_GROUPS__ПРОЧЕЕ",

             'PRETENSION_TYPES__Жалоба по ОСАГО 5+',
             'PRETENSION_TYPES__Запрос документов по делу',
             'PRETENSION_TYPES__Несогласие с суммой выплаты',
             'PRETENSION_TYPES__Отказ от ремонта (смена формы возмещения)',
             'PRETENSION_TYPES__ПРОЧЕЕ',
             'PRETENSION_TYPES__Претензия на принятое решение',
             'PRETENSION_TYPES__Претензия на сроки ремонта и согласования',
             'PRETENSION_TYPES__Претензия по качеству ремонта',
             'PRETENSION_TYPES__Претензия по смене СТОА',
             'PRETENSION_TYPES__Требование по выплате только неустойки',
            
             'PRETENSION_GET_METHOD__Интернет',
             'PRETENSION_GET_METHOD__Лично',
             'PRETENSION_GET_METHOD__ПРОЧЕЕ',
             'PRETENSION_GET_METHOD__Почта',
             'PRETENSION_GET_METHOD__Электронное',
             'PRETENSION_GET_METHOD__None',
            
             'ANSWER_TYPE__Выплата',
             'ANSWER_TYPE__Направлен ответ',
             'ANSWER_TYPE__Направлены документы',
             'ANSWER_TYPE__Отказ в удовлетворении претензии',
             'ANSWER_TYPE__Частичная выплата',
             'ANSWER_TYPE__None'
           ]
mode_cols = ['PRETENSIONTYPE','PRETENSION_GET_METHOD','LOSS_UNIT','LOSS_UNIT_ZONE','ANSWER_TYPE','PRETENSION_TYPES','PRETENSION_KINDS','INSURANCE_TYPE_GROUPS']
agg_dict = {}
nunique_col = ['PRETENSION_NUMBER','PRETENSION_TYPES','PRETENSION_KINDS','INSURANCE_TYPE_GROUPS']

agg_dict = {}

def my_mode(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan  # или np.nan, если хотите

for col in sum_cols:
    agg_dict[f"{col}_sum"] = (col, 'sum')

for col in nunique_col:
    agg_dict[f"{col}_nunique"] = (col, 'nunique')
for col in mode_cols:
    agg_dict[f"{col}_unique_list"] = (col, lambda x: list(set(x.dropna())))

df_applicant_agg = df_applicant_agg.groupby(['APPLICANT_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()

df_pretensions = df_pretensions.merge(pretension_fio_id[['PRETENSION_NUMBER','VICTIM_PERSON_ID']],how='left',on='PRETENSION_NUMBER')


import numpy as np

df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID'] = np.where(
    df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID'] == '00000000000000000000000000000000',
    df_pretensions['VICTIM_OBJECT_OWNER_PERSON_ID'],
    df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID']
)
df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID'] = np.where(
    df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID'] == '00000000000000000000000000000000',
    df_pretensions['VICTIM_PERSON_ID'],
    df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID']
)

#агрегируем для заявителя инфо по претензиям где он был страхователем, исключая те случаи где в претензии страховтель = заявителю, чтобы не учесть дважды
df_applicant_agg_2 = df.sort_values(['INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME'])[['APPLICANT_ID','INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME']]\
                    .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                    .merge(df_pretensions[df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID']!='00000000000000000000000000000000']\
                           .drop(columns=['INCIDENT_NUMBER'],axis=1),how='left',left_on='APPLICANT_ID',right_on='VICTIM_POLICYHOLDER_PERSON_ID')

df_applicant_agg_2 = df_applicant_agg_2[df_applicant_agg_2['VICTIM_POLICYHOLDER_PERSON_ID']!=df_applicant_agg_2['APPLICANT_PERSON_ID']]
df_applicant_agg_2 = df_applicant_agg_2[df_applicant_agg_2['PAYMENT_ORDER_DATE_TIME']>=df_applicant_agg_2['PRETENSION_GET_DATE']]
df_applicant_agg_2 = df_applicant_agg_2.groupby(['APPLICANT_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()

# соединяем все агрегированные датасеты с инфо по заявителю
df_applicant_agg_ = df_applicant_agg.merge(df_applicant_agg_2,on = ['APPLICANT_ID','INCIDENT_NUMBER'],how='outer')


col_for_agg = nunique_col + sum_cols
for col in col_for_agg:
    if col in sum_cols:
        df_applicant_agg_[f'APPLICANT_FTRS_PRET_{col}_SUM'] = df_applicant_agg_[f'{col}_sum_x'].fillna(0) + df_applicant_agg_[f'{col}_sum_y'].fillna(0)
        df_applicant_agg_ = df_applicant_agg_.drop(columns=[f'{col}_sum_x',f'{col}_sum_y'],axis=1)
    if col in nunique_col:
        df_applicant_agg_[f'APPLICANT_FTRS_PRET_{col}_nunique'] =  df_applicant_agg_[f'{col}_nunique_x'].fillna(0) + df_applicant_agg_[f'{col}_nunique_y'].fillna(0)
        df_applicant_agg_ = df_applicant_agg_.drop(columns=[f'{col}_nunique_x',f'{col}_nunique_y'],axis=1)
print(len(df_applicant_agg_[df_applicant_agg_['INCIDENT_NUMBER'].duplicated()]))
# агрегируем моду
mode_cols_result = {}

for col in mode_cols:
    # получаем три столбца из слипания -- без суффикса, _x и _y
    colnames = [ f"{col}_unique_list_x", f"{col}_unique_list_y"]
    def get_mode(row):
        # row - pandas.Series из трех элементов, каждый - список (или nan)
        # собираем в один список все значения
        values = []
        for val in row:
            if isinstance(val, list):
                values.extend(val)
        if values:
            return Counter(values).most_common(1)[0][0]
        else:
            return None

    # применяем по строкам
    df_applicant_agg_[f'APPLICANT_FTRS_PRET_{col}_mode'] = df_applicant_agg_[colnames].apply(get_mode, axis=1)
    df_applicant_agg_ = df_applicant_agg_.drop(columns=colnames, axis=1)

df_applicant_agg_[df_applicant_agg_['INCIDENT_NUMBER']==7996344.0]


# добавляем инфо по претензиям заявителя первичного убытка в основной датасет
print(df.shape)
df = df.merge(df_applicant_agg_,how='left',on=['INCIDENT_NUMBER','APPLICANT_ID'])
df.shape

list_for_fillna = list(df_applicant_agg_.drop(columns=['APPLICANT_ID','INCIDENT_NUMBER']).columns)
list_for_fillna
df[list_for_fillna] = df[list_for_fillna].fillna(0)


#агрегаты по страхователю убытка 
#присоединяем претензии где Страхователь  выступал в качестве заявителя претензии (Applicant)
df_applicant_agg = df.sort_values(['INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME'])[['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME']]\
                    .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                    .merge(df_pretensions[df_pretensions['APPLICANT_PERSON_ID']!='00000000000000000000000000000000']\
                           .drop(columns=['INCIDENT_NUMBER'],axis=1),how='left',left_on='VICTIM_POLICYHOLDER_PERSON_ID',right_on='APPLICANT_PERSON_ID')

df_applicant_agg = df_applicant_agg[df_applicant_agg['PAYMENT_ORDER_DATE_TIME']>=df_applicant_agg['PRETENSION_GET_DATE']]



sum_cols = ['PRETENSION_VALUE','UTS_VALUE','SURCHARGE_VALUE','UTS_SURCHARGE_VALUE','CESSION','PRETENSION_VALUE_PENALTY','SURCHARGE_VALUE_PENALTY',
           'INSURANCE_TYPE_GROUPS__КАСКО+ГО',
            'INSURANCE_TYPE_GROUPS__ОСАГО',
            'INSURANCE_TYPE_GROUPS__ПРОЧЕЕ',
            'PRETENSION_TYPES__Жалоба по ОСАГО 5+',
            'PRETENSION_TYPES__Запрос документов по делу',
            'PRETENSION_TYPES__Несогласие с суммой выплаты',
            'PRETENSION_TYPES__Отказ от ремонта (смена формы возмещения)',
            'PRETENSION_TYPES__ПРОЧЕЕ',
            'PRETENSION_TYPES__Претензия на принятое решение',
            'PRETENSION_TYPES__Претензия на сроки ремонта и согласования',
            'PRETENSION_TYPES__Претензия по качеству ремонта',
            'PRETENSION_TYPES__Претензия по смене СТОА',
            'PRETENSION_TYPES__Требование по выплате только неустойки',
            'PRETENSION_GET_METHOD__Интернет',
            'PRETENSION_GET_METHOD__Лично',
            'PRETENSION_GET_METHOD__ПРОЧЕЕ',
            'PRETENSION_GET_METHOD__Почта',
            'PRETENSION_GET_METHOD__Электронное',
            'PRETENSION_GET_METHOD__None',
            'ANSWER_TYPE__Выплата',
            'ANSWER_TYPE__Направлен ответ',
            'ANSWER_TYPE__Направлены документы',
            'ANSWER_TYPE__Отказ в удовлетворении претензии',
            'ANSWER_TYPE__Частичная выплата',
            'ANSWER_TYPE__None'
           ]
mode_cols = ['PRETENSIONTYPE','PRETENSION_GET_METHOD','LOSS_UNIT','LOSS_UNIT_ZONE','ANSWER_TYPE','PRETENSION_TYPES','PRETENSION_KINDS','INSURANCE_TYPE_GROUPS']
agg_dict = {}
nunique_col = ['PRETENSION_NUMBER','PRETENSION_TYPES','PRETENSION_KINDS','INSURANCE_TYPE_GROUPS']

agg_dict = {}

def my_mode(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan  # или np.nan, если хотите

for col in sum_cols:
    agg_dict[f"{col}_sum"] = (col, 'sum')
for col in nunique_col:
    agg_dict[f"{col}_nunique"] = (col, 'nunique')
for col in mode_cols:
    agg_dict[f"{col}_unique_list"] = (col, lambda x: list(set(x.dropna())))

df_applicant_agg = df_applicant_agg.groupby(['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()


#агрегируем для заявителя инфо по претензиям где он был страхователем, исключая те случаи где в претензии страховтель = заявителю, чтобы не учесть дважды
df_applicant_agg_2 = df.sort_values(['INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME'])[['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME']]\
                    .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                    .merge(df_claims[df_claims['VICTIM_POLICYHOLDER_PERSON_ID']!='00000000000000000000000000000000']\
                           .drop(columns=['INCIDENT_NUMBER'],axis=1),how='left',left_on='VIctimPolicyholderPersonID',right_on='VICTIM_POLICYHOLDER_PERSON_ID')

df_applicant_agg_2 = df_applicant_agg_2[df_applicant_agg_2['VICTIM_POLICYHOLDER_PERSON_ID']!=df_applicant_agg_2['APPLICANT_PERSON_ID']]
df_applicant_agg_2 = df_applicant_agg_2[df_applicant_agg_2['PAYMENT_ORDER_DATE_TIME']>=df_applicant_agg_2['PRETENSION_GET_DATE']]
df_applicant_agg_2 = df_applicant_agg_2.groupby(['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()


# соединяем все агрегированные датасеты с инфо по заявителю
df_applicant_agg_ = df_applicant_agg.merge(df_applicant_agg_2,on = ['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER'],how='outer')


col_for_agg = nunique_col + sum_cols
for col in col_for_agg:
    if col in sum_cols:
        df_applicant_agg_[f'VICTIM_PH_FTRS_PRET_{col}_SUM'] =  df_applicant_agg_[f'{col}_sum_x'].fillna(0) + df_applicant_agg_[f'{col}_sum_y'].fillna(0)
        df_applicant_agg_ = df_applicant_agg_.drop(columns=[f'{col}_sum_x',f'{col}_sum_y'],axis=1)
    if col in nunique_col:
        df_applicant_agg_[f'VICTIM_PH_FTRS_PRET_{col}_nunique'] =  df_applicant_agg_[f'{col}_nunique_x'].fillna(0) + df_applicant_agg_[f'{col}_nunique_y'].fillna(0)
        df_applicant_agg_ = df_applicant_agg_.drop(columns=[f'{col}_nunique_x',f'{col}_nunique_y'],axis=1)
print(len(df_applicant_agg_[df_applicant_agg_['INCIDENT_NUMBER'].duplicated()]))
# агрегируем моду
mode_cols_result = {}

for col in mode_cols:
    # получаем три столбца из слипания  _x и _y
    colnames = [ f"{col}_unique_list_x", f"{col}_unique_list_y"]
    def get_mode(row):
        # row - pandas.Series из трех элементов, каждый - список (или nan)
        # собираем в один список все значения
        values = []
        for val in row:
            if isinstance(val, list):
                values.extend(val)
        if values:
            return Counter(values).most_common(1)[0][0]
        else:
            return None

    # применяем по строкам
    df_applicant_agg_[f'VICTIM_PH_FTRS_PRET_{col}_mode'] = df_applicant_agg_[colnames].apply(get_mode, axis=1)
    df_applicant_agg_ = df_applicant_agg_.drop(columns=colnames, axis=1)

df_applicant_agg_[:2]

# добавляем инфо по претензиям заявителя первичного убытка в основной датасет
print(df.shape)
df = df.merge(df_applicant_agg_,how='left',on=['INCIDENT_NUMBER','VIctimPolicyholderPersonID'])
df.shape

list_for_fillna = list(df_applicant_agg_.drop(columns=['VIctimPolicyholderPersonID','INCIDENT_NUMBER']).columns)
df[list_for_fillna] = df[list_for_fillna].fillna(0)

df_claims_persons['Лицо'] = df_claims_persons['Лицо'].apply(lambda x: x.hex().upper() if x is not None  else np.nan)
df_claims_persons['ПолноеФИОЛица'] = df_claims_persons['ПолноеФИОЛица'].str.upper()
df_claims_persons[:2]

#удаляем дубликаты
print(len(df_claims_persons))
df_claims_persons = df_claims_persons.drop_duplicates()
df_claims_persons = df_claims_persons.sort_values(by=['Представитель','Цессионарий']).drop_duplicates(subset=['ПолноеФИОЛица','НомерИск'],keep='first')
len(df_claims_persons)

# добавляем инфо , что по судам
df_claims = df_claims_persons.rename(columns={'НомерИск':'INCOMING_CLAIM_NUMBER'})\
            .merge(df_claims_.drop(columns=['INCIDENT_NUMBER','LOSS_ID','LOSS_NUMBER','INCOMINGCLAIMID']),how='left',on='INCOMING_CLAIM_NUMBER')


#агрегаты по заявителю убытка , suffixes=('', '_pret')
df_applicant_agg = df.sort_values(['INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME'])[['APPLICANT_ID','INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME']]\
                    .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                    .merge(df_claims[df_claims['Лицо']!='00000000000000000000000000000000'],how='left',left_on='APPLICANT_ID',right_on='Лицо')

df_applicant_agg = df_applicant_agg[df_applicant_agg['PAYMENT_ORDER_DATE_TIME']>=df_applicant_agg['INCOMING_CLAIM_GET_DATE_1']]

df_applicant_agg.columns = df_applicant_agg.columns.str.replace(r'[\s\-]+', '_', regex=True)

start_col = 'CLAIM_ITEM_3_Е_ЛИЦО'
start_idx = df_applicant_agg.columns.get_loc(start_col)
print(list(df_applicant_agg.columns[start_idx:]))


sum_cols = [
    'Представитель',
    'Цессионарий',
    'CLAIMEDMAINDEBT_1',
    'CLAIMEDPLAINTIFFEXAMINATION_1',
    'CLAIMEDCOURTEXAMINATION_1',
    'CLAIMEDREPRESENTATIVEEXPENSES_1',
    'CLAIMEDPERCENTFORUSES_1',
    'CLAIMEDPENALTYFEE_1',
    'CLAIMEDFINE_1',
    'CLAIMEDMORALDAMAGE_1',
    'CLAIMEDOTHEREXPENSES_1',
    'CLAIMEDSTATEDUTY_1',
    'CLAIMEDLOSSCOMMODYVALUE_1',
    'CLAIMEDWEAROUT_1',
    'CLAIMEDVALUEWITHOUTSD_1',
    'CLAIMEDVALUEWITHSD_1',
    'CLAIMEDAMOUNTLOSS_1',
    'CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1',
    'CLAIM_ITEM_3_Е_ЛИЦО',
    'CLAIM_ITEM_НЕ_ПРЕДОСТАВЛЕНИЕ_НА_ОСМОТР',
    'CLAIM_ITEM_НЕКОМПЛЕКТНОСТЬ_ДОКУМЕНТОВ',
    'CLAIM_ITEM_НЕПРИЗНАНИЕ_СТРАХОВЫМ_СЛУЧАЕМ',
    'CLAIM_ITEM_НЕУСТОЙКА',
    'CLAIM_ITEM_ПРОЧЕЕ',
    'CLAIM_ITEM_СПОР_ПО_СУММЕ',
    'CLAIM_ITEM_СУДЕБНЫЕ_РАСХОДЫ_И_САНКЦИИ_ЗА_НЕИСПОЛНЕНИЕ_РЕШЕНИЯ_СУДА',
    'CLAIM_ITEM_ТРАСОЛОГИЯ',
    'CLAIM_ITEM_ФОРМА_ВОЗМЕЩЕНИЯ']
mode_cols = ['ОбращениеКФУОтЗаявителяПоступилоПосредством', 'CLAIM_ITEM','CLAIMORIGIN_1' ]

nunique_col = ['INCOMING_CLAIM_NUMBER','LINK_LOSS_NUMBER']
mean_col = ['CLAIMEDMAINDEBT_1',
        'CLAIMEDPLAINTIFFEXAMINATION_1',
        'CLAIMEDCOURTEXAMINATION_1',
        'CLAIMEDREPRESENTATIVEEXPENSES_1',
        'CLAIMEDPERCENTFORUSES_1',
        'CLAIMEDPENALTYFEE_1',
        'CLAIMEDFINE_1',
        'CLAIMEDMORALDAMAGE_1',
        'CLAIMEDOTHEREXPENSES_1', 
        'CLAIMEDSTATEDUTY_1',
        'CLAIMEDLOSSCOMMODYVALUE_1',
        'CLAIMEDWEAROUT_1',
        'CLAIMEDVALUEWITHOUTSD_1',
        'CLAIMEDVALUEWITHSD_1',
        'CLAIMEDAMOUNTLOSS_1', 
        'CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1']
agg_dict = {}

def my_mode(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan  # или np.nan, если хотите

for col in sum_cols:
    agg_dict[f'APPLICANT_FTRS_COURT_{col}_SUM'] = (col, 'sum')
for col in mean_col:
    agg_dict[f'APPLICANT_FTRS_COURT_{col}_MEAN'] = (col, 'mean')
# for col in count_cols:
#     agg_dict[f"{col}_count"] = (col, 'count')
for col in nunique_col:
    agg_dict[f'APPLICANT_FTRS_COURT_{col}_NUNIQUE'] = (col, 'nunique')
for col in mode_cols:
    agg_dict[f'APPLICANT_FTRS_COURT_{col}_MODE'] = (col, my_mode)


df_applicant_agg = df_applicant_agg.groupby(['APPLICANT_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()

# добавляем инфо по судам заявителя первичного убытка в основной датасет
print(df.shape)
df = df.merge(df_applicant_agg,how='left',on=['INCIDENT_NUMBER','APPLICANT_ID'])
df.shape

list_for_fillna = list(df_applicant_agg.drop(columns=['APPLICANT_ID','INCIDENT_NUMBER']).columns)
list_for_fillna
df[list_for_fillna] = df[list_for_fillna].fillna(0)


#агрегаты по страхователю жертве убытка , suffixes=('', '_pret')
df_applicant_agg = df.sort_values(['INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME'])[['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME','APPLICANT_ID']]\
                    .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                    .merge(df_claims[df_claims['Лицо']!='00000000000000000000000000000000'],how='left',left_on='VICTIM_POLICYHOLDER_PERSON_ID',right_on='Лицо')

df_applicant_agg = df_applicant_agg[df_applicant_agg['PAYMENT_ORDER_DATE_TIME']>=df_applicant_agg['INCOMING_CLAIM_GET_DATE_1']]


df_applicant_agg.columns = df_applicant_agg.columns.str.replace(r'[\s\-]+', '_', regex=True)

start_col = 'CLAIM_ITEM_3_Е_ЛИЦО'
start_idx = df_applicant_agg.columns.get_loc(start_col)
print(list(df_applicant_agg.columns[start_idx:]))


sum_cols = [
    'Представитель',
    'Цессионарий',
    'CLAIMEDMAINDEBT_1',
    'CLAIMEDPLAINTIFFEXAMINATION_1',
    'CLAIMEDCOURTEXAMINATION_1',
    'CLAIMEDREPRESENTATIVEEXPENSES_1',
    'CLAIMEDPERCENTFORUSES_1',
    'CLAIMEDPENALTYFEE_1',
    'CLAIMEDFINE_1',
    'CLAIMEDMORALDAMAGE_1',
    'CLAIMEDOTHEREXPENSES_1',
    'CLAIMEDSTATEDUTY_1',
    'CLAIMEDLOSSCOMMODYVALUE_1',
    'CLAIMEDWEAROUT_1',
    'CLAIMEDVALUEWITHOUTSD_1',
    'CLAIMEDVALUEWITHSD_1',
    'CLAIMEDAMOUNTLOSS_1',
    'CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1',
    'CLAIM_ITEM_3_Е_ЛИЦО',
    'CLAIM_ITEM_НЕ_ПРЕДОСТАВЛЕНИЕ_НА_ОСМОТР',
    'CLAIM_ITEM_НЕКОМПЛЕКТНОСТЬ_ДОКУМЕНТОВ',
    'CLAIM_ITEM_НЕПРИЗНАНИЕ_СТРАХОВЫМ_СЛУЧАЕМ',
    'CLAIM_ITEM_НЕУСТОЙКА',
    'CLAIM_ITEM_ПРОЧЕЕ',
    'CLAIM_ITEM_СПОР_ПО_СУММЕ',
    'CLAIM_ITEM_СУДЕБНЫЕ_РАСХОДЫ_И_САНКЦИИ_ЗА_НЕИСПОЛНЕНИЕ_РЕШЕНИЯ_СУДА',
    'CLAIM_ITEM_ТРАСОЛОГИЯ',
    'CLAIM_ITEM_ФОРМА_ВОЗМЕЩЕНИЯ']
mode_cols = ['ОбращениеКФУОтЗаявителяПоступилоПосредством', 'CLAIM_ITEM','CLAIMORIGIN_1' ]
nunique_col = ['INCOMING_CLAIM_NUMBER','LINK_LOSS_NUMBER','CLAIM_ITEM']
mean_col = ['CLAIMEDMAINDEBT_1',
        'CLAIMEDPLAINTIFFEXAMINATION_1',
        'CLAIMEDCOURTEXAMINATION_1',
        'CLAIMEDREPRESENTATIVEEXPENSES_1',
        'CLAIMEDPERCENTFORUSES_1',
        'CLAIMEDPENALTYFEE_1',
        'CLAIMEDFINE_1',
        'CLAIMEDMORALDAMAGE_1',
        'CLAIMEDOTHEREXPENSES_1', 
        'CLAIMEDSTATEDUTY_1',
        'CLAIMEDLOSSCOMMODYVALUE_1',
        'CLAIMEDWEAROUT_1',
        'CLAIMEDVALUEWITHOUTSD_1',
        'CLAIMEDVALUEWITHSD_1',
        'CLAIMEDAMOUNTLOSS_1', 
        'CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1']

agg_dict = {}

def my_mode(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan  # или np.nan, если хотите

for col in sum_cols:
    agg_dict[f'VICTIM_PH_FTRS_COURT_{col}_SUM'] = (col, 'sum')
for col in mean_col:
    agg_dict[f'VICTIM_PH_FTRS_COURT_{col}_MEAN'] = (col, 'mean')
# for col in count_cols:
#     agg_dict[f"{col}_count"] = (col, 'count')
for col in nunique_col:
    agg_dict[f'VICTIM_PH_FTRS_COURT_{col}_NUNIQUE'] = (col, 'nunique')
for col in mode_cols:
    agg_dict[f'VICTIM_PH_FTRS_COURT_{col}_MODE'] = (col, my_mode)

df_applicant_agg = df_applicant_agg.groupby(['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()


# добавляем инфо по судам заявителя первичного убытка в основной датасет
print(df.shape)
df = df.merge(df_applicant_agg,how='left',on=['INCIDENT_NUMBER','VICTIM_POLICYHOLDER_PERSON_ID'])
df.shape


list_for_fillna = list(df_applicant_agg.drop(columns=['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER']).columns)
list_for_fillna
df[list_for_fillna] = df[list_for_fillna].fillna(0)


col_mode = []
for i in df.columns:
    if '_mode' in i or '_MODE' in i:
        col_mode.append(i)
for c in col_mode:
    df[c] = df[c].replace(0,'nan').astype(str)


df.to_parquet('/home/jovyan/old_home/Litigant/data/raw/pre_final.parquet')

df = pd.read_parquet('/home/jovyan/old_home/Litigant/data/raw/pre_final.parquet')
df.shape

df_claims_inc_agg = df_claims_.groupby('INCIDENT_NUMBER')['INCOMING_CLAIM_NUMBER'].count().reset_index()
df_claims_inc_agg[:2]

df = df.merge(df_claims_inc_agg,how='left',on='INCIDENT_NUMBER')

df['TARGET'] = df['INCOMING_CLAIM_NUMBER'].fillna(0).apply(lambda x: 1 if x > 0 else 0)


df = df.drop(columns=['INCOMING_CLAIM_NUMBER'])

# оставляем только первичный убыток
df_ = df.copy()
df = df.sort_values(['INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME']).drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')
len(df)


quey_calc_agg = \
"""
with tmp as (
	SELECT
		_Period         Период	,
		itl.IncidentNumber AS INCIDENT_NUMBER,
		l.LossNumber 	AS LOSS_NUMBER	,
--		l.RefundFormDetailed,
--		l.LossProcess,
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
		ROW_NUMBER() over (partition by itl.IncidentNumber order by l.LossNumber desc) as rn
	--IncidentNumber, count(*)
	from oisuu81.dbo._InfoRg14746 i
	left join oisuu81_t_losses l on l.LossID = _Fld14747RRef
	LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl on l.LossID=itl.LossID
	where year(l.IssueDate) is not null
--	and l.RefundFormDetailed  in ('Ремонт','Денежная','Денежная. Отказ от ремонта','Ремонт. Смена СТОА')
	and l.LossProcess in ('Прямое ОСАГО (с 1 марта 2009)','Традиционное ОСАГО')
	and Risk = 'Ущерб имуществу третьих лиц'
--	and IncidentNumber = 10825255
)
SELECT *
FROM tmp
WHERE rn = 1
"""
df_calc = pd.read_sql(quey_calc_agg,conn_oisuu)
df_calc


df_calc.loc[df_calc['ПроцентИзноса'] > 50, 'ПроцентИзноса'] = 50


df_calc = df_calc.rename(columns={'Убыток':'LOSS_NUMBER'})


df_calc['SHARE_WORK'] = (df_calc['Работы']/df_calc['СуммаРемонтаБезУчётаИзноса']).round(3)
df_calc = df_calc.rename(columns={'СуммаРемонтаБезУчётаИзноса':'AMOUNT_REPAIR','ПроцентИзноса':'SHARE_WEAROUT'})


df = df.merge(df_calc[['INCIDENT_NUMBER','SHARE_WORK','AMOUNT_REPAIR','SHARE_WEAROUT']],how='left',on='INCIDENT_NUMBER')
df['AMOUNT_REPAIR'].isna().mean()


df['ISSUE_YEAR'] = df['ISSUE_DATE'].dt.year


df = df.drop(columns=['ISSUE_YEAR'])


df['FLAG_APPLICANT_SAME_VICTIM_PH'] = df.apply(lambda row: 1 if row['APPLICANT_ID']==row['VICTIM_POLICYHOLDER_PERSON_ID'] else 0, axis=1)
df['FLAG_APPLICANT_SAME_VICTIM_PH'].mean()


df['TOTAL_COUNT_PRETENSION'] = df['APPLICANT_FTRS_PRET_PRETENSION_NUMBER_nunique'] + df['VICTIM_PH_FTRS_PRET_PRETENSION_NUMBER_nunique']
df['TOTAL_COUNT_COURT'] = df['APPLICANT_FTRS_COURT_INCOMING_CLAIM_NUMBER_NUNIQUE'] + df['VICTIM_PH_FTRS_COURT_INCOMING_CLAIM_NUMBER_NUNIQUE']


target_2_ = \
"""
SELECT 
		 psr.[Номер_инциндента]
        ,sum(l.EMRValue) as EMRValue
		,sum(psr.[Выплата_по_основному_убытку]) as Выплата_по_основному_убытку
		,sum(psr.[Сумма_выплат_по_претензиям]) as Сумма_выплат_по_претензиям
		,sum(psr.[Общая_сумма_заявленных_требований_ФУ]) as Общая_сумма_заявленных_требований_ФУ
		,sum(psr.[Сумма_взыскано_по_ФУ]) as Сумма_взыскано_по_ФУ
		,sum(psr.[Общая_сумма_заявленных_требований_ИСК]) as Общая_сумма_заявленных_требований_ИСК
		,sum(psr.[Суммы_взыскано_по_иску]) as Суммы_взыскано_по_иску
		,sum(psr.[Сумма_утс]) as Сумма_утс
		,sum(psr.[Сумма_износа_по_калькуляции_инцидент]) as Сумма_износа_по_калькуляции_инцидент
		,sum(psr.[Выплаченный_износ_инцидент]) as Выплаченный_износ_инцидент
		,sum(psr.[Доп_расходы_инцидент]) as Доп_расходы_инцидент
		,sum(psr.[Взысканный_износ_ФУ]) as Взысканный_износ_ФУ
		,sum(psr.[Взысканный_износ_ИСК]) as Взысканный_износ_ИСК
  FROM [OISUU_report].[Datamart].[oisuu81_t_ПСР] as psr
  LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Losses] as l on l.LossNumber=psr.Номер_убытка
  group by [Номер_инциндента]
"""
target_2 = pd.read_sql(target_2_, conn_oisuu)
target_2.shape


conn_oisuu.close()


target_2.to_parquet('/home/jovyan/old_home/Litigant/data/raw/target_2.parquet')


target_2 = pd.read_parquet('/home/jovyan/old_home/Litigant/data/raw/target_2.parquet')
target_2


for col in target_2.columns:
    target_2[col] = target_2[col].fillna(0)


df = df.merge(target_2, how='left', left_on='INCIDENT_NUMBER', right_on='Номер_инциндента')
df.shape


df['TARGET_2'] = df['Суммы_взыскано_по_иску'] + df['Сумма_взыскано_по_ФУ']


df['TARGET_2'] = df['TARGET_2'].apply(lambda x: 1 if x > 0 else 0)
df['TARGET_2'].value_counts(dropna=False)



target_3_pretensions = \
"""
SELECT 
       itl.IncidentNumber
      ,sum(p.[SurchargeValue]) as SurchargeValue_cumsum_by_incident
      ,sum(p.[UTSSurchargeValue]) as UTSSurchargeValue_cumsum_by_incident
  FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS p
  LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl on itl.LossID=p.LossID
  WHERE InsuranceTypeGroups = 'ОСАГО'
  and PretensionType in ('Несогласие с суммой выплаты','Претензия на принятое решение')
  group by IncidentNumber
--  and LossNumber = 1304948
--  order by PretensionGetDate
"""
target_3_pretensions = pd.read_sql(target_3_pretensions, conn_oisuu)
target_3_pretensions.shape


conn_oisuu.close()

target_3_pretensions.to_parquet('/home/jovyan/old_home/Litigant/data/raw/target_3_pretensions.parquet')


target_3_pretensions = pd.read_parquet('/home/jovyan/old_home/Litigant/data/raw/target_3_pretensions.parquet')
target_3_pretensions = target_3_pretensions.rename(columns=rename_dict)
target_3_pretensions

target_3_pretensions_all = \
"""
SELECT 
       itl.IncidentNumber
      ,sum(p.[SurchargeValue]) as SurchargeValue_cumsum_by_incident_all
      ,sum(p.[UTSSurchargeValue]) as UTSSurchargeValue_cumsum_by_incident_all
  FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS p
  LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl on itl.LossID=p.LossID
  WHERE InsuranceTypeGroups = 'ОСАГО'
  group by IncidentNumber
"""
target_3_pretensions_all = pd.read_sql(target_3_pretensions_all, conn_oisuu)
target_3_pretensions_all.shape

conn_oisuu.close()


target_3_pretensions_all.to_parquet('/home/jovyan/old_home/Litigant/data/raw/target_3_pretensions_all.parquet')


target_3_pretensions_all = pd.read_parquet('/home/jovyan/old_home/Litigant/data/raw/target_3_pretensions_all.parquet')
target_3_pretensions_all = target_3_pretensions_all.rename(columns=rename_dict)
target_3_pretensions_all


target_3_claims = \
"""
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
FROM [OISUU_report].[Datamart].[oisuu81_t_IncomingClaimNewLogicByInst] as icnl
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl on itl.LossNumber=icnl.LinkLossNumber
LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Losses] as l on l.LossNumber=itl.LossNumber
WHERE (ClaimOrigin in ('ВСК', 'Обращение к ФУ') or ClaimOrigin is null)
  AND (ClaimItem != 'ВСК - 3 лицо' or ClaimItem is null)
  AND l.LossProcess IN ('Прямое ОСАГО (с 1 марта 2009)', 'Традиционное ОСАГО')
"""
target_3_claims = pd.read_sql(target_3_claims, conn_oisuu)
target_3_claims.shape

conn_oisuu.close()

target_3_claims.to_parquet('/home/jovyan/old_home/Litigant/data/raw/target_3_claims.parquet')

target_3_claims = pd.read_parquet('/home/jovyan/old_home/Litigant/data/raw/target_3_claims.parquet')
target_3_claims = target_3_claims.rename(columns=rename_dict)
target_3_claims


target_3_claims.columns = target_3_claims.columns.str.upper()
target_3_claims = target_3_claims.rename(columns=rename_dict)


df_base = target_3_claims[['INCIDENT_NUMBER', 'LOSS_ID', 'LOSS_NUMBER',
                           'INCOMINGCLAIMID', 'INCOMING_CLAIM_NUMBER',
                           'LINK_LOSS_NUMBER']].drop_duplicates(['INCOMING_CLAIM_NUMBER'])
df_base = df_base.sort_values(by=['INCOMING_CLAIM_NUMBER']).reset_index(drop=True)


# Отбираем только необходимые колонки (относящиеся к судам)
df_sorted = target_3_claims.loc[:, ['LOSS_NUMBER'] + list(target_3_claims.iloc[:, 6:].columns)]
# Удаляем ненужные колонки
df_sorted = df_sorted.drop(['INSTID', 'LINK_LOSS_NUMBER'], axis=1)
# Сортируем датафрейм по номеру иска и дате выставления, чтобы инстанции шли по временному порядку
df_sorted = df_sorted.sort_values(by=['INCOMING_CLAIM_NUMBER', 'CLAIMEDVALUEPERIOD']).reset_index(drop=True)
# Первое вхождение в какую-либо инстанцию (будь то 6, или 1 инстанции) в отдельно взятом номере иска == перва инстанция
# для данного иска (cumcount)
df_sorted['Instance'] = df_sorted.groupby(['INCOMING_CLAIM_NUMBER']).cumcount() + 1
# Сводная таблица по искам и инстанциям в строку
df_pivot = df_sorted.pivot(index=['INCOMING_CLAIM_NUMBER'], columns='Instance')
# Колонки переименовываем в соответствии с номером инстанции (ClaimedValueWithSD_1, ClaimedValueWithSD_2 И т.д.)
df_pivot.columns = [f'{col[0]}_{int(col[1])}' for col in df_pivot.columns]
df_pivot = df_pivot.reset_index()
df_pivot


target_3_claims = df_base.merge(df_pivot, how='left', on=['INCOMING_CLAIM_NUMBER'])
target_3_claims


target_3_claims = target_3_claims[['INCIDENT_NUMBER',
                                   'RECOVEREDMAINDEBT_1', 'RECOVEREDWEAROUT_1', 'RECOVEREDLOSSCOMMODYVALUE_1',
                                   'RECOVEREDMAINDEBT_2', 'RECOVEREDWEAROUT_2', 'RECOVEREDLOSSCOMMODYVALUE_2',
                                   'RECOVEREDMAINDEBT_3', 'RECOVEREDWEAROUT_3', 'RECOVEREDLOSSCOMMODYVALUE_3',
                                   'RECOVEREDMAINDEBT_4', 'RECOVEREDWEAROUT_4', 'RECOVEREDLOSSCOMMODYVALUE_4',
                                   'RECOVEREDMAINDEBT_5', 'RECOVEREDWEAROUT_5', 'RECOVEREDLOSSCOMMODYVALUE_5',
                                  
                                   'RECOVEREDVALUEWITHSD_1','RECOVEREDVALUEWITHSD_2','RECOVEREDVALUEWITHSD_3',
                                   'RECOVEREDVALUEWITHSD_4','RECOVEREDVALUEWITHSD_5']].fillna(0)


target_3_claims = target_3_claims.groupby('INCIDENT_NUMBER').sum().reset_index(drop=False)
target_3_claims


df = df.merge(target_3_claims, how='left', left_on='INCIDENT_NUMBER', right_on='INCIDENT_NUMBER')
df.shape


df = df.merge(target_3_pretensions, how='left', left_on='INCIDENT_NUMBER', right_on='INCIDENT_NUMBER')
df.shape


df = df.merge(target_3_pretensions_all, how='left', left_on='INCIDENT_NUMBER', right_on='INCIDENT_NUMBER')
df.shape


import pandas as pd
import numpy as np

# Список колонок в порядке от высшей инстанции к низшей
cols = ['RECOVEREDMAINDEBT_1', 'RECOVEREDWEAROUT_1', 'RECOVEREDLOSSCOMMODYVALUE_1',
        'RECOVEREDMAINDEBT_2', 'RECOVEREDWEAROUT_2', 'RECOVEREDLOSSCOMMODYVALUE_2',
        'RECOVEREDMAINDEBT_3', 'RECOVEREDWEAROUT_3', 'RECOVEREDLOSSCOMMODYVALUE_3',
        'RECOVEREDMAINDEBT_4', 'RECOVEREDWEAROUT_4', 'RECOVEREDLOSSCOMMODYVALUE_4',
        'RECOVEREDMAINDEBT_5', 'RECOVEREDWEAROUT_5', 'RECOVEREDLOSSCOMMODYVALUE_5']

# Функция для выбора первого ненулевого и не-NaN значения
def get_last_nonzero_or_valid(row):
    for col in reversed(cols):
        val = row[col]
        # Если значение не NaN и не 0 (если 0 считается "пустым")
        if pd.notna(val) and val != 0:
            return val
    # Если все пустые или нули — вернуть NaN или 0, как вам нужно
    return np.nan  # или 0, если предпочитаете

df['TARGET_3_SEV'] = df.apply(get_last_nonzero_or_valid, axis=1)


df[['TARGET_3_SEV', 'SurchargeValue_cumsum_by_incident', 'UTSSurchargeValue_cumsum_by_incident']] = df[['TARGET_3_SEV', 'SurchargeValue_cumsum_by_incident', 'UTSSurchargeValue_cumsum_by_incident']].fillna(0)


df['TARGET_3_SEV'] = df['TARGET_3_SEV'] + df['SurchargeValue_cumsum_by_incident'] + df['UTSSurchargeValue_cumsum_by_incident']
df['TARGET_3_SEV'].value_counts(dropna=False)


df['TARGET_3_FREQ'] = df['TARGET_3_SEV'].apply(lambda x: 1 if x > 0 else 0)
df['TARGET_3_FREQ'].value_counts(dropna=False)


df['VICTIM_VEHICLE_IS_JAPAN'] = df['VICTIM_VEHICLE_IS_JAPAN'].astype(str)


df = df[df['VICTIM_POLICYHOLDER_TYPE'] == 'Физ. Лицо'].reset_index(drop=True)
df.shape


df.to_parquet('/home/jovyan/old_home/Litigant/data/processed/df_final_3.parquet')


df = pd.read_parquet('/home/jovyan/old_home/Litigant/data/processed/df_final_3.parquet')
df.shape


df['TARGET_2'] = (df['Сумма_выплат_по_претензиям'] + df['Сумма_взыскано_по_ФУ'] + df['Суммы_взыскано_по_иску']).fillna(0)
df['TARGET_2'] = df['TARGET_2'].apply(lambda x: 1 if x > 0 else 0).astype(int)
df['TARGET_2'].value_counts(dropna=False)