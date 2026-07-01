import dataclasses
import datetime
import json
from json import JSONDecodeError
from typing import Union

import oracledb
import pandas as pd
import oracledb as cx_Oracle
import requests
from loguru import logger
from mldataworker.extractors.extractor import Extractor, spark_to_pandas, RTDMExtractor

import pandas as pd
import urllib
from sqlalchemy import create_engine, event
import warnings
warnings.filterwarnings('ignore')
import json
import oracledb as cx_Oracle

from tqdm import tqdm
tqdm.pandas()

import json
import re
import ast
import pandas as pd
import numpy as np


class DataExtractor(Extractor):

    def __init__(self, dataset_name: str, *params):
        super().__init__(*params)
        self.dataset_name = dataset_name

    def extract_dataset(self):
        # return spark_to_pandas(source=self.dataset_name)
        filials = ['Архангельский',
                   'Владимирский',
                   'Кемеровский',
                   'Курский',
                   'Магнитогорский',
                   'Марийский',
                   'Мурманский',
                   'Омский',
                   'Пермский',
                   'Петропавловск-Камчатский',
                   'Уфимский',
                   'Ярославский']
        df = pd.read_parquet(self.dataset_name)
        df = df[df['FILIAL'].isin(filials)]
        
        return df

@dataclasses.dataclass
class Response:
    text: str
    status_code: int


class Request:
    def prepare(self, ):
        pass


class FeatureAPI:
    def get(self, request: Request) -> requests.Response:
        pass


class UULogExtractor(Extractor):
    def __init__(self, config,
                 pickle_name: str,
                 log_period_days: int = 14,
                 prod_numbers=[1, 2, 3, 8],
                 conversed: bool = False,
                 response: bool = False,
                 retro: bool = False,
                 ):
        super().__init__()
        self.log_period_days = log_period_days
        self._pickle_name = pickle_name
        self.config = config
        self.load_config_from_env = False
        self.prod_numbers = prod_numbers
        self.__current_date = datetime.datetime.now()
        self.connection_config = config
        self._host = config.database_host
        self._port = config.database_port
        self._service_name = config.database_service_name
        self._dsn = f"{self._host}.vsk.ru:{self._port}/{self._service_name}"
        self.__connection = None
        self.response = response
        self.conversed = conversed
        self.retro = retro
        self.model_features = []

    def extract_dataset(self) -> pd.DataFrame:

        df = pd.read_parquet('/home/jovyan/old_home/Litigant/monitoring/logs_test_prod_for_DataDrift.parquet')
        df = df[df['EVENT_CREATED_BY_GIBDD_FLAG'] != 'Да']
        
        event_date_col = "EVENT_DATE"
        loss_date_col = "PAYMENT_ORDER_DATE_TIME"
    
        event_dt = pd.to_datetime(df[event_date_col], errors="coerce")
        loss_dt = pd.to_datetime(df[loss_date_col], errors="coerce")

        df["APPLY_DELAY"] = (loss_dt.dt.normalize() - event_dt.dt.normalize()).dt.days.fillna(-1).astype(int)
         
        return df
