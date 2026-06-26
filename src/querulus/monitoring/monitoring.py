from mldataworker.core.utils import ResultPickle
from mldataworker.monitoring_manager import MonitoringManager, MonitoringReport, MonitoringResult

import pandas as pd
import numpy as np

from mldataworker.core.email import EMailMonitoring
from mldataworker.extractors.extractor import Extractor, RTDMExtractor, spark_to_pandas
from mldataworker.datadrift import DataDrift as BaseDataDrift
from mldataworker.metrics.metrics import BaseMetric
from mldataworker.target_extrapolation import TargetModel

from querulus.monitoring.extractors import DataExtractor, UULogExtractor
from sqlalchemy import create_engine


class DataDrift(BaseDataDrift):
    def __init__(self, *args, dif_len_string: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.dif_len_string = dif_len_string

    def _stringify_diff_values(self, values: set) -> str:
        prepared = [(v.item() if hasattr(v, "item") else v) for v in values]
        prepared = sorted(prepared, key=lambda x: str(x))
        return str(prepared)[:self.dif_len_string]

    def _drift_calc(self, base: pd.Series, control: pd.Series, label: str, type: str = 'CATEGORICAL') -> dict:
        base_unique = set(base.unique())
        control_unique = set(control.unique())

        result = {
            'TYPE': type,
            "col": label,
            "NaN_train": base.isna().mean(),
            "NaN_test": control.isna().mean(),
            "uniq_train": base.nunique(dropna=False),
            "uniq_test": control.nunique(dropna=False),
            "mode_train": base.mode(dropna=False)[0],
            "mode_test": control.mode(dropna=False)[0],
            "mean_train": np.nan if type == 'CATEGORICAL' else base.mean(),
            "mean_test": np.nan if type == 'CATEGORICAL' else control.mean(),
        }

        result["dif_train"] = self._stringify_diff_values(base_unique - control_unique)
        result["dif_test"] = self._stringify_diff_values(control_unique - base_unique)
        return result


class UUEMailMonitoring(EMailMonitoring):
    def __init__(self, config):
        super().__init__(config)

    def success_mail(self, monitoring_result):
        self.base_mail(header_name=monitoring_result.group_name + str(' Monitoring'), text='Отчет по запуску мониторинга модели сутяжничества')
        self.mail.add_text(
            "Произведен расчёт датадрифта котировок по отношению к датасету, на котором модель обучалась",
            n_line_breaks=1,
        )

        self.mail.add_text(
            "Обнаружен дрифт в фичах, выше заданного значения PSI=0.3:",
            n_line_breaks=1,
        )

        drift_df = self._prepare_drift_df(monitoring_result.report)

        self.mail.add_pandas_table(drift_df,
                                   params=dict(text_align='right', font_family='sans-serif', width="180px"),
                                   )
        self.mail.add_text(
            "Полные результаты выложены в Grafana: https://grafana.vsk.ru/d/afkhng0p8bym8f/querulus",
            n_line_breaks=2,
        )
        self.mail.add_text('PSI(Population Stability Index): Изменение распределения данных.', n_line_breaks=1)
        self.mail.add_text('KL(Kullback - Leibler Divergence): Различие между распределениями.',  n_line_breaks=1)

        self.send()

    def _prepare_drift_df(self, df):
        if df is None:
            print('Нет данных для отчета')
            return pd.DataFrame()
        else:
            alarm_df = df.loc[df['PSI'] > 0.3]
            try:
                df_to_send = alarm_df[
                    ['model_name', 'col', 'PSI', 'KL', 'NaN_train', 'NaN_test', 'mode_train', 'mode_test',
                     'mean_train', 'mean_test']].sort_values(by='PSI',
                                                             ascending=False)

            except KeyError:
                df_to_send = alarm_df[['model_name', 'PSI', 'KL', 'NaN_train', 'NaN_test', 'mode_train', 'mode_test',
                                       'mean_train', 'mean_test']].sort_values(by='PSI',
                                                                               scending=False).reset_index()


            df_to_send = df_to_send.rename(columns={'NaN_train': 'NaN_dataset', 'NaN_test': 'NaN_prod',
                                                    'mode_train': 'mode_dataset', 'mode_test': 'mode_prod',
                                                    'mean_train': 'mean_dataset', 'mean_test': 'mean_prod'})
            return df_to_send


class UUMonitoringReport(MonitoringReport):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def make_report(self, monitoring_result: MonitoringResult) -> pd.DataFrame:
        report = pd.DataFrame()
        for key in monitoring_result.datadrift.keys():
            df_result = monitoring_result.datadrift[key].copy()
            df_result['model_name'] = key
            report = pd.concat([report, df_result])

        for column in report.columns:
            try:
                report[column] = report[column].astype('float')
            except:
                report[column] = report[column].astype(str)
        report['model_version'] = self.model_name
        return report


def monitoring(pickle_name: str,
               monitoring_config: str,
               dataset_name: str,
               models_config: str,
               config,
               prod_numbers=[1, 2, 3, 8],
               log_period_days: int = 10,
               check_model: bool = False,
               second_pickle_name: str = None,
               second_models_config: str = None,
               response: bool = False,
               retro: bool=False,
               ):

    grafana_connection = None
    if getattr(config, "grafana_sqlalchemy_url", ""):
        grafana_connection = create_engine(config.grafana_sqlalchemy_url)

    m = MonitoringManager(monitoring_config=monitoring_config,
                          models_config=models_config,
                          external_config=config,
                          data_extractor=DataExtractor(dataset_name=dataset_name),
                          logs_extractor=UULogExtractor(config=config,
                                                           pickle_name=pickle_name,
                                                           # prod_numbers=prod_numbers,
                                                           log_period_days=log_period_days,
#                                                           conversed=conversed,
                                                           response=response,
                                                           retro=retro,
                                                           ),
                          monitoring_report=UUMonitoringReport(model_name=pickle_name),
                          datadrift_interface=DataDrift(columns_to_exclude=['AQ', 'AY', 'POLICY_INSURANCE_TERM',
                                                                            'DRIVER_MARITAL_STATUS',
                                                                            'POLICY_LDU_TYPE',
                                                                            'IS_LEGACY'],
                                                    full_calc=True,
                                                    use_mldataworker_preprocessor=False,
                                                    dif_len_string=100),
#                          email=UUEMailMonitoring(config=config),
#                          grafana_connection=grafana_connection,
                          )
#    print(m._ds_manager.dataset['LOSS_UNIT_ZONE'].nunique())
    m.review(check_model=check_model)
#    print("datadrift keys:", list(m.result.datadrift.keys()))
    m.result.datadrift['querulus_rg_3'].to_excel('results_rg.xlsx')