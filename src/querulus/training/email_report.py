"""Рассылка результатов OutBoxML для Querulus.

OutBoxML EMailDSResult передаёт AllModelsConfig в DataSetsManager как путь к файлу.
Обход — только в коде Querulus, без правок библиотеки.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from loguru import logger

from outboxml.core.email import EMailDSResult
from outboxml.core.enums import ResultNames
from outboxml.core.pydantic_models import AllModelsConfig
from outboxml.datasets_manager import DataSetsManager
from outboxml.export_results import ResultExport


def _datasets_manager_for_result(
    model_config: Union[str, AllModelsConfig],
    external_config,
) -> DataSetsManager:
    if isinstance(model_config, AllModelsConfig):
        ds = DataSetsManager(
            config_name=model_config.model_dump(),
            external_config=external_config,
        )
        ds._all_models_config = model_config
    else:
        ds = DataSetsManager(
            config_name=model_config,
            external_config=external_config,
        )
    return ds


class QuerulusEMailDSResult(EMailDSResult):
    """EMailDSResult с корректной передачей AllModelsConfig в DataSetsManager."""

    def _metrics_description(self):
        self.mail.add_text(
            "Характеристики моделей:",
            n_line_breaks=1,
        )
        df = pd.DataFrame()

        for key in self._ds_manager_result.keys():
            model_config = self._ds_manager_result[key].config
            ds = _datasets_manager_for_result(model_config, self.config)
            res_export = ResultExport(ds_manager=ds, config=self.config)
            res_export.result = self._ds_manager_result[key]
            try:
                df1 = res_export.metrics_df(
                    model_name=key,
                    train_test="train",
                    metrics_dict=self._ds_manager_result[key].metrics,
                )
                df1 = df1.reset_index()[["index", "full"]]
                df1.columns = [ResultNames.metric, ResultNames.new_result_train]
                df2 = ResultExport(ds_manager=ds, config=self.config).metrics_df(
                    model_name=key,
                    train_test="test",
                    metrics_dict=self._ds_manager_result[key].metrics,
                )
                df2 = df2.reset_index()[["index", "full"]]
                df2.columns = [ResultNames.metric, ResultNames.new_result_test]

                metrics_df = pd.concat([df1, df2["Новая модель||Тестовая выборка"]], axis=1)
                metrics_df["Имя модели"] = key
                df = pd.concat([df, metrics_df])
            except Exception as exc:
                logger.error(exc)

        self.mail.add_pandas_table(
            df,
            params=dict(text_align="right", font_family="sans-serif", width="180px"),
        )
