import pandas as pd

from querulus.monitoring import config
from querulus.monitoring.monitoring import monitoring



def main(pickle_name: str,
         config,
         monitoring_config: str,
         dataset_name: str,
         models_config: str,
         prod_numbers: list[int] = [1, 2],
         log_period_days: int = 14,
         only_conversed: bool = False,
         response: bool = False,
         retro:bool=False
         ):

    monitoring(pickle_name=pickle_name,
               config=config,
               monitoring_config=monitoring_config,
               dataset_name=dataset_name,
               models_config=models_config,
               prod_numbers=prod_numbers,
               log_period_days=log_period_days,
#               conversed=only_conversed,
               response=response,
               retro=retro)


if __name__ == "__main__":
    main(pickle_name='/home/jovyan/old_home/Litigant/integration/results/querulus_ansamble_2026_22_04_v1.pickle',
         models_config='/home/jovyan/old_home/Litigant/configs/config_rg_3.json',
         monitoring_config='/home/jovyan/old_home/Litigant/monitoring/datadrift_rg_calc/monitoring_config.json',
         config=config,
         log_period_days=7,
         only_conversed=False,
         dataset_name='/home/jovyan/old_home/Litigant/data/processed/X_train_sev_for_DataDrift.parquet',  # Дописать свою таблицу как заработает hadoop
         response=False,
         retro=False
         )