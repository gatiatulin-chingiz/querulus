"""
Интеграционные проверки примера `querulus`.

Что проверяем
    - Совпадение предсказаний сервиса с предсказаниями, сохранёнными в обучающем датасете
      (после выравнивания строк по ключу).
    - Предобработка входного вектора ("" -> NaN, UPPER для строк, derived-поля дат).

Примечания по окружению
    Пути к датасетам и имена колонок предсказаний берутся из `examples/querulus/integration/config.py`.
    Эти тесты ориентированы на запуск в окружении, где доступны:
        - файл обучающего датафрейма (train_df),
        - pickle-группа моделей, с которой работает `main_predict`.

Важно
    Это именно интеграционные тесты "сквозь" слой препроцессинга и инференса.
    Они не подменяют модели и не мокают `prepare_dataset`.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .main import (
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
    main_predict,
    preprocessing_and_validate_vector,
)


def _require_config_str(attr_name: str, error_message: str) -> str:
    value = (getattr(config, attr_name, None) or "").strip()
    if not value:
        raise RuntimeError(error_message)
    return value


def require_outboxml_model_group() -> str:
    """
    Возвращает необходимую группу моделей для интеграционных тестов.

    Исключение:
        RuntimeError: если OUTBOXML_MODEL_GROUP не установлен.
    """
    return _require_config_str(
        "outboxml_model_group",
        "OUTBOXML_MODEL_GROUP обязателен для интеграционных тестов",
    )


def require_outboxml_train_df_path() -> str:
    """Возвращает обязательный путь к тренировочному датафрейму для интеграционных тестов."""
    return _require_config_str(
        "outboxml_train_df_path",
        "OUTBOXML_TRAIN_DF_PATH обязателен для интеграционных тестов",
    )


def require_outboxml_preds_cols() -> tuple[str, str]:
    """Возвращает имена обязательных колонок с предсказаниями (cf, rg) для интеграционных тестов."""
    cf = _require_config_str("outboxml_preds_cf_col", "OUTBOXML_PREDS_CF_COL не заполнен")
    rg = _require_config_str("outboxml_preds_rg_col", "OUTBOXML_PREDS_RG_COL не заполнен")
    return cf, rg

def _load_training_dataframe(path: Path) -> pd.DataFrame:
    """
    Загружает датафрейм по расширению файла.

    Поддерживаемые форматы:
        - .csv
        - .xlsx/.xls
        - .pkl/.pickle
        - .parquet

    Args:
        path: путь к файлу таблицы.

    Returns:
        pandas.DataFrame: загруженная таблица.

    Raises:
        ValueError: если расширение неизвестно (чтобы явно сигнализировать о конфигурации).
    """
    suffix = path.suffix.lower()
    # Выбор reader по расширению — так тесты поддерживают несколько форматов датасетов,
    # не завязываясь на один способ хранения.
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported train dataframe format: {path.name}")


class TestPredictionsRoundtrip(unittest.TestCase):
    def test_predictions_match_training_dataset_columns(self):
        """
        Сквозной тест совпадения предсказаний сервиса с обучающим датасетом.

        Сценарий:
            - Загружаем train_df и raw_df из путей, заданных в `config`.
            - Выравниваем строки train/raw по ключу (LOSS_NUMBER или INCIDENT_NUMBER).
            - Вызываем `main_predict` и сравниваем предсказания (округляя как в контракте).

        Цель:
            Обнаружить регрессии в инференсе при изменении кода сервиса, конфига или моделей.
        """
        train_df_path = Path(require_outboxml_train_df_path())
        # Явно фейлим тест с понятным сообщением, если конфиг указывает на несуществующие файлы.
        if not train_df_path.exists():
            self.fail(f"Train dataframe not found: {train_df_path}")

        # Читаем train_df:
        # - "каноническая" таблица, по которой обучались модели и где сохранены предсказания.
        train_df = _load_training_dataframe(train_df_path)

        # Названия колонок с предсказаниями определяются конфигом, чтобы тесты не были захардкожены
        # под один конкретный экспорт.
        preds_cf_col, preds_rg_col = require_outboxml_preds_cols()
        for col in [preds_cf_col, preds_rg_col]:
            if col not in train_df.columns:
                self.fail(f"Missing column in training dataframe: {col}")

        group_name = require_outboxml_model_group()

        # Считаем предсказания сервиса на тех же строках, но без колонок с "эталонными"
        # предсказаниями из train_df.
        input_df = train_df.drop(columns=[preds_cf_col, preds_rg_col], errors="ignore")
        input_records = input_df.to_dict(orient="records")
        result = main_predict(group_name, input_records)

        # Сервис возвращает списки предсказаний в JSON.
        service_cf = result["main_response"]["result"]["classification_proba"]
        service_rg = result["main_response"]["result"]["regression_predictions"]

        expected_cf = train_df[preds_cf_col].to_numpy(dtype=float)
        expected_rg = train_df[preds_rg_col].to_numpy(dtype=float)

        # Сервис по контракту округляет до 2 знаков — сравниваем уже округлённые значения.
        service_cf_rounded = np.round(np.asarray(service_cf, dtype=float), 2)
        service_rg_rounded = np.round(np.asarray(service_rg, dtype=float), 2)
        expected_cf_rounded = np.round(expected_cf.astype(float), 2)
        expected_rg_rounded = np.round(expected_rg.astype(float), 2)

        try:
            np.testing.assert_allclose(service_cf_rounded, expected_cf_rounded, rtol=0, atol=1e-9)
            np.testing.assert_allclose(service_rg_rounded, expected_rg_rounded, rtol=0, atol=1e-9)
        except AssertionError as exc:
            self.fail(f"Predictions mismatch.\n{exc}")


INCOMING_VECTOR = [
    {
        "RECIEVE_METHOD": "Не оповещать",
        "APPLICANT_FORM": "Представитель (не автоюрист)",
        "APPLICANT_AGE": 52,
        "FILIAL": "Белгородский",
        "EVENT_CREATED_BY_GIBDD_FLAG": "Да",
        "VICTIM_MAX_WEIGHT": 18000,
        "GUILTY_CAPACITY_ENGINE": 2,
        "VICTIM_VEHICLE_AGE": 2,
        "EVENT_YEAR": 2026,
        "LOSS_UNIT_ZONE": "Зона ф-ла Белгородский",
        "VICTIM_VEHICLE_COUNTRY": "Россия",
        "EVENT_DATE": "2026-03-23T08:00:00",
        "PAYMENT_ORDER_DATE_TIME": "2026-03-26T00:00:00",
        "LOSS_NUMBER": 11365778,
        "AMOUNT_REPAIR": "",
        "VICTIM_VIHICLE_BRAND": "ЛиАЗ",
        "VICTIM_VEHICLE_CATEGORY": "Иное",
        "INCIDENT_NUMBER": 11365778,
    },
    {
        "RECIEVE_METHOD": "SMS",
        "APPLICANT_FORM": "Представитель (по доверенности)",
        "APPLICANT_AGE": 57,
        "FILIAL": "ВСК-Москва",
        "EVENT_CREATED_BY_GIBDD_FLAG": "Нет",
        "VICTIM_MAX_WEIGHT": 1722,
        "GUILTY_CAPACITY_ENGINE": 1,
        "VICTIM_VEHICLE_AGE": 14,
        "EVENT_YEAR": 2025,
        "LOSS_UNIT_ZONE": "Зона Московского региона",
        "VICTIM_VEHICLE_COUNTRY": "Россия",
        "EVENT_DATE": "2025-06-26T08:30:00",
        "PAYMENT_ORDER_DATE_TIME": "2025-07-03T00:00:00",
        "LOSS_NUMBER": 10826064,
        "AMOUNT_REPAIR": 169389,
        "VICTIM_VIHICLE_BRAND": "Skoda",
        "VICTIM_VEHICLE_CATEGORY": "Средний 7+",
        "INCIDENT_NUMBER": 10826064,
    },
    {
        "RECIEVE_METHOD": "Email и SMS",
        "APPLICANT_FORM": "Потерпевший",
        "APPLICANT_AGE": 37,
        "FILIAL": "Краснодарский",
        "EVENT_CREATED_BY_GIBDD_FLAG": "Да",
        "VICTIM_MAX_WEIGHT": 0,
        "GUILTY_CAPACITY_ENGINE": 1.6,
        "VICTIM_VEHICLE_AGE": 18,
        "EVENT_YEAR": 2025,
        "LOSS_UNIT_ZONE": "Зона ф-ла Краснодарский",
        "VICTIM_VEHICLE_COUNTRY": "Россия",
        "EVENT_DATE": "2025-06-15T17:00:00",
        "PAYMENT_ORDER_DATE_TIME": "2025-06-29T00:00:00",
        "LOSS_NUMBER": 10826052,
        "AMOUNT_REPAIR": 173124,
        "VICTIM_VIHICLE_BRAND": "Mazda",
        "VICTIM_VEHICLE_CATEGORY": "Японская. Правый руль",
        "INCIDENT_NUMBER": 10826052,
    },
    {
        "RECIEVE_METHOD": "Лично",
        "APPLICANT_FORM": "Потерпевший",
        "APPLICANT_AGE": 34,
        "FILIAL": "Казанский",
        "EVENT_CREATED_BY_GIBDD_FLAG": 0,
        "VICTIM_MAX_WEIGHT": 1760,
        "GUILTY_CAPACITY_ENGINE": 1.4,
        "VICTIM_VEHICLE_AGE": 16,
        "EVENT_YEAR": 2024,
        "LOSS_UNIT_ZONE": "Зона ф-ла Казанский",
        "VICTIM_VEHICLE_COUNTRY": "Россия",
        "EVENT_DATE": "2024-12-14T11:08:00",
        "PAYMENT_ORDER_DATE_TIME": "2024-12-18T00:00:00",
        "LOSS_NUMBER": 10429742,
        "AMOUNT_REPAIR": 65241,
        "VICTIM_VIHICLE_BRAND": "Toyota",
        "VICTIM_VEHICLE_CATEGORY": "Средний 7+",
        "INCIDENT_NUMBER": 10429742,
    },
    {
        "RECIEVE_METHOD": "Лично",
        "APPLICANT_FORM": "Выгодоприобретатель",
        "APPLICANT_AGE": 36,
        "FILIAL": "Тюменский",
        "EVENT_CREATED_BY_GIBDD_FLAG": 1,
        "VICTIM_MAX_WEIGHT": 2500,
        "GUILTY_CAPACITY_ENGINE": 3,
        "VICTIM_VEHICLE_AGE": 9,
        "EVENT_YEAR": 2025,
        "LOSS_UNIT_ZONE": "Зона ф-ла Тюменский",
        "VICTIM_VEHICLE_COUNTRY": "Россия",
        "EVENT_DATE": "2025-02-23T16:42:00",
        "PAYMENT_ORDER_DATE_TIME": "2025-03-04T00:00:00",
        "LOSS_NUMBER": 10582521,
        "AMOUNT_REPAIR": 179127,
        "VICTIM_VEHICLE_BRAND": "Lexus",
        "VICTIM_VEHICLE_CATEGORY": "Премиум 7+",
        "INCIDENT_NUMBER": 10582521,
    },
]

OUTGOING_VECTOR = [
    {
        "RECIEVE_METHOD": "НЕ ОПОВЕЩАТЬ",
        "APPLICANT_FORM": "ПРЕДСТАВИТЕЛЬ (НЕ АВТОЮРИСТ)",
        "APPLICANT_AGE": 52,
        "FILIAL": "БЕЛГОРОДСКИЙ",
        "EVENT_CREATED_BY_GIBDD_FLAG": "ДА",
        "VICTIM_MAX_WEIGHT": 18000,
        "GUILTY_CAPACITY_ENGINE": 2,
        "VICTIM_VEHICLE_AGE": 2,
        "EVENT_YEAR": 2026,
        "LOSS_UNIT_ZONE": "ЗОНА Ф-ЛА БЕЛГОРОДСКИЙ",
        "VICTIM_VEHICLE_COUNTRY": "РОССИЯ",
        "EVENT_DATE": "2026-03-23T08:00:00",
        "PAYMENT_ORDER_DATE_TIME": "2026-03-26T00:00:00",
        "LOSS_NUMBER": 11365778,
        "AMOUNT_REPAIR": np.nan,
        "VICTIM_VIHICLE_BRAND": "ЛИАЗ",
        "VICTIM_VEHICLE_CATEGORY": "ИНОЕ",
        "INCIDENT_NUMBER": 11365778,
        "APPLY_DELAY": 3,
    },
    {
        "RECIEVE_METHOD": "SMS",
        "APPLICANT_FORM": "ПРЕДСТАВИТЕЛЬ (ПО ДОВЕРЕННОСТИ)",
        "APPLICANT_AGE": 57,
        "FILIAL": "ВСК-МОСКВА",
        "EVENT_CREATED_BY_GIBDD_FLAG": "НЕТ",
        "VICTIM_MAX_WEIGHT": 1722,
        "GUILTY_CAPACITY_ENGINE": 1,
        "VICTIM_VEHICLE_AGE": 14,
        "EVENT_YEAR": 2025,
        "LOSS_UNIT_ZONE": "ЗОНА МОСКОВСКОГО РЕГИОНА",
        "VICTIM_VEHICLE_COUNTRY": "РОССИЯ",
        "EVENT_DATE": "2025-06-26T08:30:00",
        "PAYMENT_ORDER_DATE_TIME": "2025-07-03T00:00:00",
        "LOSS_NUMBER": 10826064,
        "AMOUNT_REPAIR": 169389,
        "VICTIM_VIHICLE_BRAND": "SKODA",
        "VICTIM_VEHICLE_CATEGORY": "СРЕДНИЙ 7+",
        "INCIDENT_NUMBER": 10826064,
        "APPLY_DELAY": 7,
    },
    {
        "RECIEVE_METHOD": "EMAIL И SMS",
        "APPLICANT_FORM": "ПОТЕРПЕВШИЙ",
        "APPLICANT_AGE": 37,
        "FILIAL": "КРАСНОДАРСКИЙ",
        "EVENT_CREATED_BY_GIBDD_FLAG": "ДА",
        "VICTIM_MAX_WEIGHT": 0,
        "GUILTY_CAPACITY_ENGINE": 1.6,
        "VICTIM_VEHICLE_AGE": 18,
        "EVENT_YEAR": 2025,
        "LOSS_UNIT_ZONE": "ЗОНА Ф-ЛА КРАСНОДАРСКИЙ",
        "VICTIM_VEHICLE_COUNTRY": "РОССИЯ",
        "EVENT_DATE": "2025-06-15T17:00:00",
        "PAYMENT_ORDER_DATE_TIME": "2025-06-29T00:00:00",
        "LOSS_NUMBER": 10826052,
        "AMOUNT_REPAIR": 173124,
        "VICTIM_VIHICLE_BRAND": "MAZDA",
        "VICTIM_VEHICLE_CATEGORY": "ЯПОНСКАЯ. ПРАВЫЙ РУЛЬ",
        "INCIDENT_NUMBER": 10826052,
        "APPLY_DELAY": 14,
    },
    {
        "RECIEVE_METHOD": "ЛИЧНО",
        "APPLICANT_FORM": "ПОТЕРПЕВШИЙ",
        "APPLICANT_AGE": 34,
        "FILIAL": "КАЗАНСКИЙ",
        "EVENT_CREATED_BY_GIBDD_FLAG": 0,
        "VICTIM_MAX_WEIGHT": 1760,
        "GUILTY_CAPACITY_ENGINE": 1.4,
        "VICTIM_VEHICLE_AGE": 16,
        "EVENT_YEAR": 2024,
        "LOSS_UNIT_ZONE": "ЗОНА Ф-ЛА КАЗАНСКИЙ",
        "VICTIM_VEHICLE_COUNTRY": "РОССИЯ",
        "EVENT_DATE": "2024-12-14T11:08:00",
        "PAYMENT_ORDER_DATE_TIME": "2024-12-18T00:00:00",
        "LOSS_NUMBER": 10429742,
        "AMOUNT_REPAIR": 65241,
        "VICTIM_VEHICLE_BRAND": "TOYOTA",
        "VICTIM_VEHICLE_CATEGORY": "СРЕДНИЙ 7+",
        "INCIDENT_NUMBER": 10429742,
        "APPLY_DELAY": 4,
    },
    {
        "RECIEVE_METHOD": "ЛИЧНО",
        "APPLICANT_FORM": "ВЫГОДОПРИОБРЕТАТЕЛЬ",
        "APPLICANT_AGE": 36,
        "FILIAL": "ТЮМЕНСКИЙ",
        "EVENT_CREATED_BY_GIBDD_FLAG": 1,
        "VICTIM_MAX_WEIGHT": 2500,
        "GUILTY_CAPACITY_ENGINE": 3,
        "VICTIM_VEHICLE_AGE": 9,
        "EVENT_YEAR": 2025,
        "LOSS_UNIT_ZONE": "ЗОНА Ф-ЛА ТЮМЕНСКИЙ",
        "VICTIM_VEHICLE_COUNTRY": "РОССИЯ",
        "EVENT_DATE": "2025-02-23T16:42:00",
        "PAYMENT_ORDER_DATE_TIME": "2025-03-04T00:00:00",
        "LOSS_NUMBER": 10582521,
        "AMOUNT_REPAIR": 179127,
        "VICTIM_VEHICLE_BRAND": "LEXUS",
        "VICTIM_VEHICLE_CATEGORY": "ПРЕМИУМ 7+",
        "INCIDENT_NUMBER": 10582521,
        "APPLY_DELAY": 9,
    },
]


class TestIncomingOutgoingVectorsPreprocessing(unittest.TestCase):
    def test_incoming_vector_preprocessing_matches_outgoing_vector(self):
        """
        Проверяет, что предобработка "сырого" вектора даёт ожидаемый нормализованный вектор.

        В тесте есть две версии одного и того же набора заявок:
            - `INCOMING_VECTOR`: как приходит от внешней системы (строки в произвольном регистре,
              возможны пустые строки вместо NaN).
            - `OUTGOING_VECTOR`: как должно выглядеть после нормализации:
                - текстовые поля в UPPER,
                - пустые строки заменены на NaN,
                - derived-поля рассчитаны (если требуется логикой сервиса).

        Смысл:
            Этот тест фиксирует контракт предобработки и помогает быстро ловить
            изменения поведения на уровне "вход -> общий df".
        """
        prepared_from_incoming = preprocessing_and_validate_vector(INCOMING_VECTOR)
        # OUTGOING_VECTOR — вручную собранный финальный df_common (перед mldataworker),
        # поэтому не прогоняем его через препроцессинг/пайплайн сервиса.
        prepared_from_outgoing = pd.DataFrame(OUTGOING_VECTOR)

        # Сравниваем только объединение фичей, которые дальше уходят в модели.
        expected_features = CLASSIFICATION_FEATURES + [
            c for c in REGRESSION_FEATURES if c not in CLASSIFICATION_FEATURES
        ]
        # Сопоставление "ожидаемого нормализованного" и "фактически подготовленного" фиксирует
        # контракт на уровне предобработки (без привязки к конкретным моделям).
        pd.testing.assert_frame_equal(
            prepared_from_incoming.loc[:, expected_features].reset_index(drop=True),
            prepared_from_outgoing.loc[:, expected_features].reset_index(drop=True),
            check_dtype=False,
        )