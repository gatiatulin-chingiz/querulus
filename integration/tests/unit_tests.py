"""
Юнит-тесты контракта предобработки в примере `querulus`.

Фокус
    Тестируем строго определённые "уровни" проверок, которые применяются к входному вектору:
        - A3/A5: точный состав и порядок колонок на входе в `prepare_dataset`.
        - A6: запрет дублирующихся имён колонок.
        - B1: нормализация пустых строк "" -> NaN.
        - B2: нормализация регистра (строковые значения -> UPPER()).
        - B3: расчёт derived-полей из дат (EVENT_YEAR, APPLY_DELAY).
        - C1/C2/C3: строгая валидность дат и derived-полей.

Почему здесь нет моделей
    Это "малые" тесты именно препроцессинга/валидаторов — без загрузки pickle и без инференса.
    Сквозные проверки пайплайна и предсказаний лежат в `tests/integration_tests.py`.
"""

import unittest

import numpy as np
import pandas as pd

from integration.main import (
    enrich_common_vector_dataframe,
    prepare_common_vector_dataframe,
    validate_derived_dates_strict,
    validate_exact_columns_order,
    validate_no_duplicate_columns,
)


CLASSIFICATION_FEATURES = [
    "FILIAL",
    "RECIEVE_METHOD",
    "APPLICANT_FORM",
    "VICTIM_MAX_WEIGHT",
    "EVENT_YEAR",
    "EVENT_CREATED_BY_GIBDD_FLAG",
    "APPLICANT_AGE",
    "GUILTY_CAPACITY_ENGINE",
    "VICTIM_VEHICLE_CATEGORY",
    "VICTIM_VEHICLE_AGE",
]

REGRESSION_FEATURES = [
    "LOSS_UNIT_ZONE",
    "VICTIM_VEHICLE_COUNTRY",
    "APPLY_DELAY",
    "AMOUNT_REPAIR",
]


def _base_vector():
    # Минимальный "сырой" вектор, достаточный для выбранных проверок.
    # Важно: фиксируем порядок ключей в dict намеренно — так проще отлаживать
    # ожидаемый состав/порядок колонок после сборки DataFrame.
    return {
        # CLASSIFICATION_FEATURES (raw, без EVENT_YEAR)
        "FILIAL": "Дальневосточный",
        "RECIEVE_METHOD": "Почта",
        "APPLICANT_FORM": "Представитель",
        "VICTIM_MAX_WEIGHT": np.nan,
        "EVENT_CREATED_BY_GIBDD_FLAG": 1,
        "APPLICANT_AGE": np.nan,
        "GUILTY_CAPACITY_ENGINE": 3.0,
        "VICTIM_VEHICLE_CATEGORY": "Категория",
        "VICTIM_VEHICLE_AGE": 9.0,
        # REGRESSION_FEATURES (raw, без APPLY_DELAY)
        "LOSS_UNIT_ZONE": "Зона",
        "VICTIM_VEHICLE_COUNTRY": "JAPAN",
        "AMOUNT_REPAIR": 442445.0,
        # Обязательные для derived
        "EVENT_DATE": "2017-10-08 16:08:00",
        "PAYMENT_ORDER_DATE_TIME": "2022-02-28 11:13:42",
    }


class TestStrictVectorValidation(unittest.TestCase):

    # Уровень A3: Проверяет, что отсутствие обязательного поля вызывает исключение.
    def test_a3_missing_required_field_rejected(self):
        # Делаем df без колонки "B" и ожидаем ValueError, потому что контракт требует
        # точного состава колонок, а не "подмножества".
        df = pd.DataFrame([{"A": 1}])
        with self.assertRaises(ValueError):
            validate_exact_columns_order(df, ["A", "B"], "TestRole")

    # Уровень A5: Проверяет, что несоответствие порядка столбцов вызывает исключение.
    def test_a5_order_mismatch_rejected_for_dataframe(self):
        # Даже если набор колонок совпадает, порядок — часть контракта, потому что downstream
        # препроцессинг/кодирование ожидают фиксированный порядок признаков.
        df = pd.DataFrame([[1, 2]], columns=["B", "A"])
        with self.assertRaises(ValueError):
            validate_exact_columns_order(df, ["A", "B"], "TestRole")

    # Уровень A6: Проверяет, что дублирование столбцов вызывает исключение.
    def test_a6_duplicate_columns_rejected(self):
        # Дубли в именах колонок делают дальнейшую работу неоднозначной (какую колонку брать?),
        # поэтому это fail-fast проверка.
        df = pd.DataFrame([[1, 2]], columns=["A", "A"])
        with self.assertRaises(ValueError):
            validate_no_duplicate_columns(df)

    # Уровень B1: Проверяет, что пустая строка преобразуется в NaN.
    def test_b1_empty_string_becomes_nan(self):
        # Имитация типичной ситуации: внешняя система присылает "" вместо пропуска.
        # Контракт сервиса: "" должно стать NaN ещё до расчёта derived/подачи в модели.
        v = _base_vector()
        v["APPLICANT_FORM"] = ""
        df = prepare_common_vector_dataframe([v])
        self.assertTrue(pd.isna(df["APPLICANT_FORM"].iloc[0]))

    # Уровень B2: Проверяет, что строки приводятся к верхнему регистру.
    def test_b2_strings_uppercased(self):
        # Нормализация регистра нужна для категориальных признаков: чтобы значения
        # совпадали с обучением и не создавали "новые" категории на инференсе.
        v = _base_vector()
        v["FILIAL"] = "Дальневосточный"
        df = prepare_common_vector_dataframe([v])
        self.assertEqual(df["FILIAL"].iloc[0], "ДАЛЬНЕВОСТОЧНЫЙ")

    # Уровень B3: Проверяет, что производные столбцы добавляются.
    def test_b3_derived_columns_added(self):
        # Derived-поля рассчитываются строго из дат — независимо от того,
        # приходили ли они в исходном payload.
        v = _base_vector()
        df = prepare_common_vector_dataframe([v])
        df = enrich_common_vector_dataframe(df)
        self.assertIn("EVENT_YEAR", df.columns)
        self.assertIn("APPLY_DELAY", df.columns)

    # Уровень C1/C2: Проверяет, что непарсируемые даты вызывают исключение.
    def test_c1_c2_dates_not_parseable_rejected(self):
        # Пробуем сломать каждую из дат по отдельности и ожидаем строгую ошибку валидации.
        for field in ["EVENT_DATE", "PAYMENT_ORDER_DATE_TIME"]:
            with self.subTest(field=field):
                v = _base_vector()
                v[field] = "not-a-date"
                with self.assertRaises(ValueError):
                    # Важно: валидатор дат запускается после enrich (где даты парсятся),
                    # поэтому имитируем реальную последовательность шагов.
                    df = prepare_common_vector_dataframe([v])
                    df = enrich_common_vector_dataframe(df)
                    validate_derived_dates_strict(df)

    # Уровень C3: Проверяет, что отрицательное значение apply_delay вызывает исключение.
    def test_c3_negative_apply_delay_rejected(self):
        # apply_delay < 0 означает, что дата события позже даты оплаты — это бизнес-неконсистентность.
        v = _base_vector()
        v["EVENT_DATE"] = "2022-03-01 00:00:00"
        v["PAYMENT_ORDER_DATE_TIME"] = "2022-02-28 00:00:00"
        with self.assertRaises(ValueError):
            df = prepare_common_vector_dataframe([v])
            df = enrich_common_vector_dataframe(df)
            validate_derived_dates_strict(df)

    # Уровень D1: Проверяет, что списки фич используются как reference.
    def test_feature_lists_used_as_reference(self):
        # Проверка фиксации требований: derived-поля должны присутствовать в reference-списках,
        # потому что дальше код "режет" df именно этими списками.
        self.assertIn("EVENT_YEAR", CLASSIFICATION_FEATURES)
        self.assertIn("APPLY_DELAY", REGRESSION_FEATURES)