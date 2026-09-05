"""Оценка финэффекта по Excel-выгрузке (фаза 2 мониторинга).

Без факта ПСР: expected_psr = precision × Σ_I (paid×k + e_fee),
e_fee = p_fu×fu_fee + p_court×(fu_fee+court_fee).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from querulus.fin_effect.excel_explore import (
    COLUMN_ALIASES,
    _to_numeric,
    intervention_mask,
    load_excel,
    resolve_column,
)

FU_FEE_DEFAULT = 100_000.0
COURT_FEE_DEFAULT = 15_000.0


@dataclass(frozen=True)
class RetroPriors:
    """Ретро-приоры для оценки на выгрузке."""

    precision: float
    k: float
    p_pret: float
    p_fu: float
    p_court: float
    fu_fee: float = FU_FEE_DEFAULT
    court_fee: float = COURT_FEE_DEFAULT

    def expected_fee(self) -> float:
        """Средний пакет взносов на одну «успешную» интервенцию (путь суд несёт оба)."""
        return self.p_fu * self.fu_fee + self.p_court * (self.fu_fee + self.court_fee)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class MonitoringEffectResult:
    """Итог оценки на Excel."""

    frame: pd.DataFrame
    n_intervention: int
    sum_paid: float
    expected_psr: float
    cost: float
    net: float
    e_fee: float
    priors: RetroPriors
    paid_column: str


def load_retro_priors(path: str | Path) -> RetroPriors:
    """Загрузить priors из JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RetroPriors(
        precision=float(data["precision"]),
        k=float(data["k"]),
        p_pret=float(data["p_pret"]),
        p_fu=float(data["p_fu"]),
        p_court=float(data["p_court"]),
        fu_fee=float(data.get("fu_fee", FU_FEE_DEFAULT)),
        court_fee=float(data.get("court_fee", COURT_FEE_DEFAULT)),
    )


def save_retro_priors(priors: RetroPriors, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(priors.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def compute_retro_priors(
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    proba_col: str = "preds_cf",
    freq_col: str = "TARGET_FREQ",
    amount_col: str = "TARGET_FREQ_AMOUNT",
    od_col: str = "RECOVEREDMAINDEBT_LAST_INST_SUM",
    pret_col: str = "TARGET_FREQ_PRET_AMOUNT",
    fu_col: str = "Сумма_взыскано_по_ФУ",
    court_col: str = "Суммы_взыскано_по_иску",
    fu_fee: float = FU_FEE_DEFAULT,
    court_fee: float = COURT_FEE_DEFAULT,
) -> RetroPriors:
    """Посчитать priors на зрелом ретро-кадре с таргетами и proba."""
    work = df.copy()
    y = _to_numeric(work[freq_col]).fillna(0).astype(int)
    if proba_col in work.columns:
        proba = _to_numeric(work[proba_col])
        pred = (proba >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    else:
        precision = 0.5

    pos = y == 1
    amount = _to_numeric(work[amount_col]).fillna(0.0) if amount_col in work.columns else pd.Series(0.0, index=work.index)
    od = _to_numeric(work[od_col]).fillna(0.0) if od_col in work.columns else pd.Series(0.0, index=work.index)
    od_pos = pos & (od > 0)
    k = float(amount[od_pos].sum() / od[od_pos].sum()) if od_pos.any() and od[od_pos].sum() > 0 else 1.0

    pret = _to_numeric(work[pret_col]).fillna(0.0) if pret_col in work.columns else pd.Series(0.0, index=work.index)
    fu = _to_numeric(work[fu_col]).fillna(0.0) if fu_col in work.columns else pd.Series(0.0, index=work.index)
    court = (
        _to_numeric(work[court_col]).fillna(0.0)
        if court_col in work.columns
        else pd.Series(0.0, index=work.index)
    )

    n_pos = int(pos.sum())
    if n_pos == 0:
        p_pret, p_fu, p_court = 1.0, 0.0, 0.0
    else:
        is_court = pos & (court > 0)
        is_fu = pos & (fu > 0) & ~is_court
        is_pret = pos & ~is_court & ~is_fu & ((pret > 0) | (amount > 0))
        # остаток позитивов без явных сумм — в pret_only
        covered = is_court | is_fu | is_pret
        is_pret = is_pret | (pos & ~covered)
        p_court = float(is_court.sum() / n_pos)
        p_fu = float(is_fu.sum() / n_pos)
        p_pret = float(is_pret.sum() / n_pos)
        total = p_pret + p_fu + p_court
        if total > 0:
            p_pret, p_fu, p_court = p_pret / total, p_fu / total, p_court / total

    return RetroPriors(
        precision=precision,
        k=k,
        p_pret=p_pret,
        p_fu=p_fu,
        p_court=p_court,
        fu_fee=fu_fee,
        court_fee=court_fee,
    )


def resolve_paid_column(
    df: pd.DataFrame,
    paid_col: str | None = None,
) -> str:
    """Выбрать колонку paid: явная константа или эвристика по алиасам."""
    if paid_col:
        if paid_col in df.columns:
            return paid_col
        resolved = resolve_column(df, paid_col)
        if resolved is not None:
            return resolved
        raise KeyError(f"PAID_COL не найден: {paid_col}")
    for key in ("recommended", "payment", "od_to_pay", "to_pay", "od_paid"):
        col = resolve_column(df, key)
        if col is not None:
            return col
    raise KeyError("Не найдена колонка paid (рекомендованная / платёж / ОД)")


def estimate_monitoring_effect(
    df: pd.DataFrame,
    priors: RetroPriors,
    *,
    paid_col: str | None = None,
) -> MonitoringEffectResult:
    """Оценка net на выгрузке по формуле плана."""
    paid_name = resolve_paid_column(df, paid_col)
    mask = intervention_mask(df)
    paid = _to_numeric(df[paid_name]).fillna(0.0)
    paid_i = paid.where(mask, 0.0)
    e_fee = priors.expected_fee()
    sum_paid = float(paid_i.sum())
    n_i = int(mask.sum())
    # expected_psr = precision * Σ_I (paid * k + e_fee)
    expected_psr = float(priors.precision * ((paid_i * priors.k).sum() + n_i * e_fee))
    cost = sum_paid
    net = expected_psr - cost

    frame = df.copy()
    frame["_intervention"] = mask.astype(int)
    frame["_paid"] = paid
    frame["_paid_i"] = paid_i
    frame["_expected_psr_row"] = np.where(
        mask,
        priors.precision * (paid_i * priors.k + e_fee),
        0.0,
    )

    return MonitoringEffectResult(
        frame=frame,
        n_intervention=n_i,
        sum_paid=sum_paid,
        expected_psr=expected_psr,
        cost=cost,
        net=net,
        e_fee=e_fee,
        priors=priors,
        paid_column=paid_name,
    )


def sensitivity_table(
    df: pd.DataFrame,
    priors: RetroPriors,
    *,
    paid_col: str | None = None,
    rel_delta: float = 0.2,
) -> pd.DataFrame:
    """Чувствительность net к ±rel_delta по k и precision."""
    rows: list[dict[str, Any]] = []
    base = estimate_monitoring_effect(df, priors, paid_col=paid_col)
    rows.append(
        {
            "scenario": "base",
            "precision": priors.precision,
            "k": priors.k,
            "net": base.net,
            "expected_psr": base.expected_psr,
        }
    )
    for name, p_mult, k_mult in (
        ("precision -20%", 1 - rel_delta, 1.0),
        ("precision +20%", 1 + rel_delta, 1.0),
        ("k -20%", 1.0, 1 - rel_delta),
        ("k +20%", 1.0, 1 + rel_delta),
    ):
        adj = RetroPriors(
            precision=max(0.0, min(1.0, priors.precision * p_mult)),
            k=max(0.0, priors.k * k_mult),
            p_pret=priors.p_pret,
            p_fu=priors.p_fu,
            p_court=priors.p_court,
            fu_fee=priors.fu_fee,
            court_fee=priors.court_fee,
        )
        res = estimate_monitoring_effect(df, adj, paid_col=paid_col)
        rows.append(
            {
                "scenario": name,
                "precision": adj.precision,
                "k": adj.k,
                "net": res.net,
                "expected_psr": res.expected_psr,
            }
        )
    return pd.DataFrame(rows)


def default_demo_priors() -> RetroPriors:
    """Демо-приоры, пока нет ретро-parquet на контуре."""
    return RetroPriors(
        precision=0.55,
        k=1.35,
        p_pret=0.45,
        p_fu=0.30,
        p_court=0.25,
        fu_fee=FU_FEE_DEFAULT,
        court_fee=COURT_FEE_DEFAULT,
    )


_FILIALS = (
    "Курский",
    "Ярославский",
    "Пермский",
    "Архангельский",
    "Омский",
    "Владимирский",
    "Магнитогорский",
)


def build_synthetic_claims_excel(
    n_rows: int = 300,
    *,
    seed: int = 42,
    intervention_rate: float = 0.12,
) -> pd.DataFrame:
    """Синтетика с денежными identity и редкими интервенциями I."""
    if n_rows < 50:
        raise ValueError("n_rows >= 50")
    rng = np.random.default_rng(seed)
    filial = rng.choice(_FILIALS, size=n_rows)
    zones = np.array([f"Зона ф-ла {f}" for f in filial])
    incident = 11_000_000 + rng.integers(0, 900_000, size=n_rows)

    event = pd.to_datetime("2024-03-01") + pd.to_timedelta(
        rng.integers(0, 120, size=n_rows), unit="D"
    )
    apply = event + pd.to_timedelta(rng.integers(1, 15, size=n_rows), unit="D")

    # Интервенции
    n_i = max(1, int(round(n_rows * intervention_rate)))
    i_idx = rng.choice(n_rows, size=n_i, replace=False)
    is_i = np.zeros(n_rows, dtype=bool)
    is_i[i_idx] = True

    # Модель сказала платить шире, чем полное I
    result = np.zeros(n_rows, dtype=int)
    result[is_i] = 1
    extra_model = rng.choice(
        np.where(~is_i)[0],
        size=min(n_i, int((~is_i).sum())),
        replace=False,
    )
    result[extra_model] = 1
    # часть с result=1 но без соглашения
    agreement = np.zeros(n_rows, dtype=int)
    model_pay = np.zeros(n_rows, dtype=int)
    agreement[is_i] = 1
    model_pay[is_i] = 1

    recommended = np.zeros(n_rows, dtype=float)
    recommended[result == 1] = rng.uniform(15_000, 120_000, size=int((result == 1).sum()))

    # Основные суммы убытка
    wear = rng.uniform(5_000, 200_000, size=n_rows)
    other = np.where(rng.random(n_rows) < 0.15, rng.uniform(1_000, 40_000, size=n_rows), 0.0)
    to_pay = wear + other
    payment = to_pay.copy()

    claimed_od = rng.uniform(10_000, 250_000, size=n_rows)
    claimed_uts = np.where(rng.random(n_rows) < 0.2, rng.uniform(1_000, 30_000, size=n_rows), 0.0)
    claimed_evac = np.zeros(n_rows)
    claimed_storage = np.zeros(n_rows)
    od_claimed = claimed_od + claimed_uts + claimed_evac + claimed_storage

    od_to_pay = claimed_od * rng.uniform(0.4, 1.0, size=n_rows)
    # На I: выплата клиенту = рекомендованная (identity для сверки)
    payment[is_i] = recommended[is_i]
    od_to_pay[is_i] = recommended[is_i]
    to_pay[is_i] = recommended[is_i]
    wear[is_i] = recommended[is_i] * 0.85
    other[is_i] = recommended[is_i] * 0.15

    forms = np.array(
        rng.choice(
            ["Денежная", "Ремонт", "Соглашение", "Договорная", "Отказ"],
            size=n_rows,
            p=[0.45, 0.25, 0.12, 0.08, 0.10],
        )
    )
    forms[is_i] = "Соглашение"

    products = rng.choice(
        ["Традиционное ОСАГО", "Прямое ОСАГО (с 1 марта 2009)"],
        size=n_rows,
    )
    applicants = rng.choice(
        [
            "Потерпевший",
            "Юрист с потерпевшим",
            "Опытный юрист",
            "Представитель (не автоюрист)",
            "Выгодоприобретатель",
        ],
        size=n_rows,
    )

    df = pd.DataFrame(
        {
            "НомерИнцидента": incident,
            "ЗонаУрегулирования": zones,
            "Филиал": filial,
            "ДатаЗаявления": apply,
            "ДатаСобытия": event,
            "Продукт": products,
            "Тип заявителя": applicants,
            "ВозрастЗаявителя": rng.integers(18, 75, size=n_rows),
            "СпособПолученияЗаявления": rng.choice(
                ["Лично", "Электронное обращение"], size=n_rows
            ),
            "ОформленоГИБДД": rng.integers(0, 2, size=n_rows),
            "Категория ТС потерпевшего": rng.choice(
                ["Автомобили легковые", "Мотоциклы", "Микроавтобусы"], size=n_rows
            ),
            "Тип владельца транспортного средства": rng.choice(
                ["Физ. Лицо", "Юр. Лицо"], size=n_rows, p=[0.85, 0.15]
            ),
            "ФормаВозмещения": forms,
            "РезультатПроверки": result,
            "Заключено соглашение": agreement,
            "Выплата по модели в Инциденте": model_pay,
            "Сумма рекомендованная к доплате по модулю": recommended,
            "СуммаОсновногоДолгаЗаявлено": od_claimed,
            "СуммаОсновногоДолгаКВыплате": od_to_pay,
            "СтоимостьСУчетомИзноса": wear,
            "Иные затраты": other,
            "СуммаКВыплате": to_pay,
            "СуммаПлатежа": payment,
            "СуммаОД": od_to_pay,
            "СуммаОУ": payment,
            "Сумма выплаты по претензии": 0.0,
            "Кол-во претензий": 0,
            "Обращение в ФУ": 0,
            "Сумма выплат по ФУ": 0.0,
            "Обращение в суд": 0,
            "Сумма выплаты по суду": 0.0,
            "УбытокСтатус": "Первичный",
            "ОсновныеВыплатыЗаявлено_Основной долг": claimed_od,
            "ОсновныеВыплатыЗаявлено_Утрата товарной стоимости": claimed_uts,
            "ОсновныеВыплатыЗаявлено_Затраты на эвакуацию ТС": claimed_evac,
            "ОсновныеВыплатыЗаявлено_Затраты на хранение": claimed_storage,
            "ОсновныеВыплатыСуммаОплаченоДолиВыплата": od_to_pay,
        }
    )
    return df


def write_synthetic_claims_excel(
    path: str | Path,
    *,
    n_rows: int = 300,
    seed: int = 42,
) -> Path:
    """Сгенерировать и сохранить синтетический xlsx."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_synthetic_claims_excel(n_rows=n_rows, seed=seed)
    df.to_excel(path, index=False, engine="openpyxl")
    return path


# Реэкспорт для тетрадок
__all__ = [
    "COLUMN_ALIASES",
    "COURT_FEE_DEFAULT",
    "FU_FEE_DEFAULT",
    "MonitoringEffectResult",
    "RetroPriors",
    "build_synthetic_claims_excel",
    "compute_retro_priors",
    "default_demo_priors",
    "estimate_monitoring_effect",
    "intervention_mask",
    "load_excel",
    "load_retro_priors",
    "resolve_paid_column",
    "save_retro_priors",
    "sensitivity_table",
    "write_synthetic_claims_excel",
]
