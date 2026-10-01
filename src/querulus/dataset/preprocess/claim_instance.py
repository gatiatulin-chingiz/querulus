"""Константы и проверка «пустой» инстанции иска (общий лист для targets/maturity)."""
from __future__ import annotations

import pandas as pd

CLAIM_PERIOD_COL = "CLAIMEDVALUEPERIOD"
_VOID_DECISION = "не принято"
_RECOVERY_SIGNAL_COLS = (
    "RECOVEREDVALUEWITHSD",
    "RECOVEREDMAINDEBT",
    "RECOVEREDWEAROUT",
    "RECOVEREDLOSSCOMMODYVALUE",
)


def is_void_claim_instance(df: pd.DataFrame) -> pd.Series:
    """Инстанция без принятого взыскания: Decision='Не принято' или все суммы Null.

    Не опирается на номер инстанции — только Decision и наличие сумм.
    """
    void = pd.Series(False, index=df.index)
    if "DECISION" in df.columns:
        decision = df["DECISION"].fillna("").astype(str).str.strip().str.casefold()
        void = void | decision.eq(_VOID_DECISION)
    signal_cols = [col for col in _RECOVERY_SIGNAL_COLS if col in df.columns]
    if signal_cols:
        amounts = df[signal_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        void = void | amounts.isna().all(axis=1)
    return void
