"""Подстановка списка LOSS_NUMBER в текст запроса 1С (Сутяжность)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

# Маркеры исходного запроса (один параметр &Убыток → список номеров).
_FIRST_FILTER = "Убыток.Ссылка = &Убыток"
_AUDANET_FILTER = "СрезПервых(, Убыток = &Убыток)"
_FINAL_FILTER = "ЗаявлениеНаВыплату.Убыток = &Убыток"

_AUDANET_REPLACEMENT = (
    "СрезПервых(, Убыток В (ВЫБРАТЬ втУбыток.Убыток ИЗ втУбыток))"
)
_FINAL_REPLACEMENT = (
    "ЗаявлениеНаВыплату.Убыток В (ВЫБРАТЬ втУбыток.Убыток ИЗ втУбыток)"
)


def read_query_text(path: Path) -> tuple[str, str]:
    """Прочитать запрос; вернуть (текст, encoding). Сначала cp1251, затем utf-8."""
    raw = path.read_bytes()
    for encoding in ("cp1251", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось декодировать запрос: {path}")


def format_loss_literals(
    loss_numbers: Sequence[object],
    *,
    as_strings: bool = True,
) -> str:
    """Список литералов для конструкции ``В (...)`` в языке запросов 1С."""
    parts: list[str] = []
    for value in loss_numbers:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if as_strings:
            escaped = text.replace('"', '""')
            parts.append(f'"{escaped}"')
        else:
            parts.append(text)
    if not parts:
        raise ValueError("Пустой список LOSS_NUMBER для подстановки в запрос")
    return ", ".join(parts)


def inject_loss_numbers(
    query_text: str,
    loss_numbers: Sequence[object],
    *,
    as_strings: bool = True,
) -> str:
    """Заменить единственный ``&Убыток`` на фильтр по номерам + ``В (втУбыток)``."""
    literals = format_loss_literals(loss_numbers, as_strings=as_strings)
    first_replacement = f"Убыток.Номер В ({literals})"

    missing = [
        marker
        for marker in (_FINAL_FILTER, _AUDANET_FILTER, _FIRST_FILTER)
        if marker not in query_text
    ]
    if missing:
        raise ValueError(
            "В тексте запроса не найдены ожидаемые маркеры: "
            + ", ".join(repr(m) for m in missing)
        )

    # Порядок важен: сначала длинные/уникальные фрагменты.
    result = query_text.replace(_FINAL_FILTER, _FINAL_REPLACEMENT, 1)
    result = result.replace(_AUDANET_FILTER, _AUDANET_REPLACEMENT, 1)
    result = result.replace(_FIRST_FILTER, first_replacement, 1)

    if "&Убыток" in result:
        raise ValueError(
            "После подстановки в запросе остался параметр &Убыток — "
            "проверьте шаблон Сутяжность.txt"
        )
    return result


def build_query_header(
    n_losses: int,
    source_df: Path | None = None,
    *,
    n_total: int | None = None,
    random_seed: int | None = None,
) -> str:
    """Комментарий в шапке файла для 1С."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    src = str(source_df) if source_df else "—"
    sample_line = ""
    if n_total is not None and random_seed is not None and n_losses < n_total:
        sample_line = f"// sample: {n_losses} из {n_total}, random_seed={random_seed}\n"
    elif n_total is not None:
        sample_line = f"// sample: все {n_losses} (из {n_total})\n"
    return (
        f"// vector_checker: {n_losses} убытков; сгенерировано {ts}\n"
        f"{sample_line}"
        f"// source df: {src}\n"
        "// Не коммитьте этот файл как шаблон — это рабочая копия для консоли 1С.\n"
        "// Внимание: ВТ_КлассификаторРегионов в шаблоне с ПЕРВЫЕ 1 — "
        "при нескольких убытках регион может быть неполным.\n"
        "\n"
    )


def write_injected_query(
    query_path: Path,
    loss_numbers: Sequence[object],
    out_path: Path,
    *,
    source_df: Path | None = None,
    as_strings: bool = True,
    encoding_out: str | None = None,
    n_total: int | None = None,
    random_seed: int | None = None,
) -> Path:
    """Прочитать шаблон, подставить номера, записать промежуточный файл."""
    losses_list = list(loss_numbers)
    text, encoding_in = read_query_text(query_path)
    injected = inject_loss_numbers(text, losses_list, as_strings=as_strings)
    header = build_query_header(
        len(losses_list),
        source_df=source_df,
        n_total=n_total,
        random_seed=random_seed,
    )
    # Не дублировать header при повторном prepare по уже пропатченному файлу.
    body = injected
    if body.lstrip().startswith("// vector_checker:"):
        # обрежем старую шапку до первого ВЫБРАТЬ
        idx = body.find("ВЫБРАТЬ")
        if idx > 0:
            body = body[idx:]
    out_encoding = encoding_out or encoding_in
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes((header + body).encode(out_encoding))
    return out_path


def write_sample_meta(
    path: Path,
    losses: Sequence[str],
    *,
    source_df: Path | None = None,
    n_total: int,
    n: int | None,
    random_seed: int,
) -> Path:
    """Сохранить список выбранных убытков для воспроизводимого compare."""
    import json

    payload = {
        "n": n,
        "n_total": n_total,
        "n_sampled": len(losses),
        "random_seed": random_seed,
        "source_df": str(source_df) if source_df else None,
        "loss_numbers": list(losses),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_sample_meta(path: Path) -> dict:
    """Прочитать sampled_losses.json."""
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def unique_loss_numbers(values: Iterable[object]) -> list[str]:
    """Уникальные LOSS_NUMBER с сохранением порядка."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan" or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def sample_loss_numbers(
    loss_numbers: Sequence[object],
    n: int | None = None,
    *,
    random_seed: int = 42,
) -> list[str]:
    """Уникальные LOSS_NUMBER; при ``n`` — случайная выборка с фиксированным seed.

    Если ``n`` is None или ``n >= len`` — возвращает все уникальные (исходный порядок).
    """
    import random

    unique = unique_loss_numbers(loss_numbers)
    if n is None:
        return unique
    if n < 0:
        raise ValueError(f"n должно быть >= 0, получено {n}")
    if n >= len(unique):
        return unique
    rng = random.Random(int(random_seed))
    return rng.sample(unique, n)
