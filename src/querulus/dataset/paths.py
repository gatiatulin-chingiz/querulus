"""Пути к данным и версионирование parquet-артефактов."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from querulus.dataset import config

_VERSIONED_NAME_RE = re.compile(r"^(.+)_v(.+)(\.\w+)$")


@dataclass
class DataPaths:
    """Корневые каталоги и локальная папка data/."""

    data_root: Path
    raw_dir: Path
    processed_dir: Path
    victim_path: Path
    local_data_dir: Path
    artifact_version: str
    legacy_data_root: Path | None
    artifact_overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls) -> DataPaths:
        project_root = config.project_root
        project_data = project_root / "data"
        raw_dir = project_data / "raw"
        processed_dir = project_data / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        external = (config.litigant_data_root or "").strip()
        if external and Path(external).resolve() != project_root.resolve():
            root = Path(external)
        else:
            root = project_data

        legacy = (config.litigant_legacy_data_root or "").strip()
        victim = (
            config.artifact_overrides.get("victim", "")
            or (config.victim_parquet_path or "").strip()
        )
        return cls(
            data_root=root,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            victim_path=Path(victim) if victim else Path(),
            local_data_dir=processed_dir,
            artifact_version=config.litigant_artifact_version,
            legacy_data_root=Path(legacy) if legacy else None,
            artifact_overrides=dict(config.artifact_overrides),
        )

    def get_override(self, name: str) -> Path | None:
        """Явный путь из configs/dataset_sources.json."""
        stem = Path(name).stem
        raw = self.artifact_overrides.get(name) or self.artifact_overrides.get(stem)
        if raw:
            return Path(raw)
        return None

    def _data_roots(self) -> list[Path]:
        """Legacy-корень первым — при чтении старые артефакты находятся раньше."""
        roots: list[Path] = []
        if self.legacy_data_root is not None:
            roots.append(self.legacy_data_root)
        project_data = self.raw_dir.parent
        for root in (self.data_root, project_data):
            if root not in roots:
                roots.append(root)
        return roots

    def _normalized_version(self) -> str:
        return self.artifact_version.strip().lstrip("_")

    @staticmethod
    def _version_sort_key(version_str: str) -> tuple:
        try:
            return (0, int(version_str), version_str)
        except ValueError:
            return (1, 0, version_str)

    def _rel_subdir(self, directory: Path) -> Path | None:
        for root in self._data_roots():
            try:
                return directory.relative_to(root)
            except ValueError:
                continue
        return None

    def _add_dir(self, dirs: list[Path], path: Path) -> None:
        if path not in dirs:
            dirs.append(path)

    def _search_dirs(self, primary: Path) -> list[Path]:
        """Каталоги для чтения: primary, зеркало, data/raw, data/processed, parquet."""
        dirs: list[Path] = []

        self._add_dir(dirs, primary)

        rel = self._rel_subdir(primary)
        if rel is not None:
            for root in self._data_roots():
                self._add_dir(dirs, root / rel)

        for root in self._data_roots():
            self._add_dir(dirs, root / "data" / "raw")
            self._add_dir(dirs, root / "data" / "processed")
            self._add_dir(dirs, root / "parquet")

        return dirs

    def _filename_variants(self, name: str) -> list[str]:
        stem = Path(name).stem
        suffix = Path(name).suffix or ".parquet"
        variants: list[str] = [name]
        version = self._normalized_version()
        raw_version = self.artifact_version.strip()
        if version:
            variants.append(f"{stem}_v{version}{suffix}")
        if raw_version and raw_version != version:
            variants.append(f"{stem}_v{raw_version}{suffix}")
        seen: set[str] = set()
        unique: list[str] = []
        for item in variants:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def _collect_matches(
        self, lookup_dir: Path, stem: str, suffix: str, *, fixed_version: str
    ) -> list[tuple[tuple, Path]]:
        """Собрать совпадения: при fixed_version — только эта версия, иначе все _v*."""
        if not lookup_dir.is_dir():
            return []

        matches: list[tuple[tuple, Path]] = []
        unversioned = lookup_dir / f"{stem}{suffix}"
        if unversioned.exists():
            matches.append(((-1,), unversioned))

        if fixed_version:
            versioned = lookup_dir / f"{stem}_v{fixed_version}{suffix}"
            if versioned.exists():
                return [(self._version_sort_key(fixed_version), versioned)]
            return matches

        for path in lookup_dir.iterdir():
            if not path.is_file():
                continue
            matched = _VERSIONED_NAME_RE.match(path.name)
            if matched and matched.group(1) == stem and matched.group(3) == suffix:
                ver = matched.group(2)
                matches.append((self._version_sort_key(ver), path))
        return matches

    def artifact_candidates(self, directory: Path, name: str) -> list[Path]:
        """Все пути, по которым ищется артефакт при чтении (для логов ошибок)."""
        override = self.get_override(name)
        if override is not None:
            return [override]

        fixed = self._normalized_version()
        stem = Path(name).stem
        suffix = Path(name).suffix or ".parquet"
        candidates: list[Path] = []

        if fixed:
            for lookup_dir in self._search_dirs(directory):
                for fname in self._filename_variants(name):
                    candidates.append(lookup_dir / fname)
            return candidates

        for lookup_dir in self._search_dirs(directory):
            for _, path in self._collect_matches(
                lookup_dir, stem, suffix, fixed_version=""
            ):
                candidates.append(path)
        return candidates

    def artifact(self, directory: Path, name: str) -> Path:
        """Путь для записи parquet (основной каталог + версия)."""
        directory.mkdir(parents=True, exist_ok=True)
        stem = Path(name).stem
        suffix = Path(name).suffix or ".parquet"
        version = self._normalized_version()
        if version:
            return directory / f"{stem}_v{version}{suffix}"
        return directory / name

    def resolve_artifact(self, directory: Path, name: str) -> Path | None:
        """Найти parquet: override → зафиксированная версия → последняя _v* → без суффикса."""
        override = self.get_override(name)
        if override is not None:
            return override if override.exists() else None

        fixed = self._normalized_version()
        stem = Path(name).stem
        suffix = Path(name).suffix or ".parquet"
        best: tuple[tuple, Path] | None = None

        for lookup_dir in self._search_dirs(directory):
            for sort_key, path in self._collect_matches(
                lookup_dir, stem, suffix, fixed_version=fixed
            ):
                if fixed:
                    return path
                if best is None or sort_key > best[0]:
                    best = (sort_key, path)

        return best[1] if best else None
