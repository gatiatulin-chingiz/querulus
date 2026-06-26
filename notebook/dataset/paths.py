"""Пути к данным и версионирование parquet-артефактов."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config


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

    @classmethod
    def from_config(cls) -> DataPaths:
        project_root = config.base_path.parent
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
        return cls(
            data_root=root,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            victim_path=Path(config.victim_parquet_path),
            local_data_dir=processed_dir,
            artifact_version=config.litigant_artifact_version,
            legacy_data_root=Path(legacy) if legacy else None,
        )

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

    def artifact_candidates(self, directory: Path, name: str) -> list[Path]:
        """Все пути, по которым ищется артефакт при чтении."""
        return [
            lookup_dir / fname
            for lookup_dir in self._search_dirs(directory)
            for fname in self._filename_variants(name)
        ]

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
        """Первый существующий файл или None."""
        for path in self.artifact_candidates(directory, name):
            if path.exists():
                return path
        return None
