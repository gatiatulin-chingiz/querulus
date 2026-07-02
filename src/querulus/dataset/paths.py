"""Пути к parquet-артефактам в querulus/data/."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from querulus.dataset import config


@dataclass
class DataPaths:
    """Корень data/ и каталоги raw/processed."""

    data_root: Path
    raw_dir: Path
    processed_dir: Path
    victim_path: Path
    local_data_dir: Path
    artifact_overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls) -> DataPaths:
        project_root = config.project_root
        data_root = project_root / "data"
        raw_dir = data_root / "raw"
        processed_dir = data_root / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        victim = (
            config.artifact_overrides.get("victim", "")
            or (config.victim_parquet_path or "").strip()
        )
        victim_path = Path(victim) if victim else Path()
        if victim_path and not victim_path.is_absolute():
            victim_path = project_root / victim_path
        return cls(
            data_root=data_root,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            victim_path=victim_path,
            local_data_dir=processed_dir,
            artifact_overrides=dict(config.artifact_overrides),
        )

    def get_override(self, name: str) -> Path | None:
        """Явный путь из configs/dataset_sources.json."""
        stem = Path(name).stem
        raw = self.artifact_overrides.get(name) or self.artifact_overrides.get(stem)
        if raw:
            return Path(raw)
        return None

    def artifact_candidates(self, directory: Path, name: str) -> list[Path]:
        """Пути, по которым ищется артефакт (для сообщений об ошибках)."""
        override = self.get_override(name)
        if override is not None:
            return [override]
        return [directory / name]

    def artifact(self, directory: Path, name: str) -> Path:
        """Путь для записи parquet."""
        directory.mkdir(parents=True, exist_ok=True)
        return directory / name

    def resolve_artifact(self, directory: Path, name: str) -> Path | None:
        """Найти parquet: override или directory/name."""
        override = self.get_override(name)
        if override is not None:
            return override if override.exists() else None
        path = directory / name
        return path if path.exists() else None
