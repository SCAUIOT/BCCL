from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *over* into a copy of *base* (override wins).

    Sibling keys in *base* are preserved; only the paths present in
    *over* are replaced.  Used to apply per-ablation-group config
    overrides on top of the shared experiment yaml.
    """
    merged = {k: v for k, v in (base or {}).items()}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def resolve_device(value: Any) -> str | int:
    text = str(value).strip().lower()
    if text in ('cpu', 'none'):
        return 'cpu'
    if text == 'auto':
        try:
            import torch
            return 0 if torch.cuda.is_available() else 'cpu'
        except Exception:
            return 'cpu'
    if text.isdigit():
        return int(text)
    return str(value)


@dataclass
class ExpPaths:
    dataset_root: Path
    source_yaml: Path | None
    work_dir: Path
    initial_student: Path
    teacher: Path
    scores_cache: Path | None = None


@dataclass
class Experiment:
    raw: dict[str, Any]
    config_path: Path
    paths: ExpPaths

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'Experiment':
        path = Path(path).resolve()
        raw = load_yaml(path)
        base = path.parent.parent if path.parent.name == 'configs' else Path.cwd()

        def p(v: str) -> Path:
            q = Path(v)
            return q if q.is_absolute() else (base / q).resolve()

        source_yaml = raw.get('paths', {}).get('source_yaml')
        scores_cache_s = raw.get('paths', {}).get('scores_cache')
        paths = ExpPaths(
            dataset_root=p(raw['paths']['dataset_root']),
            source_yaml=p(source_yaml) if source_yaml else None,
            work_dir=p(raw['paths']['work_dir']),
            initial_student=p(raw['paths']['initial_student']),
            teacher=p(raw['paths']['teacher']),
            scores_cache=p(scores_cache_s) if scores_cache_s else None,
        )
        paths.work_dir.mkdir(parents=True, exist_ok=True)
        return cls(raw=raw, config_path=path, paths=paths)

    def stage_names(self) -> list[str]:
        return list(self.raw['continual']['stages'])

    def stage_unlabeled_dir(self, stage: str) -> Path:
        return self.paths.dataset_root / 'continual' / stage / self.raw['continual'].get('unlabeled_dir_name', 'unlabeled')

    def stage_videos_dir(self, stage: str) -> Path:
        return self.paths.dataset_root / 'continual' / stage / self.raw['continual'].get('videos_dir_name', 'videos')

    def stage_stream_type(self, stage: str) -> str:
        videos = self.stage_videos_dir(stage)
        if videos.exists() and any(videos.glob('*')):
            return 'videos'
        return 'images'

    def stage_test_root(self, stage: str) -> Path:
        return self.paths.dataset_root / 'continual' / stage / self.raw['continual'].get('test_dir_name', 'test_labeled')
    def method_dir(self, method: str, upload_ratio: float | None = None, mix_mode: str | None = None, variant_suffix: str = '') -> Path:
        suffix = method if upload_ratio is None else f'{method}_r{upload_ratio:.2f}'
        if mix_mode and method != 'source_only':
            suffix = f'{suffix}__{mix_mode}'
        suffix = suffix + variant_suffix
        out = self.paths.work_dir / 'experiments' / suffix
        out.mkdir(parents=True, exist_ok=True)
        return out
