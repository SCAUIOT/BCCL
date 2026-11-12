"""Host-local paths, shared scientific configuration, and identity checks."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import yaml
from pig_beh_ce.continuous import validate_settings

TEST_NAMES = [f"T{i}_{condition}" for i in (1, 2, 3)
              for condition in ("Day", "Night", "Feeding", "Light")]


def read_yaml(path):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value, path


def local_path(value, config_path):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(config_path).parent / p).resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("IDs must contain 1-96 ASCII letters, digits, underscores, dots or hyphens")
    return value


def token_from(config):
    name = config.get("network", {}).get("token_env", "CLOUD_EDGE_TOKEN")
    token = os.environ.get(name, "")
    if not token:
        raise ValueError(f"Set the same nonempty {name} environment variable on both hosts")
    return token


def algorithm(config, config_path):
    raw, _ = read_yaml(local_path(config["experiment"], config_path))
    validate_settings(raw["continuous"])
    mix = raw["data_mix"]
    if not 0 < float(mix["target_val_fraction"]) <= .5:
        raise ValueError("Deployment requires a nonempty holdout")
    if int(raw["continuous"]["min_batch_images"]) <= int(mix["min_target_val_images"]):
        raise ValueError("Batches must leave both train and holdout images")
    if raw["models"].get("workers", 0) != 0 or raw["models"].get("train_options", {}).get("cache", False):
        raise ValueError("Use workers=0 and cache=false for bounded storage")
    if int(raw["model_bank"]["top_k"]) > 10:
        raise ValueError("This storage profile supports at most 10 bank models")
    return raw


def algorithm_hash(raw):
    value = copy.deepcopy(raw)
    value.pop("paths", None)
    value.get("models", {}).pop("device", None)
    return digest(value)


def cloud_raw(shared, config, config_path, root):
    raw = copy.deepcopy(shared)
    paths = config["paths"]
    raw["paths"] = {k: str(local_path(paths[k], config_path))
                    for k in ("initial_student", "teacher", "source_yaml")}
    dataset = local_path(paths["dataset_root"], config_path)
    raw["paths"].update(dataset_root=str(dataset), work_dir=str(root), videos_dir=str(root / "unused_videos"))
    raw["models"]["device"] = config.get("device", "auto")
    evaluation = config.get("evaluation", {})
    raw["final_evaluation"] = {"enabled": evaluation.get("enabled", True), "datasets": [
        {"name": name, "root": str(dataset / "continual" / name / "test_labeled")}
        for name in evaluation.get("names", TEST_NAMES)]}
    raw["continual"] = {"stages": []}
    return raw
