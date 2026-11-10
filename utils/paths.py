"""Repository-relative path resolution for portable experiment runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | os.PathLike[str]) -> Path:
    """Expand a path, interpreting relative values from the repository root."""
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def artifact_root(config: Optional[DictConfig] = None) -> Path:
    """Resolve experiment.output_dir, T3D_OUTPUT_DIR, or <repo>/artifacts."""
    configured = None
    if config is not None:
        configured = OmegaConf.select(config, "experiment.output_dir", default=None)
    value = configured or os.environ.get("T3D_OUTPUT_DIR") or "artifacts"
    return resolve_repo_path(str(value))


def data_root(config: Optional[DictConfig] = None) -> Path:
    """Resolve dataset.data_dir, T3D_DATA_DIR, or <repo>/data."""
    configured = None
    if config is not None:
        configured = OmegaConf.select(config, "dataset.data_dir", default=None)
    value = configured or os.environ.get("T3D_DATA_DIR") or "data"
    return resolve_repo_path(str(value))


def dataset_file(dataset: str, config: Optional[DictConfig] = None) -> Path:
    return data_root(config) / f"{dataset}.json"


def project_dir(config: DictConfig) -> Path:
    project = str(OmegaConf.select(config, "experiment.project"))
    return artifact_root(config) / project


def training_root(config: DictConfig) -> Path:
    """Return the root below which model/dataset/run directories are created."""
    configured = OmegaConf.select(config, "experiment.save_dir", default=None)
    if configured:
        return resolve_repo_path(str(configured))
    return artifact_root(config) / "experiments"
