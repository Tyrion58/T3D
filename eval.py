"""Run rollout, optional code execution, and reward evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from termcolor import cprint

from utils.config import get_eval_config, load_config
from utils.paths import REPO_ROOT, project_dir, resolve_repo_path


def flatten_overrides(value: Any, parent: str = "") -> list[str]:
    """Flatten nested OmegaConf CLI overrides into ``key=value`` arguments."""
    items: list[str] = []
    if isinstance(value, (DictConfig, dict)):
        for key, child in value.items():
            name = f"{parent}.{key}" if parent else str(key)
            items.extend(flatten_overrides(child, name))
        return items
    if isinstance(value, (ListConfig, list)):
        rendered = OmegaConf.to_container(value, resolve=True)
        items.append(f"{parent}={rendered}")
        return items
    items.append(f"{parent}={value}")
    return items


SAMPLE_SCRIPTS = {"sdar": "sample/sdar_sample.py"}


def detect_model_base(config: DictConfig) -> str:
    """Model family named by ``model_base`` in the config (default: sdar)."""
    explicit = OmegaConf.select(config, "model_base", default=None)
    return str(explicit).lower() if explicit else "sdar"


def run_stage(script: str, config_path: Path, overrides: list[str]) -> None:
    command = [sys.executable, str(REPO_ROOT / script), f"config={config_path}", *overrides]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    config = load_config()
    cli_conf = OmegaConf.from_cli()
    config_path = resolve_repo_path(str(cli_conf.config)).resolve()
    override_conf = {key: value for key, value in cli_conf.items() if key != "config"}
    overrides = flatten_overrides(override_conf)

    model_base = detect_model_base(config)
    cprint(f"Using model_base: {model_base}", color="cyan")
    if model_base not in SAMPLE_SCRIPTS:
        raise NotImplementedError(f"model_base '{model_base}' is not supported")

    (project_dir(config) / "results").mkdir(parents=True, exist_ok=True)

    cprint("Generating rollouts", color="green")
    run_stage(SAMPLE_SCRIPTS[model_base], config_path, overrides)

    if get_eval_config(config, "data_type", "math") == "code":
        cprint("Executing generated code", color="green")
        run_stage("reward/execute.py", config_path, overrides)

    cprint("Computing rewards", color="green")
    run_stage("reward/reward.py", config_path, overrides)


if __name__ == "__main__":
    main()
