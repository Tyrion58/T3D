"""Shared OmegaConf helpers used by training, rollout, and evaluation."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, MISSING, OmegaConf


def load_config() -> DictConfig:
    """Load ``config=<yaml>`` and merge all command-line overrides."""
    cli_conf = OmegaConf.from_cli()
    config_path = cli_conf.get("config")
    if not config_path:
        raise ValueError("Missing required CLI argument: config=/path/to/config.yaml")
    yaml_conf = OmegaConf.load(config_path)
    return OmegaConf.merge(yaml_conf, cli_conf)


def get_eval_config(config: DictConfig, key: str, default: Any = MISSING) -> Any:
    """Read an evaluation option with compatibility fallbacks."""
    value = OmegaConf.select(config, f"evaluation.{key}", default=MISSING)
    if value is not MISSING:
        return value

    fallback_paths = {
        "eval_dataset": ("dataset.eval_dataset",),
        "data_type": ("dataset.data_type",),
        "num_node": ("experiment.num_node",),
        "node_index": ("experiment.node_index",),
        "num_response_per_task": ("rollout.num_response_per_task",),
        "start_with_think": ("rollout.start_with_think",),
        "block_size": ("rollout.block_size",),
        "tensor_parallel_size": ("rollout.tensor_parallel_size",),
        "stop_token_list": ("rollout.stop_token_list",),
        "temperature": ("rollout.temperature",),
        "top_k": ("rollout.top_k",),
        "top_p": ("rollout.top_p",),
        "max_token": ("rollout.max_token",),
        "max_gen_length": ("rollout.max_gen_length",),
        "steps": ("rollout.steps",),
        "target": ("rollout.target",),
        "remasking_strategy": ("rollout.remasking_strategy",),
        "denoising_steps_per_block": ("rollout.denoising_steps_per_block",),
        "dynamic_threshold": ("rollout.dynamic_threshold",),
        "further_horizon": ("rollout.further_horizon",),
        "use_cache": ("rollout.use_cache",),
        "max_active": ("rollout.max_active",),
        "random_init_ratio": ("rollout.random_init_ratio",),
        "output_unmasking_history": ("rollout.output_unmasking_history",),
        "batch_size": ("rollout.batch_size",),
    }
    for path in fallback_paths.get(key, ()):
        value = OmegaConf.select(config, path, default=MISSING)
        if value is not MISSING:
            return value

    if default is not MISSING:
        return default
    raise KeyError(f"Config key '{key}' was not found")


def build_outputs_name(config: DictConfig, pretrained_model: str, dataset: str) -> str:
    """Build a traceable filename stem from model and rollout settings."""
    parts = [f"eval-{pretrained_model.rstrip('/').split('/')[-1]}"]
    for config_key, label in (
        ("rollout.steps", "s"),
        ("rollout.denoising_steps_per_block", "dspb"),
        ("rollout.block_size", "bs"),
    ):
        value = OmegaConf.select(config, config_key, default=None)
        if value is not None:
            parts.append(f"{label}{value}")
    parts.append(dataset)
    return "-".join(parts)


def flatten_cli_value(value: Any) -> str:
    """Serialize a CLI override value in OmegaConf-compatible form."""
    if isinstance(value, (DictConfig, ListConfig, dict, list)):
        return OmegaConf.to_yaml(OmegaConf.create(value), resolve=True).strip()
    if isinstance(value, str) and " " in value:
        return f'"{value}"'
    return str(value)
