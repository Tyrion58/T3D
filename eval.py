import os
import sys
import subprocess
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf, MISSING

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def flatten_dict_to_dotlist(d, parent_key=''):
    """
    Flatten a nested dict/DictConfig to a list of 'key=value' strings in dotted format.
    e.g., {'evaluation': {'checkpoint_path': 'xxx'}} -> ['evaluation.checkpoint_path=xxx']
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_dict_to_dotlist(v, new_key))
        elif isinstance(v, (list, ListConfig)):
            # Convert list to OmegaConf-compatible format
            items.append(f'{new_key}={OmegaConf.to_yaml(OmegaConf.create(v), resolve=True).strip()}')
        else:
            # Handle strings with spaces
            if isinstance(v, str) and " " in v:
                items.append(f'{new_key}="{v}"')
            else:
                items.append(f'{new_key}={v}')
    return items

def get_eval_config(config, key, default=MISSING):
    """Helper function to get evaluation config with fallback to old locations."""
    # Try evaluation section first
    eval_val = OmegaConf.select(config, f"evaluation.{key}", default=MISSING)
    if eval_val is not MISSING:
        return eval_val
    
    # Fallback to old locations for backward compatibility
    dataset_key_map = {
        "data_type": "dataset.data_type",
        "eval_dataset": "dataset.eval_dataset",
    }
    
    if key in dataset_key_map:
        val = OmegaConf.select(config, dataset_key_map[key], default=MISSING)
        if val is not MISSING:
            return val
    
    if default is not MISSING:
        return default
    
    raise KeyError(f"Config key '{key}' not found in evaluation section or fallback locations")

if __name__ == "__main__":
    config = get_config()

    project_name = config.experiment.project
    eval_type = get_eval_config(config, "data_type", "math")
    
    # Auto-detect model_base if not explicitly specified
    if "model_base" in config:
        model_base = config.model_base
    else:
        # Try to detect from model name
        model_name = None
        if isinstance(config.model, str):
            model_name = config.model.lower()
        elif hasattr(config, "model") and hasattr(config.model, "pretrained_model"):
            model_name = config.model.pretrained_model.lower()
        
        # Check config file name
        cli_conf = OmegaConf.from_cli()
        config_file = cli_conf.get("config", "")
        
        # Detect from model name or config file name
        if model_name and "llada" in model_name:
            model_base = "llada"
        elif "llada" in config_file.lower():
            model_base = "llada"
        elif model_name and "dream" in model_name:
            model_base = "dream"
        elif "dream" in config_file.lower():
            model_base = "dream"
        else:
            # Default to "sdar"
            model_base = "sdar"
    
    cprint(f"Using model_base: {model_base}", color="cyan")

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")

    def sample(model_base):
        cprint(f"This is sampling.", color = "green")
        # Build extra config args from CLI overrides (excluding 'config' itself)
        cli_conf = OmegaConf.from_cli()
        original_config = cli_conf.get("config", f"../configs/{project_name}.yaml")
        cli_without_config = {k: v for k, v in cli_conf.items() if k != "config"}
        extra_args = flatten_dict_to_dotlist(cli_without_config)
        extra_args_str = " ".join(extra_args)
        
        if model_base == "dream":
            raise NotImplementedError("Dream is not supported yet")
            subprocess.run(
                f'python dream_sample.py '
                f'config=../configs/{project_name}.yaml {extra_args_str}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "llada":
            subprocess.run(
                f'python llada_sample.py '
                f'config=../configs/{project_name}.yaml {extra_args_str}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "sdar":
            subprocess.run(
                f'python sdar_sample.py '
                f'config=../configs/{project_name}.yaml {extra_args_str}',
                shell=True,
                cwd='sample',
                check=True,
            )
    
    def reward():
        cprint(f"This is the rewarding.", color = "green")
        # Build extra config args from CLI overrides (excluding 'config' itself)
        cli_conf = OmegaConf.from_cli()
        cli_without_config = {k: v for k, v in cli_conf.items() if k != "config"}
        extra_args = flatten_dict_to_dotlist(cli_without_config)
        extra_args_str = " ".join(extra_args)
        
        subprocess.run(
            f'python reward.py '
            f'config=../configs/{project_name}.yaml {extra_args_str}',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    def execute():
        cprint(f"This is the execution.", color = "green")
        # Build extra config args from CLI overrides (excluding 'config' itself)
        cli_conf = OmegaConf.from_cli()
        cli_without_config = {k: v for k, v in cli_conf.items() if k != "config"}
        extra_args = flatten_dict_to_dotlist(cli_without_config)
        extra_args_str = " ".join(extra_args)
        
        subprocess.run(
            f'python execute.py '
            f'config=../configs/{project_name}.yaml {extra_args_str}',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    
    
    os.makedirs(f"{project_name}/results", exist_ok=True)
    
    
    sample(model_base)
    if eval_type == "code":
        execute()
    
    reward()