import json
import os
import math_utils
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint

from omegaconf import OmegaConf, MISSING

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def get_eval_config(config, key, default=MISSING):
    """
    Helper function to get evaluation config with fallback to old locations.
    Priority: config.evaluation.* > config.dataset.* > config.rollout.* > default
    """
    # Try evaluation section first
    eval_val = OmegaConf.select(config, f"evaluation.{key}", default=MISSING)
    if eval_val is not MISSING:
        return eval_val
    
    # Fallback to old locations for backward compatibility
    dataset_key_map = {
        "eval_dataset": "dataset.eval_dataset",
        "data_type": "dataset.data_type",
    }
    
    rollout_key_map = {
        "output_unmasking_history": "rollout.output_unmasking_history",
    }
    
    # Try dataset fallback
    if key in dataset_key_map:
        dataset_val = OmegaConf.select(config, dataset_key_map[key], default=MISSING)
        if dataset_val is not MISSING:
            return dataset_val
    
    # Try rollout fallback
    if key in rollout_key_map:
        rollout_val = OmegaConf.select(config, rollout_key_map[key], default=MISSING)
        if rollout_val is not MISSING:
            return rollout_val
    
    # Return default if provided
    if default is not MISSING:
        return default
    
    # Raise error if not found
    raise KeyError(f"Config key '{key}' not found in evaluation section or fallback locations")

if __name__ == "__main__":

    config = get_config()

    project_name = config.experiment.project
    

    dataset = get_eval_config(config, "eval_dataset")
    # Get checkpoint path: evaluation.checkpoint_path > model.pretrained_model > config.model (string)
    checkpoint_path = get_eval_config(config, "checkpoint_path", MISSING)
    if checkpoint_path is MISSING or checkpoint_path is None:
        # Support both config.model (string) and config.model.pretrained_model (dict)
        if isinstance(config.model, str):
            pretrained_model = config.model
        else:
            pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = checkpoint_path
    
    # Validate that pretrained_model is set
    if pretrained_model is None:
        raise ValueError(
            "pretrained_model is None. Please set either:\n"
            "  - evaluation.checkpoint_path in config, or\n"
            "  - model (as string) or model.pretrained_model in config"
        )

    outputs_name = "eval-" + pretrained_model.split("/")[-1] + "-" + dataset
    output_base = OmegaConf.select(config, "experiment.output_dir", default=None) or getattr(config.experiment, "output_dir", "..")
    output_base = os.path.expanduser(str(output_base))
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(output_base):
        output_base = os.path.normpath(os.path.join(_script_dir, output_base))
    file_name = os.path.join(output_base, project_name, "temp_data", "outputs-" + outputs_name + ".json")

    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    response_length_list = []
    data_type = get_eval_config(config, "data_type")
    
    for i in range(len(data)):
        
        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]
        if data_type == "math":
            data[i]["correctness"] = []
            # check if there is a ground truth answer key in the data[i]
            if "ground_truth_answer" in data[i]:
                ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])
            else:
                ground_truth_list = ground_truth_list + [None] * len(data[i]["extracted_output"])
    

    if data_type == "math":

        nest_asyncio.apply()

        async def get_correctness():
            executor = ThreadPoolExecutor(max_workers=64)
            tasks = []
            for i in range(len(index_list)):
                tasks.append(math_utils.is_equal(extracted_output_list[i], ground_truth_list[i], executor))
            results = await asyncio.gather(*tasks)
            return results
    
        correctness_list = asyncio.run(get_correctness())
        for i in range(len(index_list)):
            index_i = index_list[i]
            data[index_i]["correctness"].append(correctness_list[i])



    def z_score_normalize(lst):
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]



    data_type = get_eval_config(config, "data_type")
    if data_type == "math":
        acc = sum(correctness_list)/len(correctness_list)
    else:
        num_task   = 0
        num_correct_task = 0
        for x in data:
            for y in x["correctness"]:
                num_correct_task += all(y)
                num_task += 1
        acc = num_correct_task / num_task if num_task else 0

    if not get_eval_config(config, "output_unmasking_history", True):
        for i in range(len(data)):
            data[i]["step_map"] = []
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    outputs_result_name = os.path.join(output_base, project_name, "results", "results-" + outputs_name + ".txt")
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        
        avg_len = sum(response_length_list)/len(response_length_list)

        save_and_print(f"acc: {acc:.4f}\navg length: {avg_len:.4f}")