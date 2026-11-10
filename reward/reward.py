import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward import math_utils
from utils.config import build_outputs_name, get_eval_config, load_config as get_config
from utils.paths import project_dir
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint

from omegaconf import OmegaConf, MISSING

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

    outputs_name = build_outputs_name(config, pretrained_model, dataset)
    output_dir = project_dir(config)
    file_name = output_dir / "temp_data" / f"outputs-{outputs_name}.json"

    with open(file_name, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Support new format with _config metadata
    _config_meta = None
    if isinstance(raw, dict) and "_config" in raw:
        _config_meta = raw["_config"]
        data = raw["data"]
    else:
        data = raw

    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    response_length_list = []
    data_type = get_eval_config(config, "data_type")
    
    for i in range(len(data)):
        
        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]
        if data_type in ("math", "option"):
            data[i]["correctness"] = []
            # check if there is a ground truth answer key in the data[i]
            if "ground_truth_answer" in data[i]:
                ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])
            else:
                ground_truth_list = ground_truth_list + [None] * len(data[i]["extracted_output"])
    

    if data_type in ("math", "option"):

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
    if data_type in ("math", "option"):
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
    
    file_name.parent.mkdir(parents=True, exist_ok=True)
    output = {"_config": _config_meta, "data": data} if _config_meta else data
    with file_name.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


    outputs_result_name = output_dir / "results" / f"results-{outputs_name}.txt"
    outputs_result_name.parent.mkdir(parents=True, exist_ok=True)
    with outputs_result_name.open("a", encoding="utf-8") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        
        avg_len = sum(response_length_list)/len(response_length_list)

        save_and_print(f"acc: {acc:.4f}\navg length: {avg_len:.4f}")

        # Decoding statistics recorded by the sampler (throughput, latency, steps).
        stats = (_config_meta or {}).get("decoding_stats")
        if stats:
            lines = [
                f"avg steps: {stats['avg_steps']:.2f}",
                f"avg steps per block: {stats['avg_steps_per_block']:.2f}",
                f"avg tokens per step: {stats['avg_tokens_per_step']:.2f}",
            ]
            if stats.get("throughput_tps") is not None:
                lines.append(f"throughput (tokens/s): {stats['throughput_tps']:.2f}")
                lines.append(f"latency (s/sample): {stats['avg_latency_s']:.3f}")
            save_and_print("\n".join(lines))