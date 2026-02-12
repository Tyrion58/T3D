import os as _os
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
# Cache root: set T3D_CACHE_ROOT to use a custom path (e.g. NVMe or /dev/shm)
_cache_root = _os.environ.get("T3D_CACHE_ROOT", _os.path.join(_os.path.expanduser("~"), ".cache", "t3d"))
_os.makedirs(_cache_root, exist_ok=True)
_os.environ["TORCH_EXTENSIONS_DIR"] = _os.path.join(_cache_root, "torch_extensions")
_triton_cache_dir = _os.path.join(_cache_root, "triton")
_os.makedirs(_triton_cache_dir, exist_ok=True)
# Clean up stale temporary directories in Triton cache to avoid OSError: [Errno 39] Directory not empty
# Triton creates tmp.* directories and tries to rename them; stale ones can cause conflicts
try:
    import shutil
    import glob
    _tmp_pattern = _os.path.join(_triton_cache_dir, "tmp.*")
    for _tmp_dir in glob.glob(_tmp_pattern):
        try:
            if _os.path.isdir(_tmp_dir):
                shutil.rmtree(_tmp_dir)
        except (OSError, PermissionError):
            pass  # Ignore errors when cleaning up (may be in use by another process)
except Exception:
    pass  # Ignore any errors during cleanup
_os.environ["TRITON_CACHE_DIR"] = _triton_cache_dir
_os.environ["XDG_CACHE_HOME"] = _cache_root
_os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


_os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
_os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
_os.environ.pop("NCCL_BLOCKING_WAIT", None)
_os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

import os
import re
import json
from termcolor import cprint
import random
import torch.multiprocessing as mp
from jinja2 import Template

from omegaconf import DictConfig, ListConfig, OmegaConf, MISSING

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def get_eval_config(config, key, default=MISSING):
    """
    Helper function to get evaluation config with fallback to old locations.
    Priority: config.evaluation.* > config.rollout.* > config.experiment.* > default
    """
    # Try evaluation section first
    eval_val = OmegaConf.select(config, f"evaluation.{key}", default=MISSING)
    if eval_val is not MISSING:
        return eval_val
    
    # Fallback to old locations for backward compatibility
    rollout_key_map = {
        "num_response_per_task": "rollout.num_response_per_task",
        "start_with_think": "rollout.start_with_think",
        "block_size": "rollout.block_size",
        "tensor_parallel_size": "rollout.tensor_parallel_size",
        "stop_token_list": "rollout.stop_token_list",
        "temperature": "rollout.temperature",
        "top_k": "rollout.top_k",
        "top_p": "rollout.top_p",
        "max_token": "rollout.max_token",
        "remasking_strategy": "rollout.remasking_strategy",
        "denoising_steps_per_block": "rollout.denoising_steps_per_block",
        "dynamic_threshold": "rollout.dynamic_threshold",
        "max_active": "rollout.max_active",
        "output_unmasking_history": "rollout.output_unmasking_history",
        "batch_size": "rollout.batch_size",
    }
    
    dataset_key_map = {
        "eval_dataset": "dataset.eval_dataset",
        "data_type": "dataset.data_type",
    }
    
    experiment_key_map = {
        "num_node": "experiment.num_node",
        "node_index": "experiment.node_index",
    }
    
    # Try rollout fallback
    if key in rollout_key_map:
        rollout_val = OmegaConf.select(config, rollout_key_map[key], default=MISSING)
        if rollout_val is not MISSING:
            return rollout_val
    
    # Try dataset fallback
    if key in dataset_key_map:
        dataset_val = OmegaConf.select(config, dataset_key_map[key], default=MISSING)
        if dataset_val is not MISSING:
            return dataset_val
    
    # Try experiment fallback
    if key in experiment_key_map:
        exp_val = OmegaConf.select(config, experiment_key_map[key], default=MISSING)
        if exp_val is not MISSING:
            return exp_val
    
    # Return default if provided
    if default is not MISSING:
        return default
    
    # Raise error if not found
    raise KeyError(f"Config key '{key}' not found in evaluation section or fallback locations")



# obtain prompt
def get_prompt(data_i):
    return Template(system_prompts).render(problem = data_i["question"])

def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"




def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


'''
def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node 
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]
'''
def get_data_chunk(data, num_nodes, node_idx):
    total = len(data)
    start = (total * node_idx) // num_nodes
    end   = (total * (node_idx + 1)) // num_nodes
    return data[start:end]


import socket

def _patch_safe_destroy():
    import torch.distributed as dist
    _real_destroy = dist.destroy_process_group
    def _safe_destroy(group=None):
        try:
            if not dist.is_available():
                return
            try:
                if not dist.is_initialized():
                    return
            except Exception:
                return
            _real_destroy(group)
        except AssertionError:
            pass
    dist.destroy_process_group = _safe_destroy



def _llm_worker_run(args):
    (model_path, tp, block_size, sampling_kwargs, vis_ids,
     prompts_slice, indices_slice, enforce_eager, max_active, store_port, batch_size) = args

    import os
    # 1) Setup environment (critical for correct worker behavior)
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.pop("NCCL_BLOCKING_WAIT", None)
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, vis_ids))
    #os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_ext_worker_{store_port}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(store_port)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 1.1) Create a per-worker `sitecustomize.py` and inject it into PYTHONPATH 
    # (this must be done before importing torch/jetengine)
    patch_dir = f"/tmp/je_site_{store_port}"
    os.makedirs(patch_dir, exist_ok=True)
    patch_file = os.path.join(patch_dir, "sitecustomize.py")
    # Important: the content must start at column 0 (no indentation)!
    with open(patch_file, "w") as _f:
        _f.write(
            "import os\n"
            "import torch.distributed as dist\n"
            "_real = dist.init_process_group\n"
            "def _wrapped(backend, init_method=None, *args, **kwargs):\n"
            "    port = os.environ.get('JE_TCP_PORT')\n"
            "    if port and isinstance(init_method, str) and init_method.startswith('tcp://localhost:2333'):\n"
            "        init_method = f'tcp://127.0.0.1:{port}'\n"
            "    return _real(backend, init_method, *args, **kwargs)\n"
            "dist.init_process_group = _wrapped\n"
        )
    os.environ["PYTHONPATH"] = patch_dir + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
    os.environ["JE_TCP_PORT"] = str(store_port)

    # 2) Import torch and patch the current worker process
    import torch
    import torch.distributed as dist
    _patch_dist_port(store_port)   # Patch port binding for this process
    _patch_safe_destroy()          # Avoid AssertionError in destroy_process_group
    torch.cuda.set_device(0)       # From this worker’s perspective, cuda:0 is the first visible device

    # For debugging: print the worker’s CUDA_VISIBLE_DEVICES and assigned port
    print(f"[worker pid={os.getpid()}] CVD={os.environ['CUDA_VISIBLE_DEVICES']}, port={store_port}, prompts={len(prompts_slice)}", flush=True)

    # 3) Import jetengine and create the engine 
    # (child processes inherit the sitecustomize patch)
    from jetengine_ext.llm import LLM
    from jetengine_ext.sampling_params import SamplingParams

    llm = None
    triples = []
    try:
        llm = LLM(
            model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tp,
            mask_token_id=151669,
            block_length=block_size
        )
        sp = SamplingParams(**sampling_kwargs)

        # Helper function to chunk into batches
        def _chunk_into_batches(lst, batch_size):
            """Split list into batches of specified size."""
            if batch_size is None or batch_size <= 0:
                return [lst]
            batches = []
            for i in range(0, len(lst), batch_size):
                batches.append(lst[i:i+batch_size])
            return batches

        # Process in batches if batch_size is set
        prompt_batches = _chunk_into_batches(prompts_slice, batch_size)
        index_batches = _chunk_into_batches(indices_slice, batch_size)
        
        if batch_size and batch_size > 0 and len(prompt_batches) > 1:
            print(f"[worker pid={os.getpid()}] Batch processing: {len(prompts_slice)} prompts in {len(prompt_batches)} batches (batch_size={batch_size})", flush=True)
        
        # Keep max_active sane for each worker's slice to avoid rare internal exits
        local_max_active = min(max_active, max(1, len(prompts_slice)))
        
        # Process each batch
        for batch_idx, (prompt_batch, index_batch) in enumerate(zip(prompt_batches, index_batches)):
            if batch_size and batch_size > 0 and len(prompt_batches) > 1:
                print(f"[worker pid={os.getpid()}] Processing batch {batch_idx+1}/{len(prompt_batches)} ({len(prompt_batch)} prompts)", flush=True)
            
            outs = llm.generate_streaming(prompt_batch, sp, max_active=local_max_active)
            
            # Collect results incrementally so we can return partials on any exit
            for j, o in enumerate(outs):
                triples.append((
                    index_batch[j],
                    o["text"],
                    o.get("first_unmask_times", None)
                ))
            
            # Clear CUDA cache between batches to free up memory
            if batch_size and batch_size > 0 and batch_idx < len(prompt_batches) - 1:
                torch.cuda.empty_cache()
    except BaseException as e:
        # Swallow SystemExit/KeyboardInterrupt/etc. so we can return partials
        print(f"[worker pid={os.getpid()}] Caught {type(e).__name__}: {e}. Returning partial results ({len(triples)})", flush=True)
    finally:
        try:
            if llm is not None and hasattr(llm, "shutdown"):
                llm.shutdown()
        except Exception:
            pass

    return triples



def _llm_worker_entry(args, out_q):
    import traceback, os
    try:
        res = _llm_worker_run(args)
        # Even if partial, report as 'ok' so parent can use what we have
        out_q.put(("ok", res))
    except BaseException:
        tb = traceback.format_exc()
        # Fall back to 'err' path if even the call above exploded
        try:
            out_q.put(("err", {
                "pid": os.getpid(),
                "port": args[-1],
                "traceback": tb,
            }))
        except Exception:
            pass


def _find_free_port():
    s = socket.socket(); s.bind(('', 0))
    p = s.getsockname()[1]; s.close()
    return p

def _patch_dist_port(port: int):
    import torch.distributed as _dist
    _real_init = _dist.init_process_group

    def _wrapped(backend, init_method=None, *args, **kwargs):
        # jetengine internally hardcodes "tcp://localhost:2333" — replace the port here
        if isinstance(init_method, str) and init_method.startswith("tcp://localhost:2333"):
            init_method = f"tcp://127.0.0.1:{port}"
        return _real_init(backend, init_method, *args, **kwargs)

    _dist.init_process_group = _wrapped



if __name__ == "__main__":

    config = get_config()

    

    tp = int(get_config().rollout.tensor_parallel_size)  # Or check after loading config

    if tp == 1:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        # These two are NCCL’s own variables, keep using the NCCL_ prefix
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    else:
        # For multi-GPU communication, do not disable P2P/IB;
        # also clean up related variables (both old and new names)
        for k in [
            "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
            "TORCH_NCCL_BLOCKING_WAIT", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_BLOCKING_WAIT", "NCCL_ASYNC_ERROR_HANDLING",
        ]:
            os.environ.pop(k, None)



    from transformers import AutoTokenizer

    # --- graceful shutdown & unique port ---
    import os, sys, atexit, signal, torch.distributed as dist

    # 2) Automatically set compile architecture according to the local GPU
    # (do NOT hardcode 8.0)
    def _set_arch():
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass
    _set_arch()

    

    # 1) Use a new port at each startup to avoid conflicts with 2333
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_find_free_port())
    # (If JetEngine hardcodes tcp://localhost:2333 instead of using env://,
    # see the “special case” section at the end)

    # 2) Intercept Ctrl-C/TERM to destroy distributed groups & engine gracefully
    _llm = None
    _child_ps = []    # If you create your own mp.Process/Pool, append objects here

    def _cleanup():
        # 2.1) Shutdown JetEngine engine (if API available)
        global _llm
        try:
            if _llm is not None and hasattr(_llm, "shutdown"):
                _llm.shutdown()
        except Exception:
            pass
        # 2.3) Kill/join child processes
        for p in _child_ps:
            try:
                if hasattr(p, "terminate"): p.terminate()
            except Exception:
                pass
        for p in _child_ps:
            try:
                if hasattr(p, "join"): p.join(timeout=2)
            except Exception:
                pass

    atexit.register(_cleanup)
    def _sig_handler(sig, frame):
        _cleanup()
        # 130: standard exit code for SIGINT; 143: for SIGTERM
        sys.exit(130 if sig == signal.SIGINT else 143)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)








    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    
    k_sample = get_eval_config(config, "num_response_per_task", 1)
    # non-cot prompt
    #system_prompts = '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant\n'''
    # cot prompt
    system_prompts = '''<|im_start|>user\n{{problem}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'''
    if get_eval_config(config, "start_with_think", False):
        system_prompts = '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. This is the problem:\n{{problem}}<|im_end|>\n<|im_start|>assistant<think>\n'''
    
    project_name = config.experiment.project

    code_eval = False

    dataset = get_eval_config(config, "eval_dataset")
    # Get checkpoint path: evaluation.checkpoint_path > model.pretrained_model > config.model (string)
    checkpoint_path = get_eval_config(config, "checkpoint_path", MISSING)
    if checkpoint_path is MISSING:
        if isinstance(config.model, str):
            pretrained_model = config.model
        else:
            pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = checkpoint_path
    
    data_type = get_eval_config(config, "data_type")
    if data_type == "code":
        code_eval = True
        system_prompts_function = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>\n<|im_start|>assistant\n'''
        system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant\n'''
        if get_eval_config(config, "start_with_think", False):
            system_prompts_stdio = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>\n<|im_start|>assistant<think>\n'''
    elif data_type == "option":
        system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant\n'''
        if get_eval_config(config, "start_with_think", False):
            system_prompts = '''<|im_start|>user\nThis is the problem:\n{{problem}}\nYou need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>\n<|im_start|>assistant<think>\n'''
    
    outputs_name = "eval-" + pretrained_model.split("/")[-1] + "-" + dataset

    with open("../data/" + dataset + ".json", 'r') as f:
        data = json.load(f)

    num_node = get_eval_config(config, "num_node", 1)
    node_index = get_eval_config(config, "node_index", 0)
    if num_node > 1:
        #random.shuffle(data)
        data = get_data_chunk(data, num_node, node_index)
    
    num = len(data)


    model_path = os.path.expanduser(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Initialize the LLM

    block_size = get_eval_config(config, "block_size", 8)
    


    

    

    



    # initialization
    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        # preprocess
        if code_eval:
            if data[i]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list = prefix_list + [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[i]["prefix"]
                prefix_list = prefix_list + [data[i]["prefix"]] * k_sample
        generation_prompts = generation_prompts + [get_prompt(data[i])] * k_sample
        
        index_list = index_list + [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = get_prompt(data[i])
    




    # --------------------------- 1. shuffle --------------------------
    cprint("start generation...", "green")

    all_prompts = generation_prompts
    N = len(all_prompts)

    shuffled_idx     = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]


    import torch, math
    print(f"[preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[preflight] parent sees torch.cuda.device_count()={torch.cuda.device_count()}")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible_gpus = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        device_ids = [int(x) for x in visible_gpus]         
    else:
        device_ids = list(range(torch.cuda.device_count()))
    
    gpu_num = len(device_ids)     
    tp = int(get_eval_config(config, "tensor_parallel_size", 1))
    assert gpu_num >= tp, f"Visible GPUs ({gpu_num}) < tensor_parallel_size ({tp})."
    assert gpu_num >= 1, "No GPU visible"
    if tp > 1:
        ngroups = 1
    else:
        ngroups = max(1, gpu_num // max(1, tp))
    
    groups = [ device_ids[i*tp : (i+1)*tp] for i in range(ngroups) ]



    def to_single_token_stop_ids(tokenizer, stop_token_list):
        if not stop_token_list:
            return []
        ids, seen = [], set()
        for s in stop_token_list:
            if isinstance(s, int):
                tid = [s]
            elif isinstance(s, str):
                tid = tokenizer.encode(s, add_special_tokens=False)
            elif isinstance(s, (list, tuple)) and all(isinstance(x, int) for x in s):
                tid = list(s)
            else:
                continue  
            if len(tid) == 1:
                t = tid[0]
                if t not in seen:
                    seen.add(t)
                    ids.append(t)
        return ids
    
    stop_token_list = get_eval_config(config, "stop_token_list", [])
    if stop_token_list:
        stop_token_id_list = to_single_token_stop_ids(tokenizer, stop_token_list)
    else:
        stop_token_id_list = []

    sampling_kwargs = dict(
        temperature          = get_eval_config(config, "temperature", 0.7),
        topk                 = get_eval_config(config, "top_k", 50),
        topp                 = get_eval_config(config, "top_p", 0.9),
        max_tokens           = get_eval_config(config, "max_token", 512),
        remasking_strategy   = get_eval_config(config, "remasking_strategy", "default"),
        block_length         = block_size,
        denoising_steps      = get_eval_config(config, "denoising_steps_per_block", 1),
        dynamic_threshold    = get_eval_config(config, "dynamic_threshold", 0.0),
        stop_words           = stop_token_id_list,
        random_init_ratio    = get_eval_config(config, "random_init_ratio", 0.0),
    )
    max_active_local = get_eval_config(config, "max_active", 256)
    batch_size = get_eval_config(config, "batch_size", None)  # None means process all at once

    def _chunk_by_groups(lst, ng):
        L = len(lst)
        if ng <= 1: return [lst]
        chunk_size = math.ceil(L / ng)
        return [ lst[i*chunk_size : min((i+1)*chunk_size, L)] for i in range(ng) ]
    
    def _chunk_into_batches(lst, batch_size):
        """Split list into batches of specified size."""
        if batch_size is None or batch_size <= 0:
            return [lst]
        batches = []
        for i in range(0, len(lst), batch_size):
            batches.append(lst[i:i+batch_size])
        return batches

    # First chunk by groups (for multi-GPU), then optionally chunk into batches (for KV cache management)
    prompt_chunks_by_group = _chunk_by_groups(shuffled_prompts, ngroups)
    index_chunks_by_group = _chunk_by_groups(shuffled_idx, ngroups)

    for a, b in zip(prompt_chunks_by_group, index_chunks_by_group):
        assert len(a) == len(b)

    seq_pairs = []

    if ngroups == 1:
        from jetengine_ext.llm import LLM
        from jetengine_ext.sampling_params import SamplingParams

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, groups[0]))
        import torch
        torch.cuda.set_device(0)

        if tp > 1:
            enforce_eager = False
        else:
            enforce_eager = True
        
        print(f"[DEBUG] Creating LLM with tp={tp}, enforce_eager={enforce_eager}, block_size={block_size}", flush=True)
        llm = LLM(
            model_path,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tp,
            mask_token_id=151669,   # Optional: only needed for masked/diffusion models
            block_length=block_size
        )
        print(f"[DEBUG] LLM created successfully", flush=True)
        _llm = llm

        # Set sampling/generation parameters
        sampling_params = SamplingParams(
            temperature=get_eval_config(config, "temperature", 0.7),
            topk=get_eval_config(config, "top_k", 50),
            topp=get_eval_config(config, "top_p", 0.9),
            max_tokens=get_eval_config(config, "max_token", 512),
            remasking_strategy=get_eval_config(config, "remasking_strategy", "default"),
            block_length=block_size,
            denoising_steps=get_eval_config(config, "denoising_steps_per_block", 1),
            dynamic_threshold=get_eval_config(config, "dynamic_threshold", 0.0),
            stop_words=stop_token_id_list,
            random_init_ratio=get_eval_config(config, "random_init_ratio", 0.0),
        )
        # Process in batches if batch_size is set
        prompt_batches = _chunk_into_batches(prompt_chunks_by_group[0], batch_size)
        index_batches = _chunk_into_batches(index_chunks_by_group[0], batch_size)
        
        print(f"[DEBUG] batch_size={batch_size}, prompt_batches={len(prompt_batches)}, total_prompts={len(prompt_chunks_by_group[0])}", flush=True)
        
        if batch_size and batch_size > 0 and len(prompt_batches) > 1:
            print(f"[Batch Processing] Processing {len(prompt_chunks_by_group[0])} prompts in {len(prompt_batches)} batches "
                  f"(batch_size={batch_size}) to avoid KV cache exhaustion", flush=True)
        else:
            print(f"[DEBUG] Starting generation with {len(prompt_chunks_by_group[0])} prompts, max_active={max_active_local}", flush=True)
        
        try:
            total_processed = 0
            for batch_idx, (prompt_batch, index_batch) in enumerate(zip(prompt_batches, index_batches)):
                if batch_size and len(prompt_batches) > 1:
                    print(f"[Batch {batch_idx+1}/{len(prompt_batches)}] Processing {len(prompt_batch)} prompts "
                          f"(total progress: {total_processed}/{len(prompt_chunks_by_group[0])})", flush=True)
                
                outputs = llm.generate_streaming(prompt_batch, sampling_params, max_active=max_active_local)
                if batch_size and batch_size > 0 and len(prompt_batches) > 1:
                    print(f"[Batch {batch_idx+1}/{len(prompt_batches)}] Completed {len(outputs)} outputs", flush=True)
                for j, o in enumerate(outputs):
                    seq_pairs.append( (
                        index_batch[j],
                        o["text"],
                        o.get("first_unmask_times", None)
                    ) )
                total_processed += len(outputs)
                
                # Clear CUDA cache between batches to free up memory
                if batch_size and batch_size > 0 and batch_idx < len(prompt_batches) - 1:
                    import torch
                    torch.cuda.empty_cache()
        finally:
            _cleanup()
    else:
        import time
        ctx = mp.get_context("spawn")
        enforce_eager_local = False if tp > 1 else True

        base_port = 29000
        store_ports = [base_port + g for g in range(ngroups)]

        out_q = ctx.Queue()
        procs = []
        for g in range(ngroups):
            if len(prompt_chunks_by_group[g]) == 0:
                continue
            args = (
                model_path, tp, block_size, sampling_kwargs, groups[g],
                prompt_chunks_by_group[g], index_chunks_by_group[g],
                enforce_eager_local, max_active_local, store_ports[g], batch_size,
            )
            p = ctx.Process(target=_llm_worker_entry, args=(args, out_q), daemon=False)
            p.start()
            #time.sleep(2)
            procs.append(p)
            _child_ps.append(p)   

        import queue, time

        results_needed = len(procs)
        results_got = 0

        while results_got < results_needed:
            try:
                kind, payload = out_q.get(timeout=3600 * 24) 
            except queue.Empty:
                dead = [p for p in procs if not p.is_alive()]
                if dead:
                    for p in dead:
                        print(f"[parent] worker pid={p.pid} exitcode={p.exitcode} (no result)", flush=True)
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    for p in procs:
                        p.join(timeout=5)
                    raise RuntimeError("Some workers died without returning results. See logs above.")
                continue

            if kind == "ok":
                seq_pairs.extend(payload)
                results_got += 1
            else:  # "err"
                print(f"[parent] worker error on port {payload['port']} pid {payload['pid']}:\n{payload['traceback']}", flush=True)
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join(timeout=5)
                raise RuntimeError("Worker failed. See traceback above.")

        for p in procs:
            p.join()


    # ------------------- 3. restore original order -------------------


    restored_outputs = [None] * N
    restored_steps   = [None] * N

    for item in seq_pairs:
        if len(item) == 2:
            gi, text = item
            steps = None
        else:
            gi, text, steps = item
        restored_outputs[gi] = text
        restored_steps[gi]   = steps


    for i in range(N):
        if restored_outputs[i] is None:
            restored_outputs[i] = ""
        if restored_steps[i] is None:
            restored_steps[i] = ""

    cprint("generation job done!", "green")






    def get_token_lengths(strings, tokenizer):
        pad_token = tokenizer.pad_token

        escaped = re.escape(pad_token)
        pattern = rf"(?:{escaped})+"
        remove_pattern = escaped

        collapse_re = re.compile(pattern)

        lengths = []
        for s in strings:
            s_clean = collapse_re.sub(lambda _: pad_token if isinstance(pad_token, str) else '', s)
            s_clean = re.sub(remove_pattern, '', s_clean)
            lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
        return lengths

    response_length = get_token_lengths(restored_outputs, tokenizer)
    mean_response_length = sum(response_length) / len(response_length)




    # process generated codes
    i = 0
    for full_output in restored_outputs:
        if code_eval:
            if data[int(i/k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[i] + full_output)
            else:
                extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)
        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        step_map_i = restored_steps[i] if restored_steps[i] is not None else []
        #print(step_map_i)
        data[index_i]["step_map"].append(step_map_i)
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        i += 1

    # output the data
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

