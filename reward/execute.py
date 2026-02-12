import argparse, ast, io, json, os, sys, time, textwrap, multiprocessing as mp
import concurrent.futures as cf
from pathlib import Path
import math
import numpy as np
from termcolor import cprint
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf, MISSING


def get_config():
    cli_conf   = OmegaConf.from_cli()
    yaml_conf  = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)






from concurrent.futures import as_completed

import textwrap

def _run_many_pipe(snippet: str, tests: list[str], conn):
    import textwrap
    import sys
    import traceback
    results = []
    debug_info = []
    
    try:
        ns = {}
        # Normalize line endings to \n before dedent and execution
        normalized_snippet = snippet.replace('\r\n', '\n').replace('\r', '\n')
        dedented_snippet = textwrap.dedent(normalized_snippet)
        
        debug_info.append(f"Original snippet (repr): {repr(snippet[:200])}")
        debug_info.append(f"Normalized snippet (repr): {repr(normalized_snippet[:200])}")
        debug_info.append(f"Dedented snippet (repr): {repr(dedented_snippet[:200])}")
        
        # Execute the snippet
        exec(dedented_snippet, ns, ns)
        debug_info.append(f"Functions in namespace after exec: {[k for k in ns.keys() if not k.startswith('__')]}")
        
        # Run tests
        for idx, stmt in enumerate(tests):
            try:
                exec(stmt, ns, ns)
                results.append(True)
                debug_info.append(f"Test {idx+1}/{len(tests)} '{stmt}' PASSED")
            except SystemExit:
                results.append(True)
                debug_info.append(f"Test {idx+1}/{len(tests)} '{stmt}' PASSED (SystemExit)")
            except AssertionError as e:
                results.append(False)
                debug_info.append(f"Test {idx+1}/{len(tests)} '{stmt}' FAILED (AssertionError): {e}")
            except Exception as e:
                results.append(False)
                debug_info.append(f"Test {idx+1}/{len(tests)} '{stmt}' FAILED ({type(e).__name__}): {e}")
                debug_info.append(f"  Traceback: {traceback.format_exc()}")
        
        # Send results with debug info
        conn.send((results, debug_info))
    except SystemExit as e:
        debug_info.append(f"Snippet execution SystemExit: {e}")
        conn.send(([True] * len(tests), debug_info))
    except Exception as e:
        debug_info.append(f"Snippet execution FAILED ({type(e).__name__}): {e}")
        debug_info.append(f"  Traceback: {traceback.format_exc()}")
        conn.send(([False] * len(tests), debug_info))
    finally:
        try: conn.close()
        except Exception: pass


def _check_snippet_many(snippet: str, tests: list[str], t_limit: int,
                        spawn_slack: float = 5.0, debug_file=None) -> tuple[list[bool], list[str]]:
    import time, multiprocessing as mp
    import os, sys
    ctx = mp.get_context("spawn") 
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_run_many_pipe, args=(snippet, tests, child_conn), daemon=True)
    
    # Start process and close child connection
    try:
        p.start()
    except Exception as e:
        child_conn.close()
        parent_conn.close()
        return ([False] * len(tests), [f"Failed to start process: {e}"])
    
    child_conn.close()

    # Increase timeout: t_limit for execution + spawn_slack for process startup/overhead
    deadline = time.monotonic() + t_limit + spawn_slack
    res = None
    debug_info = []
    process_exited = False
    
    try:
        # Poll for result with timeout
        while time.monotonic() < deadline:
            if parent_conn.poll(0.1):  # Check every 100ms
                try:
                    res = parent_conn.recv()
                    break
                except (EOFError, OSError) as e:
                    # Connection closed or error
                    debug_info.append(f"Connection error: {e}")
                    res = None
                    break
            
            # Check if process has exited
            if not p.is_alive():
                process_exited = True
                # Try to get result one more time
                if parent_conn.poll(0.1):
                    try:
                        res = parent_conn.recv()
                        break
                    except (EOFError, OSError):
                        pass
                break

        # Final attempt to get result
        if res is None and parent_conn.poll(0.1):
            try:
                res = parent_conn.recv()
            except (EOFError, OSError):
                pass

        # If no result, mark as timeout
        if res is None:
            if p.is_alive():
                try:
                    p.terminate()
                    p.join(timeout=1.0)
                    if p.is_alive():
                        p.kill()
                except Exception:
                    pass
            timeout_msg = "Timeout or no response from child process"
            if process_exited:
                timeout_msg += " (process exited without sending result)"
            res = ([False] * len(tests), [timeout_msg])
        else:
            # Handle both old format (list) and new format (tuple)
            if isinstance(res, tuple):
                res, debug_info = res
            else:
                debug_info = ["Legacy format: no debug info"]
    except Exception as e:
        # Unexpected error
        try:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        except Exception:
            pass
        res = ([False] * len(tests), [f"Unexpected error in _check_snippet_many: {e}"])
        debug_info = []
    finally:
        # Cleanup
        try:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        except Exception:
            pass
        try:
            parent_conn.close()
        except Exception:
            pass

    # Write debug info to file if provided
    if debug_file:
        try:
            # Ensure directory exists
            debug_dir = os.path.dirname(debug_file)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Snippet (first 200 chars): {repr(snippet[:200])}\n")
                f.write(f"Tests: {tests}\n")
                f.write(f"Results: {res}\n")
                for line in debug_info:
                    f.write(line + "\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            # Don't fail silently - at least print to stderr
            import sys
            print(f"Warning: Could not write to debug file {debug_file}: {e}", file=sys.stderr)

    return ([bool(x) for x in res], debug_info)

from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_function_dataset(data: list[dict], n_workers: int | None = None, debug_file: str | None = None):
    import os
    import time
    n_cpu = os.cpu_count() or 4
    n_workers = max(1, int(n_workers)) if n_workers is not None else n_cpu

    # Create debug file if not provided
    if debug_file is None:
        debug_file = "./tmp/execute_debug.log"
    
    # Ensure directory exists
    debug_dir = os.path.dirname(debug_file)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
    # Clear debug file at start
    try:
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"Debug log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        print(f"Debug log file: {os.path.abspath(debug_file)}")
    except Exception as e:
        print(f"Warning: Could not create debug file {debug_file}: {e}", file=sys.stderr)

    for item in data:
        m_code = len(item["extracted_output"])
        m_test = len(item["test_list"])
        item["execution_result"] = [[None]  * m_test for _ in range(m_code)]
        item["correctness"]      = [[False] * m_test for _ in range(m_code)]
        item.setdefault("step_map", [])

    tasks = []
    for idx, item in enumerate(data):
        t_limit = item.get("test_time_limit", 1)
        tests   = item["test_list"]
        for i, snippet in enumerate(item["extracted_output"]):
            tasks.append((idx, i, snippet, tests, t_limit))

    futures = {}
    from tqdm.auto import tqdm
    with ThreadPoolExecutor(max_workers=n_workers) as pool, \
        tqdm(total=len(tasks)*len(data[0]["test_list"]), desc=f"Function tests ({n_workers} threads)",
            dynamic_ncols=True, mininterval=0.1, miniters=1) as pbar:

        for idx, i, snippet, tests, t_limit in tasks:
            fut = pool.submit(_check_snippet_many, snippet, tests, t_limit, debug_file=debug_file)
            futures[fut] = (idx, i)

        for fut in as_completed(futures):
            idx, i = futures[fut]
            try:
                result = fut.result()
                # Handle both old format (list) and new format (tuple)
                if isinstance(result, tuple):
                    ok_list, debug_info = result
                    # Write debug info to file for failed cases
                    if any(not ok for ok in ok_list):
                        try:
                            with open(debug_file, 'a', encoding='utf-8') as f:
                                f.write(f"\n[FAILED CASE] Item {idx}, Code {i}\n")
                                f.write(f"Tests: {data[idx]['test_list']}\n")
                                f.write(f"Results: {ok_list}\n")
                                for line in debug_info:
                                    f.write(line + "\n")
                        except Exception:
                            pass
                else:
                    ok_list = result
            except Exception as e:
                ok_list = [False] * len(data[idx]["test_list"])
                try:
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n[EXCEPTION] Item {idx}, Code {i}: {e}\n")
                except Exception:
                    pass

            # Ensure ok_list is a list and matches the expected number of tests
            expected_tests = len(data[idx]["test_list"])
            if not isinstance(ok_list, list):
                ok_list = [False] * expected_tests
                try:
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n[TYPE ERROR] Item {idx}, Code {i}: "
                               f"ok_list is not a list, got {type(ok_list)}\n")
                except Exception:
                    pass
            elif len(ok_list) != expected_tests:
                # Log the mismatch for debugging before fixing
                original_len = len(ok_list)
                try:
                    with open(debug_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n[LENGTH MISMATCH] Item {idx}, Code {i}: "
                               f"Expected {expected_tests} tests, got {original_len} results\n")
                except Exception:
                    pass
                # Pad or truncate to match expected length
                if len(ok_list) < expected_tests:
                    ok_list = ok_list + [False] * (expected_tests - len(ok_list))
                else:
                    ok_list = ok_list[:expected_tests]
            
            # Ensure execution_result and correctness lists are properly initialized
            if i >= len(data[idx]["execution_result"]):
                # Extend if needed
                while len(data[idx]["execution_result"]) <= i:
                    data[idx]["execution_result"].append([None] * expected_tests)
                    data[idx]["correctness"].append([False] * expected_tests)
            elif len(data[idx]["execution_result"][i]) != expected_tests:
                # Resize if needed
                data[idx]["execution_result"][i] = [None] * expected_tests
                data[idx]["correctness"][i] = [False] * expected_tests
            
            for j, ok in enumerate(ok_list):
                if j < len(data[idx]["execution_result"][i]):
                    data[idx]["execution_result"][i][j] = bool(ok)
                    data[idx]["correctness"][i][j]      = bool(ok)
                pbar.update(1)

    return data





def worker_stdio(script, input_val, output_queue):
    # Create an iterator over the input lines.
    input_lines = iter(input_val.splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")
    
    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin



def run_scripts_with_timeout(scripts, inputs, time_limits, worker, batch_size=100):
    results = [None] * len(scripts)
    n = len(scripts)
    
    # Process in batches to avoid exhausting file descriptors
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        processes = []
        queues = []
        deadlines = []
        
        for i in range(batch_start, batch_end):
            q = mp.Queue()
            p = mp.Process(target=worker, args=(scripts[i], inputs[i], q))
            processes.append(p)
            queues.append(q)
            p.start()
            deadlines.append(time.time() + time_limits[i])

        while any(p.is_alive() for p in processes):
            now = time.time()
            for idx, p in enumerate(processes):
                i = batch_start + idx
                if p.is_alive() and now >= deadlines[idx]:
                    p.terminate()
                    results[i] = "Timeout Error"
            time.sleep(0.001)

        for idx, p in enumerate(processes):
            i = batch_start + idx
            if results[i] is None:
                try:
                    results[i] = queues[idx].get_nowait()
                except Exception as e:
                    results[i] = f"Execution Error: {e}"
        
        # Clean up processes
        for p in processes:
            try:
                p.join(timeout=0.1)
            except Exception:
                pass

    return results

def test_if_eq(x, y): 
    return " ".join(x.split()) == " ".join(y.split())

def get_chunk_indices(n, num_chunks):
    size, rem = divmod(n, num_chunks)
    idx, start = [], 0
    for i in range(num_chunks):
        extra = 1 if i < rem else 0
        end   = start + size + extra
        idx.append((start, end)); start = end
    return idx







from tqdm import tqdm   

def run_scripts_with_chunk(code_list, test_input_list, time_limit_list,
                           worker, num_chunks):
    chunks = get_chunk_indices(len(code_list), num_chunks)

    exe_results = []
    pbar = tqdm(total=len(code_list), desc=f"STDIO tests ({num_chunks} ch)")  

    for start, end in chunks:
        sub_code_list       = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]

        sub_exe_results = run_scripts_with_timeout(
            sub_code_list,
            sub_test_input_list,
            sub_time_limit_list,
            worker
        )
        exe_results.extend(sub_exe_results)
        pbar.update(end - start)  

    pbar.close()          
    return exe_results


def evaluate_stdio_dataset(data: list[dict], num_chunks: int):
    
    idx_code, idx_case = [], []
    code_list, inp_list, tl_list = [], [], []

    for idx, item in enumerate(data):
        tl = item.get("test_time_limit", 1)
        m_code = len(item["extracted_output"])
        m_case = len(item["test_input"])

        data[idx]["execution_result"] = [[] for _ in range(m_code)]
        data[idx]["correctness"] = [[] for _ in range(m_code)]
        item.setdefault("step_map",           [])

        for c_idx, code in enumerate(item["extracted_output"]):
            for k in range(m_case):
                idx_code.append((idx, c_idx)) 
                idx_case.append(k)           
                code_list.append(code)
                inp_list.append(item["test_input"][k])
                tl_list.append(tl)

    exe_results = run_scripts_with_chunk(
        code_list, inp_list, tl_list, worker_stdio, num_chunks
    )

    for i, res in enumerate(exe_results):
        idx, c_idx = idx_code[i]
        k          = idx_case[i]
        item       = data[idx]

        while len(item["execution_result"][c_idx]) < k + 1:
            item["execution_result"][c_idx].append("")
            item["correctness"][c_idx].append(False)
        item["execution_result"][c_idx][k] = res
        exp_out = item["test_output"][k]
        item["correctness"][c_idx][k]      = test_if_eq(res, exp_out)

    return data




def main():
    cfg          = get_config()
    project_name = cfg.experiment.project
    
    # Get checkpoint path: evaluation.checkpoint_path > model.pretrained_model > config.model (string)
    checkpoint_path = OmegaConf.select(cfg, "evaluation.checkpoint_path", default=MISSING)
    if checkpoint_path is MISSING or checkpoint_path is None:
        # Support both config.model (string) and config.model.pretrained_model (dict)
        if isinstance(cfg.model, str):
            pretrained_model = cfg.model
        else:
            pretrained_model = cfg.model.pretrained_model
    else:
        pretrained_model = checkpoint_path
    
    # Validate that pretrained_model is set
    if pretrained_model is None:
        raise ValueError(
            "pretrained_model is None. Please set either:\n"
            "  - evaluation.checkpoint_path in config, or\n"
            "  - model (as string) or model.pretrained_model in config"
        )
    
    dataset = OmegaConf.select(cfg, "evaluation.eval_dataset", default=None)
    if dataset is None:
        raise ValueError("evaluation.eval_dataset is required in config")
    
    outputs_name = "eval-" + pretrained_model.split("/")[-1] + "-" + dataset

    output_base = OmegaConf.select(cfg, "experiment.output_dir", default=None) or getattr(cfg.experiment, "output_dir", "..")
    output_base = os.path.expanduser(str(output_base))
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(output_base):
        output_base = os.path.normpath(os.path.join(_script_dir, output_base))

    num_node = cfg.evaluation.num_node
    node_index = cfg.evaluation.node_index

    if num_node > 1:
        file_name = os.path.join(output_base, project_name, "temp_data", f"outputs-{node_index}-{outputs_name}.json")
    else:
        file_name = os.path.join(output_base, project_name, "temp_data", f"outputs-{outputs_name}.json")

    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    func_items  = [itm for itm in data if itm.get("test_method","function") == "function"]
    stdio_items = [itm for itm in data if itm.get("test_method") == "stdio"]

    # --- 1) function ---
    if func_items:
        updated_func = evaluate_function_dataset(func_items, n_workers=cfg.execute.num_chunk)
        
        func_iter = iter(updated_func)
        for i,it in enumerate(data):
            if it.get("test_method","function") == "function":
                data[i] = next(func_iter)


    # --- 2) stdio ---
    if stdio_items:
        total_scripts = sum(len(it["extracted_output"]) for it in stdio_items)
        num_chunks    = max(1, math.ceil(total_scripts / cfg.execute.num_chunk))
        updated_stdio = evaluate_stdio_dataset(stdio_items, num_chunks=num_chunks)
        it_stdio = iter(updated_stdio)
        for i, it in enumerate(data):
            if it.get("test_method") == "stdio":
                data[i] = next(it_stdio)
    
    

    # --- save JSON ---
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8", errors="surrogatepass") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    

    

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  
    main()