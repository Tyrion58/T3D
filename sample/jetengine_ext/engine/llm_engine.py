import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
# Added imports for profiling
import torch
from torch import nn
from contextlib import nullcontext
import torch.profiler as torch_profiler

from jetengine_ext.config import Config
from jetengine_ext.sampling_params import SamplingParams
from jetengine_ext.engine.sequence import Sequence, RunType
from jetengine_ext.engine.scheduler import Scheduler
from jetengine_ext.engine.model_runner import ModelRunner
from jetengine_ext.utils.loader import load_from_hf_model


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        config.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.pad_token_id
        assert config.mask_token_id is not None, "Model tokenizer must have a mask_token_id or pad_token_id"

        self.config = config
        self.scheduler = Scheduler(config)
        self.scheduler.consistent_sampling_params = False
        atexit.register(self.exit)

    def offload_parameters(self, include_buffers: bool = False):
        """
        Replace all parameter (and buffer) storages with meta tensors.
        Keeps shapes/dtypes, frees GPU/CPU memory.
        """

        def offload_parameters_keep_buffers(model: torch.nn.Module):
            """
            Move *parameters* to meta to free memory while keeping buffers unchanged.
            Works for any module tree.
            """
            # 1) Snapshot real buffers (module reference + buffer name + tensor)
            saved_buffers = []
            for mod in model.modules():
                for bname, buf in list(mod._buffers.items()):
                    if buf is not None:
                        saved_buffers.append((mod, bname, buf))

            # 2) Move everything to meta
            model.to_empty(device=torch.device("meta"))

            # 3) Restore the saved, real buffers
            for mod, bname, buf in saved_buffers:
                # Reattach the original tensor (device/dtype preserved)
                mod._buffers[bname] = buf

            torch.cuda.empty_cache()
        if include_buffers:
            self.model_runner.model.to_empty(device=torch.device("meta"))
        else:
            offload_parameters_keep_buffers(self.model_runner.model)

        print("Successfully cleaned old parameters (buffers kept)." if not include_buffers
              else "Successfully cleaned old parameters and buffers.")

    def reload_parameters(self, hf_model: nn.Module):
        load_from_hf_model(self.model_runner.model, hf_model=hf_model)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if isinstance(prompt, list):
            if self.tokenizer.pad_token_id in prompt:
                start = prompt.index(self.tokenizer.pad_token_id) + 1
                prompt = prompt[start:]
        seq = Sequence(prompt, self.config.mask_token_id, sampling_params)
        seq.eos_token_id = self.tokenizer.eos_token_id
        seq.vocab_size = self.tokenizer.vocab_size
        seq._apply_random_init_to_first_block()  # Apply random initialization to first block
        self.scheduler.add(seq)

    def step(self):
        # CRITICAL: Always cleanup finished sequences before scheduling
        # This ensures KV cache is freed immediately when sequences complete
        # This is especially important when resources are tight
        self.scheduler._cleanup_finished_sequences()
        
        scheduled_seqs, run_type = self.scheduler.schedule()
        if scheduled_seqs is None:
            # Debug: print scheduler state when nothing can be scheduled
            waiting = sum(1 for s in self.scheduler.running if s.status.name == "WAITING")
            prefilling = sum(1 for s in self.scheduler.running if s.status.name == "PREFILLING")
            denoising = sum(1 for s in self.scheduler.running if s.status.name == "DENOISING")
            saving = sum(1 for s in self.scheduler.running if s.status.name == "SAVING")
            free_blocks = len(self.scheduler.block_manager.free_block_ids)
            total_blocks = len(self.scheduler.block_manager.blocks)
            free_ratio = free_blocks / max(total_blocks, 1)
            if self.scheduler.running:
                print(f"[DEBUG step] Nothing scheduled! running={len(self.scheduler.running)} "
                      f"(W={waiting},P={prefilling},D={denoising},S={saving}) "
                      f"free_blocks={free_blocks}/{total_blocks} ({free_ratio:.1%})", flush=True)
            return [], 0 # Nothing to run

        logits = self.model_runner.call("run", scheduled_seqs, run_type)
        self.scheduler.postprocess(scheduled_seqs, logits, run_type)
        
        # Cleanup again after postprocess in case sequences finished during processing
        self.scheduler._cleanup_finished_sequences()
        
        #finished_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_seqs if seq.is_finished]
        
        finished_outputs = [
            (seq.seq_id, seq.completion_token_ids, seq.first_unmask_steps)
            for seq in scheduled_seqs
            if seq.is_finished
        ]

        # Throughput calculation needs to be adapted for block-wise generation
        num_tokens = [self.scheduler.running[i].num_to_transfer if hasattr(self.scheduler.running[i], 'num_to_transfer') else 0 for i in range(len(self.scheduler.running))]
        return finished_outputs, sum(num_tokens)

    def is_finished(self):
        return self.scheduler.is_finished()




    def _clean_token_ids(self, token_ids):
        # Accept tensors, numpy ints, etc.
        try:
            token_ids = list(token_ids)
        except Exception:
            token_ids = [token_ids]
        
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        mask_id = getattr(self.config, "mask_token_id", None)

        cleaned = []
        for t in token_ids:
            if t is None or t < 0 or t == mask_id or t >= vocab_size:
                if t not in special_ids:
                    cleaned.append(0)
                    continue
            cleaned.append(t)
        return cleaned

    def _safe_decode(self, token_ids):
        ids = self._clean_token_ids(token_ids)
        # skip_special_tokens can be True or False; doesn't affect the None issue
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    


    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
    ) -> list[str]:
        # ... (This method remains largely the same, but the progress bar will update differently) ...
        # The logic inside the `while not self.is_finished()` loop correctly calls `self.step()`
        # and collects outputs.
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            self.scheduler.consistent_sampling_params = True
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        
        total_generated_tokens = 0
        start_time = perf_counter()

        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        with prof_ctx as prof:
            while not self.is_finished():
                output, num_processed = self.step()
                if profile:
                    prof.step()
                total_generated_tokens += num_processed
                
                throughput = total_generated_tokens / (perf_counter() - start_time)
                if use_tqdm:
                    pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})

                #for seq_id, token_ids in output:
                #    outputs[seq_id] = token_ids
                for seq_id, token_ids, unmask_times in output:
                    outputs[seq_id] = {"token_ids": token_ids, "unmask_times": unmask_times}
                    if use_tqdm:
                        pbar.update(1)

        #outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        #outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [
            {
                "text": self._safe_decode(item["token_ids"]),
                "token_ids": self._clean_token_ids(item["token_ids"]),
                "first_unmask_times": item["unmask_times"],   # 与 token_ids 等长
            }
            for item in outputs
        ]

        if use_tqdm:
            pbar.close()
        return outputs

    def generate_streaming(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        max_active: int | None = None,
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
    ) -> list[str]:
        """
        Stream prompts through the engine while keeping up to `max_active` sequences running.
        As sequences finish, new prompts are added from the pending list to maximize GPU utilization.
        """
        total = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * total
            self.scheduler.consistent_sampling_params = True

        if max_active is None:
            max_active = getattr(self.scheduler, "max_num_seqs", 32)

        if use_tqdm:
            pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)

        outputs: dict[int, list[int]] = {}
        pending_idx = 0

        # Prime initial requests up to capacity
        initial = min(max_active, total)
        for i in range(initial):
            self.add_request(prompts[i], sampling_params[i])
        pending_idx = initial

        total_generated_tokens = 0
        start_time = perf_counter()

        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        import time
        consecutive_empty_steps = 0
        last_progress_time = perf_counter()
        
        # Estimate blocks needed per sequence (rough estimate)
        block_size = self.scheduler.block_manager.block_size
        total_blocks = len(self.scheduler.block_manager.blocks)
        avg_blocks_per_seq = max(1, (self.config.max_model_len + block_size - 1) // block_size // 4)
        
        # Dynamic max_active adjustment based on available resources
        effective_max_active = max_active
        
        with prof_ctx as prof:
            while not self.is_finished() or pending_idx < total:
                # Top up to capacity before each step, but check KV cache availability
                running = getattr(self.scheduler, "running", [])
                free_blocks = len(self.scheduler.block_manager.free_block_ids)
                free_ratio = free_blocks / max(total_blocks, 1)
                
                # Dynamic adjustment of max_active based on KV cache availability
                if free_ratio < 0.1:  # Less than 10% free
                    effective_max_active = max(1, len(running))  # Don't add more, just maintain current
                elif free_ratio < 0.2:  # Less than 20% free
                    effective_max_active = min(max_active, max(1, int(len(running) * 1.2)))
                else:
                    effective_max_active = max_active
                
                deficit = effective_max_active - len(running)
                
                # Only add new requests if we have enough KV cache headroom
                # Reserve blocks more aggressively when resources are low
                if free_ratio < 0.1:
                    min_free_blocks_required = len(running) * 3  # Very conservative
                elif free_ratio < 0.2:
                    min_free_blocks_required = max(avg_blocks_per_seq, len(running) * 2)
                else:
                    min_free_blocks_required = max(avg_blocks_per_seq, len(running))
                
                while deficit > 0 and pending_idx < total and free_blocks > min_free_blocks_required:
                    self.add_request(prompts[pending_idx], sampling_params[pending_idx])
                    pending_idx += 1
                    deficit -= 1
                    free_blocks -= avg_blocks_per_seq  # Rough estimate

                output, num_processed = self.step()
                if profile:
                    prof.step()
                total_generated_tokens += num_processed

                # Track progress and detect stalls
                if len(output) > 0 or num_processed > 0:
                    consecutive_empty_steps = 0
                    last_progress_time = perf_counter()
                else:
                    consecutive_empty_steps += 1
                    # If no progress for many steps, add a small sleep to avoid CPU spinning
                    if consecutive_empty_steps > 100:
                        time.sleep(0.01)
                    
                    # Re-check free blocks after potential cleanup
                    free_blocks = len(self.scheduler.block_manager.free_block_ids)
                    total_blocks = len(self.scheduler.block_manager.blocks)
                    free_ratio = free_blocks / max(total_blocks, 1)
                    
                    # Aggressive cleanup when resources are low: check every N steps
                    if free_ratio < 0.15 and consecutive_empty_steps % 50 == 0:
                        # Force cleanup check
                        cleaned = self.scheduler._cleanup_finished_sequences()
                        if cleaned > 0:
                            print(f"[Aggressive Cleanup] Freed {cleaned} sequences, new free_blocks={len(self.scheduler.block_manager.free_block_ids)}", flush=True)
                            consecutive_empty_steps = 0  # Reset counter if we made progress
                    
                    # Warn if stuck for too long
                    if consecutive_empty_steps == 1000:
                        stall_time = perf_counter() - last_progress_time
                        print(f"[WARNING] No progress for {consecutive_empty_steps} steps ({stall_time:.1f}s). "
                              f"pending_idx={pending_idx}/{total}, running={len(self.scheduler.running)}, "
                              f"free_blocks={free_blocks}/{total_blocks} ({free_ratio:.1%}), "
                              f"completed={len(outputs)}", flush=True)
                    
                    # Emergency recovery: if stuck for too long and resources are low, force finish some sequences
                    if consecutive_empty_steps > 2000 and free_ratio < 0.15:
                        print(f"[EMERGENCY] Attempting emergency recovery: forcing completion of sequences near max_tokens", flush=True)
                        # The scheduler's emergency recovery will handle this in the next step()

                if use_tqdm:
                    throughput = total_generated_tokens / (perf_counter() - start_time + 1e-6)
                    pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})
                    pbar.update(len(output))

                #for seq_id, token_ids in output:
                #    outputs[seq_id] = token_ids
                for seq_id, token_ids, unmask_times in output:
                    outputs[seq_id] = {"token_ids": token_ids, "unmask_times": unmask_times}

        #outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        #results = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs_list]
        outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        results = [
            {
                "text": self._safe_decode(item["token_ids"]),
                "token_ids": self._clean_token_ids(item["token_ids"]),
                "first_unmask_times": item["unmask_times"],
            }
            for item in outputs_list
        ]

        if use_tqdm:
            pbar.close()
        return results