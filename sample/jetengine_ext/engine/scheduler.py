from collections import deque
import torch
from torch.nn import functional as F
import numpy as np

from jetengine_ext.config import Config
from jetengine_ext.engine.sequence import Sequence, SequenceStatus, RunType
from jetengine_ext.engine.block_manager import BlockManager
from jetengine_ext.layers.sampler import sample_with_temperature_topk_topp
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, TopK, Sample


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []
        self.enable_emergency_recovery = getattr(config, 'enable_emergency_recovery', True)
        self.early_termination_threshold = getattr(config, 'early_termination_threshold', 0.95)
        self.sample_pipe = LogitsPipe([
                                Temperature(),      # Scale logits by temperature
                                TopK(),             # Apply top-k filtering
                                Softmax(),          # Convert logits to probabilities
                                TopP(),             # Apply top-p filtering
                            ])
        self.sample_pipe_topk0 = LogitsPipe([
                        Temperature(),      # Scale logits by temperature
                        Softmax(),          # Convert logits to probabilities
                        TopP(),             # Apply top-p filtering
                        ])

    def add(self, seq: Sequence):
        self.running.append(seq)

    def is_finished(self):
        return not self.running

    def _cleanup_finished_sequences(self):
        """Actively clean up finished sequences to free KV cache immediately."""
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        if finished_seqs:
            for seq in finished_seqs:
                self.block_manager.deallocate(seq)
            self.running = [seq for seq in self.running if not seq.is_finished]
            return len(finished_seqs)
        return 0

    def schedule(self) -> tuple[list[Sequence], RunType] | tuple[None, None]:
        # CRITICAL: Always cleanup finished sequences first to free KV cache immediately
        # This ensures we don't hold onto resources for sequences that are already done
        finished_count = self._cleanup_finished_sequences()
        if finished_count > 0:
            print(f"[Cleanup] Freed {finished_count} finished sequences, free_blocks={len(self.block_manager.free_block_ids)}", flush=True)
        
        # Emergency resource recovery: if KV cache is critically low, force finish sequences
        free_blocks = len(self.block_manager.free_block_ids)
        total_blocks = len(self.block_manager.blocks)
        free_ratio = free_blocks / max(total_blocks, 1)
        
        # If free blocks < 5% and we have sequences that are close to completion, force finish them
        if self.enable_emergency_recovery and free_ratio < 0.05 and self.running:
            # Find sequences that are close to max_tokens (>= 80% of max_tokens)
            force_finished = []
            for seq in self.running:
                if seq.status in (SequenceStatus.DENOISING, SequenceStatus.SAVING):
                    completion_ratio = seq.num_completion_tokens / max(seq.max_tokens, 1)
                    if completion_ratio >= 0.8:
                        # Force finish this sequence to free up resources
                        seq.status = SequenceStatus.FINISHED
                        force_finished.append((seq.seq_id, completion_ratio))
            
            if force_finished:
                print(f"[Emergency] Force finishing {len(force_finished)} sequences to free KV cache: {force_finished}", flush=True)
                # Immediately cleanup the force-finished sequences
                self._cleanup_finished_sequences()
                # Recalculate free_ratio after cleanup
                free_blocks = len(self.block_manager.free_block_ids)
                free_ratio = free_blocks / max(total_blocks, 1)
        
        # 1. Schedule new sequences for prefill
        prefill_candidates = [s for s in self.running if s.status == SequenceStatus.WAITING]
        if prefill_candidates:
            prefill_batch = []
            # Simple batching: take as many as fit
            for seq in prefill_candidates:
                # num_tokens for a waiting seq is its prefill length
                if len(prefill_batch) < self.max_num_seqs and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.PREFILLING
                    prefill_batch.append(seq)
            if prefill_batch:
                return prefill_batch, RunType.PREFILL   
        # 2. If no prefilling, create a DENOISE batch.
        denoise_candidates = [s for s in self.running if s.status == SequenceStatus.DENOISING or s.status == SequenceStatus.SAVING]
        if denoise_candidates:
            denoise_batch = []
            # Sort candidates with multi-criteria:
            # 1. Priority: sequences needing fewer blocks (0 blocks first)
            # 2. Secondary: sequences closer to completion (to free resources faster)
            denoise_candidates_sorted = sorted(
                denoise_candidates, 
                key=lambda s: (
                    s.num_new_blocks_needed(self.block_manager.block_size),  # Fewer blocks first
                    -s.num_completion_tokens / max(s.max_tokens, 1)  # More complete first (negative for descending)
                )
            )
            for seq in denoise_candidates_sorted:
                num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
                if len(denoise_batch) < self.max_num_seqs and self.block_manager.can_append_blocks(num_new_blocks):
                    self.block_manager.append_blocks(seq, num_new_blocks)
                    denoise_batch.append(seq)
            if denoise_batch:
                return denoise_batch, RunType.DENOISE

        return None, None     

    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType):
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        
        elif run_type == RunType.DENOISE:
            start_idx = 0
            if self.consistent_sampling_params:
                if seqs[0].top_k > 0:
                    probs = self.sample_pipe(logits, temperature=seqs[0].temperature, top_k=seqs[0].top_k, top_p=seqs[0].top_p) 
                else:
                    probs = self.sample_pipe_topk0(logits, temperature=seqs[0].temperature, top_p=seqs[0].top_p)
            for seq in seqs:
                # Extract the part of the tensors relevant to this sequence
                if seq.status == SequenceStatus.DENOISING:
                    block_len = seq.block_length
                    if not self.consistent_sampling_params:
                        if seq.top_k > 0:
                            probs = self.sample_pipe(logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_k=seq.top_k, top_p=seq.top_p) 
                        else:
                            probs = self.sample_pipe_topk0(logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_p=seq.top_p)
                        seq_x0 = torch.multinomial(probs, num_samples=1).squeeze(-1) 
                        seq_x0_p = torch.gather(probs, -1, seq_x0.unsqueeze(-1)).squeeze(-1)    
                    else:
                        seq_x0 = torch.multinomial(probs[start_idx : start_idx + block_len], num_samples=1).squeeze(-1) 
                        seq_x0_p = torch.gather(probs[start_idx : start_idx + block_len], -1, seq_x0.unsqueeze(-1)).squeeze(-1)    
                    
                    current_block_tensor = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                    # mask_index includes both mask tokens and randomly initialized positions
                    mask_index = (current_block_tensor == self.mask_token_id)
                    # Also include randomly initialized positions as "mask-like" (can be updated)
                    if hasattr(seq, 'random_init_positions') and seq.random_init_positions:
                        random_init_mask = torch.zeros(block_len, dtype=torch.bool, device=logits.device)
                        for pos in seq.random_init_positions:
                            if 0 <= pos < block_len:
                                random_init_mask[pos] = True
                        mask_index = mask_index | random_init_mask
                    num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                    
                    transfer_index = torch.zeros_like(seq_x0, dtype=torch.bool)
                    
                    if seq.remasking_strategy == 'sequential':
                        if mask_index.any():
                            first_mask_pos = mask_index.nonzero(as_tuple=True)[0].min().item()
                            end_pos = min(first_mask_pos + num_to_transfer, block_len)
                            transfer_index[first_mask_pos:end_pos] = True
                    
                    elif 'low_confidence_static' in seq.remasking_strategy:
                        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                        # For dynamic, add threshold logic here if desired
                        _, top_indices = torch.topk(confidence, num_to_transfer)
                        transfer_index[top_indices] = True
                    
                    elif 'low_confidence_dynamic' in seq.remasking_strategy:
                        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                        transfer_index = torch.where(confidence > seq.dynamic_threshold, True, False)
                        if sum(transfer_index) < num_to_transfer:
                            _, top_indices = torch.topk(confidence, num_to_transfer)
                            transfer_index[top_indices] = True
                        num_to_transfer = transfer_index.sum().item() if transfer_index.sum().item() > 0 else num_to_transfer
                    elif 'entropy_bounded' in seq.remasking_strategy:
                        block_probs = probs[start_idx : start_idx + block_len]
                        P = block_probs[mask_index]
                        eps = 1e-12
                        entropies = -(P.clamp_min(eps) * (P.clamp_min(eps)).log()).sum(dim=-1)
                        ent_sorted, order = torch.sort(entropies, dim=0, descending=False)
                        cumsum = torch.cumsum(ent_sorted, dim=0)
                        k = torch.searchsorted(cumsum, torch.tensor(seq.eb_threshold, device=P.device), right=False).item()
                        if k == 0:
                            k = 1
                        # print(k)
                        selected_token_indices = mask_index.nonzero(as_tuple=True)[0][order[:k]]
                        # print(selected_token_indices)
                        transfer_index[selected_token_indices] = True
                        num_to_transfer = k

                    # update
                    new_block_list = current_block_tensor.tolist()
                    accepted_tokens = seq_x0[transfer_index].tolist()
                    original_indices = transfer_index.nonzero(as_tuple=True)[0].tolist()





                    # newly added
                    if seq.block_first_unmask_steps is None or len(seq.block_first_unmask_steps) != block_len:
                        seq.block_first_unmask_steps = [0] * block_len
                    first_time_global = seq.global_denoising_step + 1
                    for idx in original_indices:
                        if seq.block_first_unmask_steps[idx] == 0:
                            seq.block_first_unmask_steps[idx] = first_time_global



                    

                    for idx, token in zip(original_indices, accepted_tokens):
                        new_block_list[idx] = token
                        # Remove from random_init_positions once it's been updated
                        if hasattr(seq, 'random_init_positions') and idx in seq.random_init_positions:
                            seq.random_init_positions.remove(idx)
                    seq.intermediate_block_tokens = new_block_list
                    
                    seq.current_denoising_step += 1
                    seq.global_denoising_step += 1
                    
                    # Check if block is fully denoised
                    is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or \
                                        (seq.current_denoising_step >= seq.denoising_steps)

                    if is_fully_denoised:
                        # Block is done, commit it and check if generation is finished
                        seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = num_to_transfer
                    
                elif seq.status == SequenceStatus.SAVING:
                    # If saving, commit the block and start a new one
                    # Use early termination if KV cache is low (free < 10%)
                    free_blocks = len(self.block_manager.free_block_ids)
                    total_blocks = len(self.block_manager.blocks)
                    free_ratio = free_blocks / max(total_blocks, 1)
                    # More aggressive early termination when resources are tight
                    if self.enable_emergency_recovery:
                        early_threshold = 0.90 if free_ratio < 0.1 else (0.95 if free_ratio < 0.2 else self.early_termination_threshold)
                    else:
                        early_threshold = 1.0  # No early termination
                    seq.commit_block(seq.intermediate_block_tokens, early_termination_threshold=early_threshold)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()

                start_idx += seq.block_length
                
        # Cleanup finished sequences after processing (but also done at start of schedule())
        # This is a safety net in case sequences finish during postprocess
        self._cleanup_finished_sequences()