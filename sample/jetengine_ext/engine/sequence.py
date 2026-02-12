from copy import copy
from enum import Enum, auto
from itertools import count
import random
from jetengine_ext.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()      # Has a prompt part to prefill
    PREFILLING = auto()   # Is currently in a prefill model run
    DENOISING = auto()    # Is ready for or in a denoise model run
    SAVING = auto()       # Is ready for or in a save model run
    FINISHED = auto()
    
class RunType(Enum):
    PREFILL = auto()
    DENOISE = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, prompt_token_ids: list[int], mask_token_id: int, sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.block_length = sampling_params.block_length
        self.prompt_token_ids = prompt_token_ids
        prompt_len = len(self.prompt_token_ids)
        
        self.num_prefill_tokens = (prompt_len // self.block_length) * self.block_length
        prefill_part = self.prompt_token_ids[:self.num_prefill_tokens]
        
        first_denoise_part = self.prompt_token_ids[self.num_prefill_tokens:]
        
        self.token_ids = prefill_part
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = prompt_len # Keep track of the original full prompt length
        
        # Initialize first block with optional random tokens (will be applied after vocab_size is set)
        mask_fill_length = self.block_length - len(first_denoise_part)
        self.intermediate_block_tokens = first_denoise_part + [mask_token_id] * mask_fill_length
        self._first_block_base_tokens = first_denoise_part  # Store for later random initialization
        self._first_block_mask_length = mask_fill_length
        self.random_init_positions = set()  # Track positions that were randomly initialized (to treat as "mask-like")
        self.num_to_transfer = 0
        self.current_denoising_step = 0


        self.first_unmask_steps: list[int] = []
        self.block_first_unmask_steps: list[int] | None = [0] * len(self.intermediate_block_tokens)
        self.global_denoising_step = 0
        
        # initial status based on whether prefill is needed.
        if self.num_prefill_tokens > 0:
            self.status = SequenceStatus.WAITING
        else:
            self.status = SequenceStatus.DENOISING

        # Block Diffusion parameters
        self.temperature = sampling_params.temperature
        self.stop_words = sampling_params.stop_words if sampling_params.stop_words is not None else []
        self.top_k = sampling_params.topk
        self.top_p = sampling_params.topp
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.denoising_steps = sampling_params.denoising_steps
        self.remasking_strategy = sampling_params.remasking_strategy
        self.dynamic_threshold = sampling_params.dynamic_threshold
        self.eb_threshold = sampling_params.eb_threshold
        self.random_init_ratio = sampling_params.random_init_ratio
        self.mask_token_id = mask_token_id
        self.num_transfer_tokens_per_step = self._get_num_transfer_tokens()
        self.vocab_size = None  # Will be set by LLMEngine
        self.eos_token_id = None  # Will be set by LLMEngine

        # State for KV Caching
        self.num_cached_tokens = 0
        self.block_table = []
        
    def _apply_random_init_to_first_block(self):
        """Apply random initialization to the first block if vocab_size is set."""
        if hasattr(self, '_first_block_base_tokens') and self.random_init_ratio > 0.0:
            if self.vocab_size is not None and self.vocab_size > 0:
                self.intermediate_block_tokens, random_positions = self._init_block_with_random(
                    self._first_block_base_tokens, 
                    self._first_block_mask_length, 
                    self.mask_token_id
                )
                # Store random positions relative to block start
                self.random_init_positions = random_positions
                # Clear the stored values after use
                delattr(self, '_first_block_base_tokens')
                delattr(self, '_first_block_mask_length')

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    def _get_num_transfer_tokens(self):
        base = self.block_length // self.denoising_steps
        remainder = self.block_length % self.denoising_steps
        num_tokens = [base] * self.denoising_steps
        for i in range(remainder):
            num_tokens[i] += 1
        return num_tokens

    def _init_block_with_random(self, base_tokens: list[int], mask_fill_length: int, mask_token_id: int) -> tuple[list[int], set[int]]:
        """
        Initialize a block with base_tokens + mask tokens, optionally replacing some masks with random tokens.
        
        Args:
            base_tokens: Initial tokens (e.g., from prompt)
            mask_fill_length: Number of mask tokens to add
            mask_token_id: The mask token ID
        
        Returns:
            Tuple of (block tokens, set of random initialized positions relative to block start)
        """
        block = base_tokens + [mask_token_id] * mask_fill_length
        random_positions_set = set()
        
        # Apply random initialization if enabled
        if hasattr(self, 'random_init_ratio') and self.random_init_ratio > 0.0:
            vocab_size = getattr(self, 'vocab_size', None)
            if vocab_size is not None and vocab_size > 0:
                # Only randomize mask positions
                mask_positions = [i for i in range(len(base_tokens), len(block)) if block[i] == mask_token_id]
                num_random = int(len(mask_positions) * self.random_init_ratio)
                
                if num_random > 0 and mask_positions:
                    random_positions = random.sample(mask_positions, min(num_random, len(mask_positions)))
                    # Avoid special tokens (mask_token_id, eos_token_id, pad_token_id)
                    special_tokens = {mask_token_id}
                    if hasattr(self, 'eos_token_id') and self.eos_token_id is not None:
                        special_tokens.add(self.eos_token_id)
                    
                    for pos in random_positions:
                        # Sample a random token, retry if it's a special token
                        max_retries = 10
                        for _ in range(max_retries):
                            random_token = random.randint(0, vocab_size - 1)
                            if random_token not in special_tokens:
                                block[pos] = random_token
                                random_positions_set.add(pos)
                                break
        
        return block, random_positions_set

    def start_new_block(self):
        self.current_denoising_step = 0
        # Use random initialization if enabled
        self.intermediate_block_tokens, random_positions = self._init_block_with_random(
            [], self.block_length, self.mask_token_id
        )
        # Store random positions relative to current block (always starts at 0 for new blocks)
        self.random_init_positions = random_positions
        self.status = SequenceStatus.DENOISING

    '''
    def commit_block(self, block_tokens: list[int]):
        # Trim block if it exceeds max_tokens or contains EOS
        final_block = []
        for token_id in block_tokens:
            if not self.ignore_eos and (token_id == self.eos_token_id or token_id in self.stop_words):
                final_block.append(token_id)
                self.status = SequenceStatus.FINISHED
                break
            if self.num_completion_tokens + len(final_block) >= self.max_tokens:
                self.status = SequenceStatus.FINISHED
                break
            final_block.append(token_id)

        self.token_ids.extend(final_block)
        self.num_tokens = len(self.token_ids)
        self.intermediate_block_tokens = []

        if self.num_tokens >= self.num_prompt_tokens + self.max_tokens:
             self.status = SequenceStatus.FINISHED'''
    





    def commit_block(self, block_tokens: list[int], early_termination_threshold: float = 0.95):
        # 1) take token one by one, stop when EOS / reach max_tokens
        # early_termination_threshold: finish early if completion >= threshold * max_tokens (to free resources)
        final_block = []
        k = 0
        for token_id in block_tokens:
            if not self.ignore_eos and (token_id == self.eos_token_id or token_id in self.stop_words):
                final_block.append(token_id)
                k += 1
                self.status = SequenceStatus.FINISHED
                break
            if self.num_completion_tokens + k >= self.max_tokens:
                self.status = SequenceStatus.FINISHED
                break
            # Early termination: if we're close to max_tokens and resources are tight, finish early
            # This helps free up KV cache when resources are constrained
            if self.num_completion_tokens + k >= int(self.max_tokens * early_termination_threshold):
                # Only early terminate if we're in the last block (to avoid cutting off mid-thought)
                # Check if we're past the minimum reasonable length
                if self.num_completion_tokens + k >= max(32, int(self.max_tokens * 0.8)):
                    final_block.append(token_id)
                    k += 1
                    self.status = SequenceStatus.FINISHED
                    break
            final_block.append(token_id)
            k += 1

        # 2) self.token_ids
        before_ntok = self.num_tokens                     # pre length
        self.token_ids.extend(final_block)
        self.num_tokens = len(self.token_ids)
        self.intermediate_block_tokens = []

        # 3) merge first unmask step into list
        # only completion 
        if self.block_first_unmask_steps is not None:
            prompt_gap = max(0, self.num_prompt_tokens - before_ntok)
            # completion start index
            start = min(prompt_gap, k)
            if start < k:
                self.first_unmask_steps.extend(self.block_first_unmask_steps[start:k])
            self.block_first_unmask_steps = None

        if self.num_tokens >= self.num_prompt_tokens + self.max_tokens:
            self.status = SequenceStatus.FINISHED


    




    def get_len_for_next_step(self):
        return self.num_tokens + self.block_length

    def num_new_blocks_needed(self, block_size: int) -> int:
        if not self.block_table:
            return (self.num_tokens + self.block_length + block_size - 1) // block_size

        last_block_capacity = block_size - (self.num_tokens % block_size)
        if last_block_capacity == block_size: # Current tokens perfectly fill blocks
            last_block_capacity = 0
        
        remaining_tokens_to_add = self.block_length - last_block_capacity
        if remaining_tokens_to_add <= 0:
            return 0
            
        return (remaining_tokens_to_add + block_size - 1) // block_size

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    '''
    def __getstate__(self):
        # Simplified for multiprocessing; customize as needed
        return (self.seq_id, self.status, self.token_ids, self.num_tokens, self.num_prompt_tokens, 
                self.num_cached_tokens, self.block_table, self.intermediate_block_tokens, self.current_denoising_step)

    def __setstate__(self, state):
        (self.seq_id, self.status, self.token_ids, self.num_tokens, self.num_prompt_tokens, 
         self.num_cached_tokens, self.block_table, self.intermediate_block_tokens, self.current_denoising_step) = state'''
    
    def __getstate__(self):
        # Simplified for multiprocessing; customize as needed
        return (self.seq_id, self.status, self.token_ids, self.num_tokens, self.num_prompt_tokens, 
                self.num_cached_tokens, self.block_table, self.intermediate_block_tokens, self.current_denoising_step,
                self.first_unmask_steps, self.block_first_unmask_steps, self.global_denoising_step,
                self.random_init_positions)

    def __setstate__(self, state):
        (self.seq_id, self.status, self.token_ids, self.num_tokens, self.num_prompt_tokens, 
         self.num_cached_tokens, self.block_table, self.intermediate_block_tokens, self.current_denoising_step,
         self.first_unmask_steps, self.block_first_unmask_steps, self.global_denoising_step,
         self.random_init_positions) = state
    
    

