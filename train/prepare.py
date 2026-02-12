import torch

def make_basic_block_attention(
    N: int,
    start_pos: int,            # = L0
    block_size: int,           # = b
) -> torch.Tensor:
    B = 1
    L0     = start_pos
    L1     = (N - L0) // 2          # N = L0 + 2·L1 
    assert L0 + 2 * L1 == N, "input length must be L0 + 2*L1"

    # all -inf first
    bias = torch.full((B, 1, N, N), 0)

    rows = torch.arange(L0 + L1, L0 + 2 * L1)              # (L1,)
    rows_token = torch.arange(L0, L0 + L1)              # (L1,)

    # update block by block
    for bi in range((L1 + block_size - 1) // block_size):
        #  [bi*b , min((bi+1)*b, L1))
        left_end   = L0 + min((bi) * block_size, L1)        
        right_start= L0 + L1 + (left_end - L0)

        i_start = bi * block_size
        i_end   = min((bi + 1) * block_size, L1)              # no i_end

        block_rows = rows[i_start:i_end]                    
        bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
        bias[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = 1

        block_rows = rows_token[i_start:i_end]
        left_end   = L0 + min((bi + 1) * block_size, L1)
        bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
    
    if L0 > 0:
        num_blocks_pre = (L0 + block_size - 1) // block_size
        for bi in range(num_blocks_pre):
            # row interval [row_start, row_end)
            row_end   = max(L0 - bi * block_size, 0)
            row_start = max(L0 - (bi + 1) * block_size, 0)
            if row_end > row_start:
                block_rows = torch.arange(row_start, row_end)
                bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1
    
    return bias        # (B,1,N,N)

def process_pad(attn, input_ids, L0, L1, start_pos, pad_id):
    N = L0 + 2 * L1
    device = input_ids.device

    cols = torch.arange(N, device=device)                  # (N,)
    key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)  # (B, N)

    # set -inf
    attn.masked_fill_(key_mask[:, None, None, :], 0)

    # avoid +-inf or none in forward
    A = attn[:, 0]  # (B, N, N)
    bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
    b, r = bad.nonzero(as_tuple=True)
    A[b, r, :] = 0; A[b, r, r] = 1  

    return attn

def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
    """
    Perform a single "round" on one sample b:
    - For each block, take the minimum non -1 value in step_map.
    - Create pmask (positions equal to the block minimum).
    - Create a noise mask for the extended segment (positions >= block minimum).
    - Mark the chosen minimum positions in step_map as -1 for the next round.

    Returns:
    extended_input_ids_b : Tensor with duplicated + masked response segment
    pmask_b              : Boolean mask for tokens selected in this round
    new_step_map_b       : Updated step_map (selected positions set to -1)
    has_any              : Whether any position was selected in this round
    """
    device = input_ids_b.device
    NB = (L1 + block_size - 1) // block_size
    pad_len = NB * block_size - L1

    # Reshape step_map into [NB, block_size], fill last incomplete block with -1
    step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
    step_pad[:L1] = step_map_b
    step_blk = step_pad.view(NB, block_size)                      # [NB, Bk]

    valid = step_blk.ge(0)                                        # Valid positions (not -1)
    big = torch.iinfo(step_blk.dtype).max
    tmp = step_blk.masked_fill(~valid, big)                       # Fill invalid positions with a large value
    min_vals, _ = tmp.min(dim=1, keepdim=True)                    # Current minimum for each block

    # Select positions equal to block minimum (only valid positions)
    pmask_blk = step_blk.eq(min_vals) & valid                     
    if not pmask_blk.any():
        # No positions left to select in this round
        return None, None, step_map_b, False

    # Noise mask for extended segment: mark positions >= block minimum
    ge_mask_blk = step_blk.ge(min_vals) & valid                   # [NB, Bk]

    # Flatten back to length L1 (discard padding)
    pmask_tail = pmask_blk.view(-1)[:L1]                          # [L1]
    ge_mask_tail = ge_mask_blk.view(-1)[:L1]                      # [L1]

    # Construct pmask_b: [0:L0] = False, [L0:] = pmask_tail
    pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
    pmask_b[L0:] = pmask_tail

    # Build extended segment: duplicate response and replace noise positions with mask_id
    tail = input_ids_b[L0:L0+L1].clone()
    tail[ge_mask_tail] = mask_id

    extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
    extended_input_ids_b[:L0+L1] = input_ids_b
    extended_input_ids_b[L0+L1:] = tail

    # Update step_map: mark selected minimum positions as -1 for the next round
    new_step_map_b = step_map_b.clone()
    new_step_map_b[pmask_tail] = -1

    return extended_input_ids_b, pmask_b, new_step_map_b, True


def collapse_k_unique(lst, k: int):
    if k <= 0:
        raise ValueError("k must be > 0")
    uniq = sorted(set(lst))

    mapping = {}
    n = len(uniq)
    for idx, val in enumerate(uniq):
        group = idx // k
        end_idx = min((group + 1) * k - 1, n - 1)
        rep = uniq[end_idx]
        mapping[val] = rep
    return [mapping[x] for x in lst]

def collect_training_data(config, input_ids, start_pos, pad_id, mask_id, vocab_size=None, post_num=None, step_map_list=None):
    B, L = input_ids.shape
    L0    = start_pos
    L1    = L - L0

    # block_size = config.training.block_size

    # lower = config.training.lower_p
    # upper = config.training.upper_p

    if config.training.method == "semi-ar":
        # Get mask_ratios from config (e.g., [1.0, 0.75, 0.5, 0.25] for variable mask ratios)
        # If not specified, defaults to [1.0] (fully masked, backward compatible)
        mask_ratios = config.training.get("mask_ratios", None)
        if mask_ratios is None:
            mask_ratios = [1.0]  # Default: fully masked (backward compatible)
        elif isinstance(mask_ratios, (int, float)):
            mask_ratios = [mask_ratios]  # Convert single value to list
        else:
            # Convert OmegaConf ListConfig to Python list if needed
            try:
                mask_ratios = list(mask_ratios)
            except (TypeError, AttributeError):
                pass  # Already a list or compatible type
        
        # Get random_ratio from config, default to 0.0 (no random tokens, only mask tokens)
        random_ratio = config.model.get("random_ratio", 0.0)
        
        # Get mask_strategy from config: "trace" (use step_map decode order) or "random" (random masking)
        # Default to "trace" for backward compatibility
        mask_strategy = config.training.get("mask_strategy", "trace")
        if mask_strategy not in ["trace", "random"]:
            raise ValueError(f"mask_strategy must be 'trace' or 'random', got '{mask_strategy}'")
        
        # Get block_size from config
        block_size = config.training.block_size
        
        device = input_ids.device
        
        # Calculate probability weights for each mask ratio with exponential scaling
        # Use exponential weights to create extreme differences: weights = mask_ratios ** exponent
        # Higher mask ratio gets exponentially higher probability
        mask_ratio_exponent = config.training.get("mask_ratio_exponent", 4.0)  # Default: 4 (exponential scaling)
        mask_ratios_tensor = torch.tensor(mask_ratios, dtype=torch.float32, device=device)
        weights = mask_ratios_tensor ** mask_ratio_exponent  # Exponential weights for extreme distribution
        # Normalize weights to get probability distribution
        probs = weights / weights.sum()  # Probability distribution: higher mask_ratio has exponentially higher probability
        
        # For each input sample, sample one mask_ratio according to the probability distribution
        # Higher mask_ratio has higher probability of being selected
        selected_mask_ratios_list = []
        
        # Sample mask_ratio for each input sample
        # Use multinomial to sample indices according to probabilities
        sampled_indices = torch.multinomial(probs.unsqueeze(0).expand(B, -1), num_samples=1, replacement=True).squeeze(-1)  # [B]
        selected_mask_ratios_list = [mask_ratios[idx.item()] for idx in sampled_indices]
        
        # Expand step_map if provided (one step_map per sample, matching the sampled mask_ratio)
        # Treat empty list as None
        step_map_expanded = None
        if step_map_list is not None and len(step_map_list) > 0:
            step_map_expanded = []
            for b in range(B):
                sm = step_map_list[b]
                if isinstance(sm, (list, tuple)):
                    step_map_expanded.append(torch.tensor(sm, dtype=torch.long))
                elif isinstance(sm, torch.Tensor):
                    step_map_expanded.append(sm.clone())
                else:
                    step_map_expanded.append(torch.tensor(sm, dtype=torch.long))
        
        # Each input sample generates exactly one training sample
        input_ids_expanded = input_ids  # [B, L] - no expansion, one sample per input
        expanded_B = B  # Batch size remains the same
        selected_mask_ratios = torch.tensor(selected_mask_ratios_list, device=device, dtype=torch.float32)  # [B]
        
        # 2) Construct the noisy tail
        noise_tail = input_ids_expanded[:, L0:].clone()  # [expanded_B, L1]
        
        # Create response mask indicating which tokens should be masked
        response_mask = torch.zeros(expanded_B, L1, dtype=torch.bool, device=device)  # [expanded_B, L1]
        
        # Choose masking strategy based on config and step_map availability
        use_trace_masking = (mask_strategy == "trace") and (step_map_expanded is not None)
        
        if use_trace_masking:
            # Use step_map to determine decode order for masking
            # Lower step_map values = decoded earlier = mask first
            step_map_tensors = []
            for sm in step_map_expanded:
                if isinstance(sm, (list, tuple)):
                    step_map_tensors.append(torch.tensor(sm, dtype=torch.long))
                elif isinstance(sm, torch.Tensor):
                    step_map_tensors.append(sm)
                else:
                    step_map_tensors.append(torch.tensor(sm, dtype=torch.long))
            
            # Stack step_map tensors, handling variable lengths
            # All items in step_map_tensors should already be tensors from the previous loop
            max_len = max(sm.shape[0] for sm in step_map_tensors)
            step_map_padded = []
            for sm in step_map_tensors:
                sm_len = sm.shape[0]
                if sm_len < max_len:
                    # Pad with large values (will be masked anyway)
                    padding = torch.full((max_len - sm_len,), 999999, dtype=sm.dtype)
                    sm = torch.cat([sm, padding], dim=0)
                elif sm_len > max_len:
                    sm = sm[:max_len]
                step_map_padded.append(sm)
            
            step_map = torch.stack(step_map_padded, dim=0).to(device)  # [expanded_B, max_len]
            
            # Trim or pad step_map to match L1
            if step_map.shape[1] > L1:
                step_map = step_map[:, :L1]  # [expanded_B, L1]
            elif step_map.shape[1] < L1:
                # Pad with large values
                pad_len = L1 - step_map.shape[1]
                padding = torch.full((expanded_B, pad_len), 999999, dtype=step_map.dtype, device=device)
                step_map = torch.cat([step_map, padding], dim=1)  # [expanded_B, L1]
            
            # For each sample, mask tokens according to decode order (step_map) per block
            NB = (L1 + block_size - 1) // block_size  # Number of blocks
            for b in range(expanded_B):
                mask_ratio = selected_mask_ratios[b].item()
                step_map_b = step_map[b]  # [L1]
                
                # Process each block
                for bi in range(NB):
                    block_start = bi * block_size
                    block_end = min((bi + 1) * block_size, L1)
                    block_len = block_end - block_start
                    
                    # Get step_map and indices for this block
                    block_step_map = step_map_b[block_start:block_end]  # [block_len]
                    block_indices = torch.arange(block_start, block_end, device=device)  # Global indices
                    
                    # Find valid positions in this block (exclude padding values)
                    valid_mask = block_step_map < 999999
                    valid_block_indices = block_indices[valid_mask]  # Global indices of valid positions
                    valid_block_step_map = block_step_map[valid_mask]  # Step map values for valid positions
                    
                    if len(valid_block_indices) > 0:
                        # Sort by step_map to get decode order (lower = earlier = mask first)
                        sorted_order = torch.argsort(valid_block_step_map)
                        sorted_valid_indices = valid_block_indices[sorted_order]  # Global indices sorted by decode order
                        
                        # Number of tokens to mask in this block based on mask_ratio
                        num_to_mask_in_block = int(len(sorted_valid_indices) * mask_ratio)
                        if num_to_mask_in_block > 0:
                            # Mask the first num_to_mask_in_block tokens in decode order within this block
                            mask_indices_in_block = sorted_valid_indices[:num_to_mask_in_block]
                            response_mask[b, mask_indices_in_block] = True
        else:
            # Random masking: randomly mask tokens per block (same as collect_training_data_sft)
            NB = (L1 + block_size - 1) // block_size  # Number of blocks
            for b in range(expanded_B):
                mask_ratio = selected_mask_ratios[b].item()
                
                # Process each block
                for bi in range(NB):
                    block_start = bi * block_size
                    block_end = min((bi + 1) * block_size, L1)
                    block_len = block_end - block_start
                    
                    # Number of tokens to mask in this block based on mask_ratio
                    num_to_mask_in_block = int(block_len * mask_ratio)
                    if num_to_mask_in_block > 0:
                        # Randomly select positions to mask within this block
                        block_positions = torch.randperm(block_len, device=device)[:num_to_mask_in_block]
                        mask_indices_in_block = block_start + block_positions
                        response_mask[b, mask_indices_in_block] = True
        
        # 1) Create pmask: prefix all False, response mask based on variable mask ratios
        p_mask = torch.cat([
            torch.zeros(expanded_B, L0, dtype=torch.bool, device=device),
            response_mask
        ], dim=1)  # [expanded_B, L]
        
        # Apply masking to noise_tail: tokens that should be masked
        if random_ratio > 0 and vocab_size is not None:
            # Hybrid strategy: within the masked positions, apply random_ratio
            # Some positions get random tokens, others get mask tokens
            masked_positions = response_mask  # [expanded_B, L1]
            
            # Initialize random_mask to all False
            random_mask = torch.zeros(expanded_B, L1, dtype=torch.bool, device=device)
            
            # For each sample, randomly select positions for random tokens within masked positions
            for b in range(expanded_B):
                masked_idx = torch.where(masked_positions[b])[0]  # Positions that should be masked
                if len(masked_idx) > 0:
                    num_random = max(1, int(len(masked_idx) * random_ratio))  # At least 1 if ratio > 0
                    num_random = min(num_random, len(masked_idx))
                    if num_random > 0:
                        # Randomly select which masked positions get random tokens
                        random_idx = masked_idx[torch.randperm(len(masked_idx), device=device)[:num_random]]
                        random_mask[b, random_idx] = True
            
            mask_token_mask = masked_positions & (~random_mask)
            
            # Replace selected positions with random tokens
            if random_mask.any():
                num_random = random_mask.sum().item()
                random_tokens = torch.randint(0, vocab_size, (num_random,), 
                                             device=device, dtype=noise_tail.dtype)
                noise_tail[random_mask] = random_tokens
            
            # Replace remaining masked positions with mask_id
            if mask_token_mask.any():
                noise_tail[mask_token_mask] = mask_id
        else:
            # Simple case: replace all masked positions with mask_id
            noise_tail[response_mask] = mask_id
        
        # 3) Concatenate original sequence with noisy tail
        extended_input_ids = torch.cat([input_ids_expanded, noise_tail], dim=1)  # [expanded_B, L + L1]

    else:
        raise ValueError(f"Method {config.training.method} not supported")
    
    pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask        
    if post_num is not None:
        cum_pad = torch.cumsum(pad_resp.int(), dim=1)
        p_mask &= ~(pad_resp & (cum_pad > post_num))
    
    labels = extended_input_ids[:, :L].clone()

    idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
    valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)      
    tok_idx = valid.long().cumsum(dim=-1) - 1         
    tok_idx = tok_idx.masked_fill(~valid, 1)
    tok_idx_resp = tok_idx[:, start_pos:]  
    tok_idx_ext  = torch.cat([tok_idx, tok_idx_resp], dim=1)

    keep = p_mask.view(p_mask.size(0), -1).any(dim=1)

    extended_input_ids = extended_input_ids[keep]
    p_mask            = p_mask[keep]
    tok_idx_ext       = tok_idx_ext[keep]
    labels            = labels[keep]

    return extended_input_ids, p_mask, tok_idx_ext, labels