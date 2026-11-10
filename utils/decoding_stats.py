"""Throughput, latency and denoising-step statistics for a generation run."""

IM_END_TOKEN = "<|im_end|>"


def truncate_at_im_end(text):
    """Cut a response right after its first ``<|im_end|>`` (the token is kept)."""
    idx = text.find(IM_END_TOKEN)
    return text if idx == -1 else text[: idx + len(IM_END_TOKEN)]


def decoding_statistics(step_maps, token_lengths, block_size, total_time=None):
    """Throughput, latency and denoising-step statistics of one generation run.

    Args:
        step_maps: per response, the denoising step at which each token was
            first unmasked (as returned by the engine), or None.
        token_lengths: per response, number of generated tokens (padding
            removed, truncated at ``<|im_end|>``).
        block_size: decoding block size.
        total_time: wall-clock seconds spent in generation, or None.
    """
    steps_per_response, steps_per_block = [], []
    for steps, n_tokens in zip(step_maps, token_lengths):
        if not isinstance(steps, (list, tuple)):
            steps_per_response.append(0)
            continue
        steps = list(steps[:n_tokens])
        valid = [s for s in steps if isinstance(s, (int, float))]
        steps_per_response.append(max(valid) if valid else 0)
        prev_max = 0
        for b in range(0, len(steps), block_size):
            chunk = [s for s in steps[b:b + block_size] if isinstance(s, (int, float))]
            if not chunk:
                continue
            block_max = max(chunk)
            if block_max - prev_max > 0:
                steps_per_block.append(block_max - prev_max)
            prev_max = block_max

    n = len(token_lengths)
    total_tokens = sum(token_lengths)
    stats = {
        "num_responses": n,
        "total_tokens": total_tokens,
        "avg_length": total_tokens / n if n else 0.0,
        "avg_steps": sum(steps_per_response) / n if n else 0.0,
        "avg_steps_per_block": (sum(steps_per_block) / len(steps_per_block)) if steps_per_block else 0.0,
        "total_time_s": total_time,
        "throughput_tps": None,
        "avg_latency_s": None,
    }
    stats["avg_tokens_per_step"] = (total_tokens / sum(steps_per_response)) if sum(steps_per_response) else 0.0
    if total_time and total_time > 0:
        stats["throughput_tps"] = total_tokens / total_time
        stats["avg_latency_s"] = total_time / n if n else None
    return stats
