"""Path-consistency regularisation for block-wise few-step decoding.

The loss weights the cross-entropy of a token by *when* the teacher decoded it
inside its block: tokens revealed at an earlier decoding step matter more,
because their errors propagate to every later step.

Notation: a block of ``block_size`` tokens is decoded in
``B = ceil(block_size / k_steps)`` steps, ``k_steps`` tokens per step, and
``pi_i`` in ``{1, ..., B}`` is the step at which token ``i`` is revealed
(obtained from the intra-block rank of the teacher's ``step_map``).

``stepwise`` (default, used for the released checkpoints)
    For every step ``s`` take the mean cross-entropy over the tokens revealed by
    step ``s``, then average over the ``B`` steps::

        L = 1/B * sum_s  mean_{i : pi_i <= s}  CE_i

    A token revealed at step ``pi_i`` is therefore counted ``B - pi_i + 1``
    times, each time normalised by the number of tokens revealed so far.

``weighted``
    The same counts written as one explicit weight per token with a single
    global normalisation::

        L = 1/N * sum_i  w_i * CE_i,    w_i = (B - pi_i + 1) / B
"""

import torch
import torch.nn.functional as F

PATH_LOSS_TYPES = ("stepwise", "weighted")

# step_map entries >= this value are padding (see train/prepare.py).
_STEP_MAP_PAD = 999999


def _block_step_index(step_map, L1, block_size, k_steps):
    """Per-token decoding step inside its block.

    Args:
        step_map: [B, *] teacher decode order of the response tokens (lower =
            decoded earlier). Negative or ``>= _STEP_MAP_PAD`` entries are padding.

    Returns:
        step_idx:  [B, L1] 0-indexed step at which each token is revealed.
        has_order: [B, L1] True where the token has a real ``step_map`` entry.
        block_has_order: [B, L1] True where the token's block has at least one
            real entry.
    """
    B = step_map.shape[0]
    sm = step_map.long()
    if sm.shape[1] > L1:
        sm = sm[:, :L1]
    elif sm.shape[1] < L1:
        sm = F.pad(sm, (0, L1 - sm.shape[1]), value=-1)

    NB = (L1 + block_size - 1) // block_size
    pad_len = NB * block_size - L1
    if pad_len > 0:
        sm = F.pad(sm, (0, pad_len), value=-1)

    has_order = (sm >= 0) & (sm < _STEP_MAP_PAD)
    # Tokens without an order sort after every ordered token of their block.
    key = torch.where(has_order, sm, torch.full_like(sm, torch.iinfo(sm.dtype).max))
    key = key.view(B, NB, block_size)
    ranks = key.argsort(dim=-1, stable=True).argsort(dim=-1)
    step_idx = (ranks // k_steps).view(B, -1)[:, :L1]

    block_has_order = has_order.view(B, NB, block_size).any(dim=-1, keepdim=True)
    block_has_order = block_has_order.expand(-1, -1, block_size).reshape(B, -1)[:, :L1]
    return step_idx, has_order[:, :L1], block_has_order


def _token_nll(logits_x0, labels, start_pos):
    logits_resp = logits_x0[:, start_pos:, :]
    labels_resp = labels[:, start_pos:]
    log_probs = F.log_softmax(logits_resp, dim=-1)
    nll = -log_probs.gather(-1, labels_resp.unsqueeze(-1)).squeeze(-1)
    return nll, labels_resp


def compute_path_loss_stepwise(logits_x0, labels, step_map, block_size, k_steps, pad_id, start_pos):
    """Step-normalised path loss: average over decoding steps of the mean
    cross-entropy on the tokens revealed by that step."""
    nll, labels_resp = _token_nll(logits_x0, labels, start_pos)
    L1 = labels_resp.shape[1]
    steps_per_block = (block_size + k_steps - 1) // k_steps
    step_idx, has_order, block_has_order = _block_step_index(step_map, L1, block_size, k_steps)

    not_pad = labels_resp != pad_id
    # Tokens beyond the recorded trajectory count as already revealed, as long
    # as their block contains at least one ordered token.
    always_revealed = ~has_order & block_has_order

    step_losses = []
    for s in range(steps_per_block):
        revealed = ((has_order & (step_idx <= s)) | always_revealed) & not_pad
        count = revealed.sum()
        if count > 0:
            step_losses.append((nll * revealed).sum() / count)

    if not step_losses:
        return nll.sum() * 0.0
    return sum(step_losses) / len(step_losses)


def compute_path_loss_weighted(logits_x0, labels, step_map, block_size, k_steps, pad_id, start_pos):
    """Token-weighted path loss, ``w_i = (B - pi_i + 1) / B``, with a single
    global normalisation."""
    nll, labels_resp = _token_nll(logits_x0, labels, start_pos)
    L1 = labels_resp.shape[1]
    steps_per_block = (block_size + k_steps - 1) // k_steps
    step_idx, _, _ = _block_step_index(step_map, L1, block_size, k_steps)

    weights = (steps_per_block - step_idx).float() / steps_per_block

    # Positions the collate function padded with a negative step are skipped.
    sm = step_map.long()
    if sm.shape[1] > L1:
        sm = sm[:, :L1]
    elif sm.shape[1] < L1:
        sm = F.pad(sm, (0, L1 - sm.shape[1]), value=_STEP_MAP_PAD)
    valid = (labels_resp != pad_id) & (sm >= 0)

    return (weights * nll * valid).sum() / valid.sum().clamp(min=1)


def compute_path_loss(logits_x0, labels, step_map, block_size, k_steps, pad_id, start_pos,
                      loss_type="stepwise"):
    """Dispatch on ``training.path_loss_type``."""
    if loss_type == "stepwise":
        return compute_path_loss_stepwise(
            logits_x0, labels, step_map, block_size, k_steps, pad_id, start_pos)
    if loss_type == "weighted":
        return compute_path_loss_weighted(
            logits_x0, labels, step_map, block_size, k_steps, pad_id, start_pos)
    raise ValueError(f"Unknown path_loss_type: {loss_type!r} (choose from {PATH_LOSS_TYPES})")
