import math
import unittest

import torch

from train.path_loss import (
    compute_path_loss,
    compute_path_loss_stepwise,
    compute_path_loss_weighted,
)

PAD = 0


def make_inputs(nll_per_token, step_map, start_pos=2, vocab=5, pad_tail=0):
    """Logits whose cross-entropy on the label equals ``nll_per_token``."""
    L1 = len(nll_per_token)
    labels = torch.full((1, start_pos + L1), 3, dtype=torch.long)
    if pad_tail:
        labels[0, -pad_tail:] = PAD
    logits = torch.zeros(1, start_pos + L1, vocab, dtype=torch.float64)
    for i, nll in enumerate(nll_per_token):
        # softmax over {label, one other class}: p(label) = exp(-nll)
        p = math.exp(-nll)
        logits[0, start_pos + i] = -1e9
        logits[0, start_pos + i, labels[0, start_pos + i]] = math.log(p)
        logits[0, start_pos + i, 4] = math.log(1 - p)
    return logits, labels, torch.tensor([step_map], dtype=torch.long)


class PathLossTests(unittest.TestCase):
    def setUp(self):
        # One block of 4 tokens decoded in 2 steps (k=2). Teacher order:
        # positions 1 and 3 first (step 1), then positions 0 and 2 (step 2).
        self.nll = [0.4, 0.1, 0.8, 0.3]
        self.args = make_inputs(self.nll, step_map=[7, 5, 8, 6])
        self.kw = dict(block_size=4, k_steps=2, pad_id=PAD, start_pos=2)

    def test_stepwise_matches_definition(self):
        early = (0.1 + 0.3) / 2            # tokens revealed by step 1
        everything = sum(self.nll) / 4     # tokens revealed by step 2
        loss = compute_path_loss_stepwise(*self.args, **self.kw)
        self.assertAlmostEqual(loss.item(), (early + everything) / 2, places=9)

    def test_weighted_matches_definition(self):
        early, late = 0.1 + 0.3, 0.4 + 0.8  # weights 2/2 and 1/2
        loss = compute_path_loss_weighted(*self.args, **self.kw)
        self.assertAlmostEqual(loss.item(), (1.0 * early + 0.5 * late) / 4, places=6)

    def test_early_tokens_weigh_more(self):
        logits, labels, step_map = self.args
        logits.requires_grad_(True)
        # d nll / d logit = -(1 - p); dividing it out leaves the token weight
        norm = torch.tensor([1 - math.exp(-x) for x in self.nll], dtype=torch.float64)
        for loss_type in ("stepwise", "weighted"):
            logits.grad = None
            compute_path_loss(logits, labels, step_map, **self.kw, loss_type=loss_type).backward()
            weight = logits.grad[0, 2:, 3].abs() / norm
            self.assertGreater(weight[1].item(), weight[0].item(), msg=loss_type)
            self.assertGreater(weight[3].item(), weight[2].item(), msg=loss_type)

    def test_padding_is_ignored(self):
        logits, labels, step_map = make_inputs([0.4, 0.1, 0.8, 0.3, 2.0, 2.0, 2.0, 2.0],
                                               step_map=[7, 5, 8, 6], pad_tail=4)
        # second block is all padding and has no recorded order
        self.assertAlmostEqual(
            compute_path_loss_stepwise(logits, labels, step_map, **self.kw).item(),
            compute_path_loss_stepwise(*self.args, **self.kw).item(), places=9)
        self.assertAlmostEqual(
            compute_path_loss_weighted(logits, labels, step_map, **self.kw).item(),
            compute_path_loss_weighted(*self.args, **self.kw).item(), places=9)

    def test_no_valid_token_gives_zero(self):
        logits, labels, step_map = make_inputs([1.0, 1.0, 1.0, 1.0], step_map=[1, 2, 3, 4], pad_tail=4)
        for loss_type in ("stepwise", "weighted"):
            loss = compute_path_loss(logits, labels, step_map, **self.kw, loss_type=loss_type)
            self.assertEqual(loss.item(), 0.0)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            compute_path_loss(*self.args, **self.kw, loss_type="nope")


if __name__ == "__main__":
    unittest.main()
