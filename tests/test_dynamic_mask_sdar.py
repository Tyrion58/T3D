import unittest

import torch
from omegaconf import OmegaConf

from train.prepare import (
    _sdar_mask_config,
    collect_clean_data_sdar,
    sample_and_mask_sdar_one,
)


class DynamicMaskSDARTests(unittest.TestCase):
    def setUp(self):
        self.prompt_len = 4
        self.response_len = 8
        self.config = OmegaConf.create({
            "training": {
                "method": "semi-ar",
                "mask_ratios": [1.0, 0.75, 0.5, 0.25],
                "mask_ratio_exponent": 4.0,
                "mask_strategy": "trace",
                "block_size": 4,
                "post_num": 1,
            },
            "model": {"random_ratio": 0.0},
        })
        inputs = torch.arange(1, 1 + 2 * (self.prompt_len + self.response_len)).reshape(2, -1)
        steps = [list(range(self.response_len)) for _ in range(2)]
        self.clean, self.labels, self.positions, self.step_maps, self.keep = collect_clean_data_sdar(
            self.config, inputs, self.prompt_len, 0, steps
        )

    def test_clean_precompute_shapes(self):
        self.assertEqual(tuple(self.clean.shape), (2, 12))
        self.assertEqual(tuple(self.positions.shape), (2, 20))
        self.assertEqual(tuple(self.step_maps.shape), (2, 8))
        self.assertTrue(self.keep.all())

    def test_mask_is_dynamic_and_nonempty(self):
        mask_config = _sdar_mask_config(self.config)
        torch.manual_seed(0)
        counts = set()
        for _ in range(40):
            extended, mask = sample_and_mask_sdar_one(
                self.clean[0], self.step_maps[0], self.prompt_len,
                0, 99, 1000, mask_config, post_num=1,
            )
            self.assertEqual(tuple(extended.shape), (20,))
            self.assertTrue(mask.any())
            counts.add(int(mask.sum()))
        self.assertGreater(len(counts), 1)


if __name__ == "__main__":
    unittest.main()
