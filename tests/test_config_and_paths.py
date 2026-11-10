import os
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from eval import detect_model_base, flatten_overrides
from utils.config import build_outputs_name, get_eval_config
from utils.paths import REPO_ROOT, artifact_root, dataset_file, training_root


class ConfigAndPathTests(unittest.TestCase):
    def test_output_name_contains_decoding_settings(self):
        config = OmegaConf.create({
            "rollout": {"steps": 64, "denoising_steps_per_block": 2, "block_size": 8}
        })
        self.assertEqual(
            build_outputs_name(config, "org/model", "MATH500"),
            "eval-model-s64-dspb2-bs8-MATH500",
        )

    def test_eval_fallbacks(self):
        config = OmegaConf.create({"dataset": {"data_type": "code"}, "rollout": {"steps": 32}})
        self.assertEqual(get_eval_config(config, "data_type"), "code")
        self.assertEqual(get_eval_config(config, "steps"), 32)

    def test_default_and_overridden_roots(self):
        self.assertEqual(artifact_root(), REPO_ROOT / "artifacts")
        with patch.dict(os.environ, {"T3D_OUTPUT_DIR": "/tmp/t3d-output"}):
            self.assertEqual(artifact_root(), Path("/tmp/t3d-output"))
        config = OmegaConf.create({"experiment": {"save_dir": "custom-runs"}})
        self.assertEqual(training_root(config), REPO_ROOT / "custom-runs")
        self.assertEqual(dataset_file("MATH500"), REPO_ROOT / "data" / "MATH500.json")

    def test_eval_launcher_helpers(self):
        self.assertEqual(detect_model_base(OmegaConf.create({})), "sdar")
        self.assertEqual(detect_model_base(OmegaConf.create({"model_base": "SDAR"})), "sdar")
        overrides = flatten_overrides({"evaluation": {"seed": 42}, "rollout": {"block_size": 8}})
        self.assertIn("evaluation.seed=42", overrides)
        self.assertIn("rollout.block_size=8", overrides)


if __name__ == "__main__":
    unittest.main()
