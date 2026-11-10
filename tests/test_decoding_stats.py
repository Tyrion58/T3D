import unittest

from utils.decoding_stats import decoding_statistics, truncate_at_im_end


class DecodingStatsTests(unittest.TestCase):
    def test_truncate_at_im_end(self):
        self.assertEqual(truncate_at_im_end("answer<|im_end|>garbage"), "answer<|im_end|>")
        self.assertEqual(truncate_at_im_end("no end token"), "no end token")

    def test_static_decoding(self):
        # two blocks of 4 tokens, 2 tokens per step -> 4 steps for 8 tokens
        steps = [1, 1, 2, 2, 3, 3, 4, 4]
        stats = decoding_statistics([steps], [8], block_size=4, total_time=2.0)
        self.assertEqual(stats["avg_steps"], 4)
        self.assertEqual(stats["avg_steps_per_block"], 2)
        self.assertEqual(stats["avg_tokens_per_step"], 2)
        self.assertEqual(stats["throughput_tps"], 4.0)
        self.assertEqual(stats["avg_latency_s"], 2.0)

    def test_steps_are_cut_at_response_length(self):
        # only the first 4 tokens belong to the response
        stats = decoding_statistics([[1, 2, 3, 4, 5, 6, 7, 8]], [4], block_size=4)
        self.assertEqual(stats["avg_steps"], 4)
        self.assertIsNone(stats["throughput_tps"])

    def test_missing_step_map(self):
        stats = decoding_statistics(["", [1, 1, 1, 1]], [0, 4], block_size=4, total_time=1.0)
        self.assertEqual(stats["num_responses"], 2)
        self.assertEqual(stats["avg_steps"], 0.5)
        self.assertEqual(stats["avg_latency_s"], 0.5)


if __name__ == "__main__":
    unittest.main()
