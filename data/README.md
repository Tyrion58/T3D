# Data

Dataset files are downloaded or generated locally and are ignored by Git.

| Script | Produces |
| --- | --- |
| `download_data.py --dataset <name>` | `MATH_train`, `PrimeIntellect` (training prompts); `MATH500`, `GSM8K`, `MBPP`, `HumanEval` (evaluation) |
| `rename_key.py --model_name <model> --dataset <name>` | `<model>-<name>_renamed.json` training file from a rollout |

`download_data.py` writes into the current directory, so run it from `data/`.

Training files follow the schema `question`, `answer`, `step_map`
(the teacher's per-token decode step) and `response_length`.
