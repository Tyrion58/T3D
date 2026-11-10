# Local artifacts

Default home for everything the code generates; ignored by Git.

- `experiments/`: training runs and checkpoints.
- `<experiment.project>/temp_data/`: raw rollouts and decoding statistics.
- `<experiment.project>/results/`: accuracy summaries.

Set `T3D_OUTPUT_DIR` or `experiment.output_dir` to place artifacts on a
different filesystem.
