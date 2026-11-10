<h1 align="center"> Few-Step Diffusion Language Models via Trajectory Self-Distillation
</h1>

<div align="center" class="project-authors"
  data-authors="tunyu-zhang,xinxi-zhang,ligong-han,haizhou-shi,xiaoxiao-he,zhuowei-li,hao-wang2,kai-xu,akash-srivastava,hao-wang,vladimir-pavlovic,dimitris-metaxas"
  data-stars="tunyu-zhang,xinxi-zhang">
  <a href="https://tyrion58.github.io/" target="_blank"><strong style="text-decoration: underline;">Tunyu&nbsp;Zhang</strong></a><sup>1*</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://xinxi-zhang.github.io/WEB_XINXI/" target="_blank"><strong style="text-decoration: underline;">Xinxi&nbsp;Zhang</strong></a><sup>1*</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=n2v43R4AAAAJ&hl=en" target="_blank">Ligong&nbsp;Han</a><sup>2,3</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=-OF9RG8AAAAJ&hl=en" target="_blank">Haizhou&nbsp;Shi</a><sup>1</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=WmK6IOEAAAAJ&hl=en" target="_blank">Xiaoxiao&nbsp;He</a><sup>1</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=51OJEPcAAAAJ&hl=en" target="_blank">Zhuowei&nbsp;Li</a><sup>1</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=A3WtYhAAAAAJ&hl=en" target="_blank">Hao&nbsp;Wang</a><sup>2,3</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=kf3C60wAAAAJ&hl=en" target="_blank">Kai&nbsp;Xu</a><sup>2,3</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=2h6SZeEAAAAJ&hl=en" target="_blank">Akash&nbsp;Srivastava</a><sup>2,3</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=pTTEiHUAAAAJ&hl=en" target="_blank">Chengzhi&nbsp;Mao</a><sup>1</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=NrOA9QoAAAAJ&hl=en" target="_blank">Hao&nbsp;Wang</a><sup>1</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=8MQT8skAAAAJ" target="_blank">Vladimir&nbsp;Pavlovic</a><sup>1</sup>
  &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=a7VNhCIAAAAJ&hl=en" target="_blank">Dimitris&nbsp;N.&nbsp;Metaxas</a><sup>1</sup>

  <br>

  <sup>1</sup> Rutgers University &emsp; <sup>2</sup> Red Hat AI Innovation &emsp; <sup>3</sup> MIT-IBM Watson AI Lab &emsp;
  <br>
  <sup>*</sup> Equal contribution &emsp; <br>
</div>

<h3 align="center">[<a href="https://arxiv.org/abs/2602.12262">arXiv</a>]</h3>
<br>

![Overview](./img/T3D.png)
<b>Overview</b>: We proposed **T3D** (Trajectory Self-Distillation via Direct Discriminative Optimization), a trajectory self-distillation framework designed to enable **high-quality few-step inference** for Diffusion Large Language Models (DLLMs).

While DLLMs provide strong potential for parallel token generation, they often suffer from severe quality degradation when the number of denoising steps is aggressively reduced.
T3D addresses this challenge by distilling **generative trajectories** from a full-step teacher model, significantly narrowing the performance gap between few-step and full-step decoding.

## Key Ideas
- **Label-Free Training:** Learns entirely from self-generated trajectories without external labels.

- **Trajectory Self-Distillation:** Distills on-policy teacher rollouts to reduce train–test mismatch.

- **Direct Discriminative Optimization:** Uses reverse-KL-style optimization to focus on high-probability teacher modes.

## Installation

### 1. Environment Setup

```bash
conda create -n dllm python=3.10
conda activate dllm
```

### 2. Install Dependencies

```bash
pip install torch==2.6.0
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
conda install -c nvidia cuda
```

### 3. (Optional) Paths

All commands are run from the repository root. Defaults can be redirected with environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `T3D_DATA_DIR` | datasets and training files | `<repo>/data` |
| `T3D_OUTPUT_DIR` | rollouts, results, checkpoints | `<repo>/artifacts` |
| `T3D_CACHE_ROOT` | Triton / Torch extension cache (use a local disk) | `~/.cache/t3d` |

## Data Preparation

Dataset JSON files live under `data/`. Download them with:

```bash
cd data
python download_data.py --dataset MATH_train
python download_data.py --dataset MATH500
cd ..
```

Available datasets: `MATH_train`, `PrimeIntellect` (training prompts) and `MATH500`, `GSM8K`, `MBPP`, `HumanEval` (evaluation).

## Workflow

The training pipeline consists of three stages:

1. Rollout trajectory generation
2. Trajectory preprocessing
3. T3D training

## Stage 1 — Rollout (Trajectory Generation)

Rollout and evaluation read the model from a local directory, so download the teacher first:

```bash
huggingface-cli download JetLM/SDAR-4B-Chat --local-dir /path/to/SDAR-4B-Chat
```

```bash
python sample/sdar_sample.py config=configs/rollout/sdar_sample.yaml \
    evaluation.checkpoint_path=/path/to/SDAR-4B-Chat \
    evaluation.eval_dataset=MATH_train \
    evaluation.data_type=math \
    rollout.block_size=4 \
    rollout.denoising_steps_per_block=4 \
    rollout.max_token=2000 \
    rollout.temperature=0.1 \
    rollout.top_p=1.0 \
    rollout.top_k=0 \
    rollout.remasking_strategy="low_confidence_static"
```

The rollout is written to `artifacts/rollout_sdar/temp_data/`.

## Stage 2 — Preprocess Trajectories

Copy the rollout file into `data/`:

```bash
cp artifacts/rollout_sdar/temp_data/outputs-eval-SDAR-4B-Chat-dspb4-bs4-MATH_train.json \
   data/SDAR-4B-Chat-MATH_train.json
```

Then preprocess:

```bash
python data/rename_key.py --model_name SDAR-4B-Chat --dataset MATH_train
```

This produces `data/SDAR-4B-Chat-MATH_train_renamed.json`, the file referenced by `dataset.optimization_data` in the training config.

## Stage 3 — Training (T3D)

### (Optional) W&B Login

```bash
wandb login
```

### Launch Training

```bash
accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8889 \
  --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero1.yaml \
  train/self_ddo_sdar_full.py \
  config=configs/train/ddo_sdar_full_self.yaml
```

Checkpoints are saved under `artifacts/experiments/<model>/<data_type>/<project>/<timestamp>/ckpt/`. The reference model is refreshed from the student every `training.round_interval` global steps, and a checkpoint named `optimized-round<N>-step<S>` is written at each refresh.

| Setting | Config |
| --- | --- |
| SDAR-4B-Chat, math | `configs/train/ddo_sdar_full_self.yaml` |
| SDAR-4B-Chat, code | `configs/train/ddo_sdar_4b_code.yaml` |
| SDAR-1.7B-Chat, math | `configs/train/ddo_sdar_1_7b_math.yaml` |
| SDAR-1.7B-Chat, code | `configs/train/ddo_sdar_1_7b_code.yaml` |

The configs assume 8 GPUs; with fewer GPUs, scale `training.gradient_accumulation_steps` to keep the effective batch size and pick the matching file in `accelerate_configs/`.

## Checkpoint

We provide a [checkpoint](https://huggingface.co/Tyrion279/SDAR-4B-Chat-t3d-math): SDAR-4B-Chat trained with T3D on MATH_train.

```bash
huggingface-cli download Tyrion279/SDAR-4B-Chat-t3d-math --local-dir /path/to/checkpoint
```

## Evaluation

`eval.py` runs the rollout, executes generated code when `data_type=code`, and scores the answers:

```bash
python eval.py config=configs/eval/eval_sdar.yaml \
    evaluation.checkpoint_path=/path/to/checkpoint \
    evaluation.eval_dataset=MATH500 \
    evaluation.data_type=math \
    rollout.block_size=4 \
    rollout.denoising_steps_per_block=1 \
    rollout.temperature=0.1
```

The number of tokens decoded per step is `rollout.block_size / rollout.denoising_steps_per_block`. Results are appended to `artifacts/<experiment.project>/results/`, together with decoding statistics (average denoising steps, tokens per step, throughput and latency).

| Scenario | Overrides |
| --- | --- |
| Dynamic decoding | `rollout.remasking_strategy=low_confidence_dynamic rollout.dynamic_threshold=0.9` |
| Greedy decoding | `rollout.temperature=0` |
| Code benchmarks | `evaluation.eval_dataset=MBPP evaluation.data_type=code` (or `HumanEval`) |

## Configuration Guide

| Component  | Location                          |
| ---------- | --------------------------------- |
| Rollout    | `configs/rollout/sdar_sample.yaml` |
| Training   | `configs/train/`                  |
| Evaluation | `configs/eval/eval_sdar.yaml`     |
| Multi-GPU  | `accelerate_configs/`             |

All entry points accept OmegaConf overrides on the command line (`key.subkey=value`). The SDAR base models are loaded at a pinned Hub revision (`utils/hf_compat.py`) that matches `transformers==4.52.4`.

A CPU-only test suite is available with `python -m pytest tests/`.

## Troubleshooting: CUDA / Compilation Issues

If compilation fails, try:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"

export PATH="$CONDA_PREFIX/bin:$PATH"
export CPATH="$CONDA_PREFIX/include:$CPATH"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
```

## BibTeX
```bibtex
@misc{zhang2026fewstepdiffusionlanguagemodels,
      title={Few-Step Diffusion Language Models via Trajectory Self-Distillation}, 
      author={Tunyu Zhang and Xinxi Zhang and Ligong Han and Haizhou Shi and Xiaoxiao He and Zhuowei Li and Hao Wang and Kai Xu and Akash Srivastava and Chengzhi Mao and Hao Wang and Vladimir Pavlovic and Dimitris N. Metaxas},
      year={2026},
      eprint={2602.12262},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.12262}, 
}
```

# License
The code is released under the MIT License.
