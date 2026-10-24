# A Theoretical Analysis of Test-Driven Code Generation

[![Paper](https://img.shields.io/badge/Paper-NeurIPS_2026-blue)](#)
[![arXiv](https://img.shields.io/badge/arXiv-2602.06098-B31B1B.svg)](https://arxiv.org/abs/2602.06098)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](#)

Welcome to the official repository for the paper **"A Theoretical Analysis of Test-Driven Code Generation"**. This paper was accepted at **NeurIPS 2026** (The Fortieth Annual Conference on Neural Information Processing Systems) as a **spotlight paper**. This repository provides all the code, environments, and instructions necessary to reproduce our empirical results.

---

## Table of Contents
1. [Hardware and Operating System](#hardware-and-operating-system)
2. [Environment Setup](#environment-setup)
3. [LLM Hosting](#llm-hosting)
4. [Main Experiments](#main-experiments)
    - [Code-Test Generation](#1-code-test-generation)
    - [Code-Test Cross-Execution](#2-code-test-cross-execution)
    - [Post-Generation Selection Heuristics](#3-post-generation-selection-heuristics)
5. [Results Processing and Visualization](#results-processing-and-visualization)
6. [Full Agentic Setting](#full-agentic-setting)
7. [Citation](#citation)

---

## Hardware and Operating System

All experiments were conducted on compute nodes with the following specifications:
- **CPU**: AMD EPYC 7763 (64 Cores)
- **RAM**: 2TB
- **GPU**: NVIDIA A100-SXM4 (80GB)
- **OS**: Red Hat Enterprise Linux 9.4
- **CUDA**: 12.4

---

## Environment Setup

We use [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) for fast and reliable environment management. 

1. Ensure `micromamba` is installed and on your `PATH`.
2. Define your environments directory:
   ```bash
   export MICROMAMBA_ENVS="?/micromamba/envs"
   ```
3. Configure all required dataset-specific environments:
   ```bash
   for f in environments/*.yaml; do micromamba create -f "$f" -y; done
   ```
4. Activate the main environment for this project:
   ```bash
   micromamba activate code_assistant
   ```
5. Verify the installed environments by running the canonical solutions against ground-truth tests:
   ```bash
   python -m datasets.verify_env
   ```
   *Note: Using the `-m` flag is required for correct dependency resolution. Some environments (e.g., those requiring `matplotlib`) will initialize on the first run. The `BigCodeBenchHard` dataset may occasionally fail due to external live dependencies (like website content changes).*

---

## LLM Hosting

Our experiments evaluate several state-of-the-art open-weight models. To start a `vLLM` server, allocate a compute node and run the corresponding command for your target model:

| Model | Required GPUs (A100 80GB) | Command |
|-------|:---:|---------|
| **Qwen3-235B-A22B-Instruct** | 8 | `vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 --tensor_parallel_size 8 --port 8000 --enforce-eager` |
| **MiniMax-M2.1** | 4 | `vllm serve MiniMaxAI/MiniMax-M2.1 --tensor_parallel_size 4 --trust_remote_code --port 8000` |
| **GPT-OSS-120B** | 4 | `vllm serve openai/gpt-oss-120b --tensor_parallel_size 4 --enforce-eager --port 8000` |
| **Qwen3-Next-80B-A3B** | 4 | `vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --tensor-parallel-size 4 --served-model-name qwen3-next --port 8000 --enforce-eager` |

---

## Main Experiments

> **Warning**
> Ensure your system has sufficient memory and CPU timeout slack (5-10x the canonical solution's time). Certain tests in `BigCodeBenchHard` can consume up to 128GB of RAM.

### 1. Code-Test Generation

Generate code-test pairs across multiple rounds using the following scripts located in `scripts/`:

| Method | Script | Config Key |
|--------|--------|------------|
| **Unconditioned Code Generation** | `gen_codes.py` | `gen_codes` |
| **Unconditioned Test Generation** | `gen_tests.py` | `gen_tests` |
| **In-Context Thompson Sampling** | `in_context_thompson_sampling.py` | `thompson_sampling_?` |

**Execution Parameters**:
Pass `--config_key`, `--config`, `--api_key` (use `NA` for vLLM), `--base_url`, and `--seed`. The `--config` parameter maps to specific strategy definitions (e.g., `thompson_sampling_ind`) found in the `configs/` directory.

### 2. Code-Test Cross-Execution

To enable post-generation selection heuristics, generated codes must be executed against both generated tests and the ground-truth test suite.

```bash
python -m scripts.run_codes_on_tests \
    --config configs/<your_config>.json \
    --config_key <your_key> \
    --seed <your_seed>
```

### 3. Post-Generation Selection Heuristics

After cross-execution, calculate `Pass@k` scores based on various heuristic strategies:

```bash
python -m scripts.process_results \
    --config configs/<your_config>.json \
    --config_key <your_key> \
    --selection_strategy <strategy_name> \
    --seed <your_seed>
```
*Supported Strategies:* `max_pass_hard`, `max_pass_soft`, `mbr_hard`, `mbr_soft`, `codet_hard`, `codet_soft`, `last`.

---

## Results Processing and Visualization

Tools for processing data and compiling visual assets are located in the `process_results/` directory:

- **Interactive Dashboard**: Launch a Gradio web UI to explore Thompson Sampling iterations and the Code vs. Test Suite performance matrix interactively:
  ```bash
  python process_results/dashboard.py
  ```
- **Paper Figures & Tables**: Use the `process_results/plot.ipynb` Jupyter Notebook to aggregate results, compile LaTeX tables, and generate PGF plots (using `pgf_compiler.sh`). Outputs are saved to `process_results/plots/`.
- **Advanced Analytics**: Execute standalone scripts like `analyze_test_precision_recall_length_drift.py` and `calculate_correlation_snr.py` for deeper dives into signal-to-noise ratios and structural drift.

---

## Full Agentic Setting

For instructions and a comprehensive tutorial on running the full agentic setting (including setting up SWE-bench, mini-swe-agent, and the oracle test injection), please refer to the dedicated [SWE-bench tutorial](scripts/swe-bench/README.md).


---

## Citation

If you use our codebase or find our work helpful, please consider citing our paper:

```bibtex
@inproceedings{
menet2026theoreticalanalysistestdrivencodegeneration,
title={A Theoretical Analysis of Test-Driven Code Generation},
author={Nicolas Menet and Michael Hersche and Andreas Krause and Abbas Rahimi},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://arxiv.org/abs/2602.06098}
}

```