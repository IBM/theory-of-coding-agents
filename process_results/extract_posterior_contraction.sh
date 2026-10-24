#!/bin/bash
set -e
export PYTHONPATH=.

datasets=("big-code-bench-hard" "leet-code" "qiskit-human-eval" "qiskit-human-eval-X")
seeds=("10" "11" "12" "13" "14")
model="qwen3-235B-A22B-instruct"

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Extracting posterior contraction (estimating deltas) for $dataset, seed $seed"
        ~/.local/bin/micromamba run -n code_assistant python scripts/process_results.py \
            --config "configs/${dataset}-${model}.json" \
            --config_key execute_thompson_sampling_tfs \
            --seed "$seed" \
            --k 1 \
            --selection_strategy last \
            --estimate_delta \
            || echo "Failed for $dataset $seed"
    done
done
