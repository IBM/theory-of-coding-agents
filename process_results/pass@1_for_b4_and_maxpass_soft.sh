#!/bin/bash
set -e
export PYTHONPATH=.

datasets=("big-code-bench-hard" "leet-code" "qiskit-human-eval")
seeds=("10" "11" "12" "13" "14")
strategies=("b4" "max_pass_soft")
model="qwen3-235B-A22B-instruct"

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        for strategy in "${strategies[@]}"; do
            echo "Running $strategy evaluation for $dataset, seed $seed"
            ~/.local/bin/micromamba run -n code_assistant python scripts/process_results.py --config "configs/${dataset}-${model}.json" --config_key execute_codes_on_tests --seed "$seed" --k 1 --selection_strategy "$strategy" || echo "Failed for $dataset $seed $strategy"
        done
    done
done
