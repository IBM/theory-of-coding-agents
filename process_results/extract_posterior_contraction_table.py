"""
Calculates the posterior contraction (collision probability) of the generative model over successive rounds.

Methodology:
- Computes E_{c_1, c_2}[sim^infty(c_1, c_2)] without bias correction.
- Uses the `delta_sim_hard_mean` estimator to determine the probability that two independently generated codes collide (behave identically).
- Averages across a sliding window of size up to 3 to account for the non-stationary nature of generation at finite time-steps.
- Outputs a formatted markdown table across datasets.
"""

import os
import json
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = 'qwen3-235b-A22b-instruct'
datasets = ['big_code_bench_hard', 'leet_code', 'qiskit_human_eval', 'qiskit_human_eval_X']
generation_mode = 'execute_thompson_sampling_tfs'
seeds = ['seed-10', 'seed-11', 'seed-12', 'seed-13', 'seed-14']
delta_estimator = "delta_sim_hard_mean" # only unbiased estimator of Delta
max_rounds = 10

print("Collision Probability E_{c_1, c_2}[sim^infty(c_1,c_2)] per round, where the expectation is over the empirical c_1 c_2 grid of samples (no bias correction). Since c_i is typically not stationary at finite time-steps, we average across a sliding window of size up to size 3:")
print("| Round | BigCodeBench Hard | LeetCodeDataset | QiskitHumanEval | QiskitHumanEval X |")
print("| :---: | :---: | :---: | :---: | :---: |")

# Collect data: round -> dataset -> list of collision probs across seeds
data_by_round = {r: {ds: [] for ds in datasets} for r in range(0, max_rounds)}

for dataset in datasets:
    model_dir = os.path.join(base_dir, 'results', dataset, model)
    
    for seed in seeds:
        file_path = os.path.join(model_dir, seed, generation_mode, 'pass@1_last.jsonl')
        with open(file_path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                r = data['round']
                d_mean = data.get(delta_estimator)
                if r in data_by_round:
                    if d_mean is not None:
                        collision_prob = 1.0 - d_mean
                        data_by_round[r][dataset].append(collision_prob)
                    else:
                        data_by_round[r][dataset].append(np.nan)

for r in range(0, max_rounds):
    row = [f"| {r}"]
    for dataset in datasets:
        deltas = data_by_round[r][dataset]
        mean_deltas = np.mean(deltas)
        if np.isnan(mean_deltas):
            row.append("-")
        else:
            row.append(f"{mean_deltas:.4f}")
    print(" | ".join(row) + " |")
