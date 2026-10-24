"""
Generates a comparative markdown table for final Pass@1 performance between the B4 baseline and MaxPass Soft.

Methodology:
- Extracts the final round (r=9) Pass@1 rates for both B4 and MaxPass Soft strategies across all seeds.
- Computes the mean and standard error (std / sqrt(n)) to show statistical significance.
- Demonstrates the downstream performance gains achieved by leveraging the high-SNR Soft estimation strategy.
"""

import os
import json
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = 'qwen3-235b-A22b-instruct'
datasets = ['big_code_bench_hard', 'leet_code', 'qiskit_human_eval']
generation_mode = "execute_codes_on_tests"
seeds = ['seed-10', 'seed-11', 'seed-12', 'seed-13', 'seed-14']

print("Dataset | B4 Pass@1 | MaxPass Soft Pass@1")
print("---|---|---")

for dataset in datasets:
    model_dir = os.path.join(base_dir, 'results', dataset, model)
    b4_rates = []
    mps_rates = []
    
    for seed in seeds:
        # B4
        b4_path = os.path.join(model_dir, seed, generation_mode, 'pass@1_b4.jsonl')
        if os.path.exists(b4_path):
            with open(b4_path, 'r') as f:
                for line in f.readlines():
                    data = json.loads(line)
                    if data['round'] == 9:
                        b4_rates.append(data['pass_rate'])
                        break
                        
        # MaxPass Soft
        mps_path = os.path.join(model_dir, seed, generation_mode, 'pass@1_max_pass_soft.jsonl')
        if os.path.exists(mps_path):
            with open(mps_path, 'r') as f:
                for line in f.readlines():
                    data = json.loads(line)
                    if data['round'] == 9:
                        mps_rates.append(data['pass_rate'])
                        break
    
    b4_str = "-"
    if b4_rates:
        b4_str = f"{np.mean(b4_rates)*100:.2f} ± {(np.std(b4_rates)/np.sqrt(len(b4_rates)))*100:.2f}%"
        
    mps_str = "-"
    if mps_rates:
        mps_str = f"{np.mean(mps_rates)*100:.2f} ± {(np.std(mps_rates)/np.sqrt(len(mps_rates)))*100:.2f}%"
        
    print(f"{dataset} | {b4_str} | {mps_str}")
