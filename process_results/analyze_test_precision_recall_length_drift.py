"""
Analyzes the drift of Test Suite Metrics (Recall/TPR and Precision/PPV) over Thompson Sampling rounds.

Methodology:
- For each round `r`, we evaluate the generated test suite `T_r` against the code `c_r` generated in the same round.
- A code `c_r` is "correct" if its oracle score == 1.0, and "incorrect" if its oracle score != 1.0.
- A test suite `T_r` "passes" the code if all individual tests in the suite pass.
- Computes:
  - Recall (TPR): Proportion of correct codes that pass the generated test suite.
  - Precision (PPV): Probability that a code is genuinely correct given that it passes the test suite.
  - Average test suite length: Used to prove the agent does not trivially "delete" tests to bypass constraints.
  
Why Precision?: As the generated code improves over rounds, the remaining incorrect codes become subtle "hard negatives". 
An increasing Precision proves that the test suites do not lose their constraints against these harder bugs, and passing codes become *more* trustworthy over time.
"""

import os
import sys
import numpy as np
# Add the parent directory to PYTHONPATH to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_jsonl

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = 'qwen3-235b-A22b-instruct'
datasets = ['big_code_bench_hard', 'leet_code', 'qiskit_human_eval']
seeds = ['seed-10', 'seed-11', 'seed-12', 'seed-13', 'seed-14']
generation_mode = 'execute_thompson_sampling_tfs'
max_rounds = 10

print("Analyzing Test Metrics Drift over rounds...")
print("| Dataset | Round | Avg Test Suite Length | Recall (TPR) | Precision (PPV) |")
print("| :---: | :---: | :---: | :---: | :---: |")

for dataset in datasets:
    # Aggregate across all tasks and seeds for the dataset
    all_test_lengths = {r: [] for r in range(max_rounds)}
    total_incorrect = {r: 0 for r in range(max_rounds)}
    total_correct = {r: 0 for r in range(max_rounds)}
    total_false_positives = {r: 0 for r in range(max_rounds)}
    total_true_positives = {r: 0 for r in range(max_rounds)}
    
    for seed in seeds:
        result_dir = os.path.join(base_dir, 'results', dataset, model, seed, generation_mode)
        task_files = sorted([f for f in os.listdir(result_dir) if f.startswith('result_') and f.endswith('.jsonl')])
        for task_file in task_files:
            exec_file = os.path.join(result_dir, task_file)
            execs = read_jsonl(exec_file)
            
            for r in range(max_rounds):
                # Find oracle evaluation for code r
                oracle_eval = next(res for res in execs if res['r_code'] == r and res['r_tests'] == -1)
                is_incorrect = oracle_eval['evaluation']['score'] != 1.0
                
                # Find evaluation for code r on test suite r
                test_eval = next(res for res in execs if res['r_code'] == r and res['r_tests'] == r)
                status = test_eval['evaluation']['status']
                
                all_test_lengths[r].append(len(status))
                
                if is_incorrect:
                    total_incorrect[r] += 1
                    # A test suite passes the code if all individual tests pass
                    if len(status) > 0 and sum(status) == len(status):
                        total_false_positives[r] += 1
                else:
                    total_correct[r] += 1
                    if len(status) > 0 and sum(status) == len(status):
                        total_true_positives[r] += 1
                        
    # Print dataset results
    for r in range(max_rounds):
        avg_length = np.mean(all_test_lengths[r])
        
        # Guard against zero-division
        if total_correct[r] > 0:
            tpr = total_true_positives[r] / total_correct[r]
        else:
            tpr = np.nan
            
        # Calculate Precision (Positive Predictive Value)
        total_predicted_positive = total_true_positives[r] + total_false_positives[r]
        if total_predicted_positive > 0:
            precision = total_true_positives[r] / total_predicted_positive
        else:
            precision = np.nan
            
        print(f"| {dataset} | {r} | {avg_length:.2f} | {tpr:.4f} | {precision:.4f} |")
