import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = 'qwen3-235b-A22b-instruct'
datasets = ['big_code_bench_hard', 'leet_code', 'qiskit_human_eval']
seeds = ['seed-10', 'seed-11', 'seed-12', 'seed-13', 'seed-14']
generation_mode = 'execute_thompson_sampling_tfs'
max_rounds = 10

def process_task_file(exec_file):
    with open(exec_file, 'r') as f:
        execs = [json.loads(line) for line in f]
        
    local_total_tests = 0
    local_trivial_assert_true = 0
    local_trivial_no_assert = 0
    local_failure_rates = []
    
    test_suites = {}
    
    # Extract test suites and check for trivial tests
    for res in execs:
        if res.get('r_code') == 0 and res.get('r_tests') != -1: # not oracle test and only comparison with first code to not count any tests double
            r_tests = res['r_tests']
            tests = res.get('tests', [])
            test_suites[r_tests] = tests
            
            for test_str in tests:
                local_total_tests += 1
                if "assert True" in test_str:
                    local_trivial_assert_true += 1
                if "assert " not in test_str:
                    local_trivial_no_assert += 1
                    
    fails_per_test = {r_tests: [0] * len(tests) for r_tests, tests in test_suites.items()} # test suite round as key, list of failure counts per test as value
    num_programs = {r_tests: 0 for r_tests in test_suites.keys()} # test suite round as key, number of programs as value
    
    # Process evaluations
    for res in execs:
        r_tests = res.get('r_tests')
        r_code = res.get('r_code')
        if r_tests in test_suites and r_code in range(max_rounds):
            num_programs[r_tests] += 1
            status = res.get('evaluation', {}).get('status', [])
            num_tests = len(test_suites[r_tests])
            for k in range(num_tests):
                if k < len(status):
                    if status[k] == 0:
                        fails_per_test[r_tests][k] += 1
                else:
                    fails_per_test[r_tests][k] += 1 # handles compilation/syntax errors and timeouts
                    
    for r_tests, fails in fails_per_test.items():
        total_programs = num_programs[r_tests]
        if total_programs > 0:
            for fails_k in fails:
                local_failure_rates.append(fails_k / total_programs) # we compute the failure rate of each individual test case
                
    return (local_total_tests, local_trivial_assert_true, local_trivial_no_assert, local_failure_rates)

if __name__ == '__main__':
    exec_files = []
    for dataset in datasets:
        for seed in seeds:
            result_dir = os.path.join(base_dir, 'results', dataset, model, seed, generation_mode)
            if not os.path.exists(result_dir):
                continue
            task_files = sorted([f for f in os.listdir(result_dir) if f.startswith('result_') and f.endswith('.jsonl')])
            for task_file in task_files:
                exec_files.append(os.path.join(result_dir, task_file))
                
    total_tests = 0
    trivial_tests_assert_true = 0
    trivial_tests_no_assert = 0
    test_failure_rates = []

    print(f"Processing {len(exec_files)} task files using multiprocessing...")
    
    # Use all available CPU cores (typically optimal for CPU/IO-bound parsing tasks)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_task_file, ef): ef for ef in exec_files}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                total_tests += res[0]
                trivial_tests_assert_true += res[1]
                trivial_tests_no_assert += res[2]
                test_failure_rates.extend(res[3])
                
                if i % 100 == 0:
                    print(f"Processed {i}/{len(exec_files)} files...")
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")

    print(f"\nTotal individual tests analyzed: {total_tests}")
    if total_tests > 0:
        print(f"Tests with 'assert True': {trivial_tests_assert_true} ({trivial_tests_assert_true/total_tests*100:.4f}%)")
        print(f"Tests with NO assert: {trivial_tests_no_assert} ({trivial_tests_no_assert/total_tests*100:.4f}%)")

    if test_failure_rates:
        print("\nDistribution of probability that a generated program will fail a test:")
        print(f"Mean failure probability: {np.mean(test_failure_rates):.4f}")
        print("Percentiles:")
        for p in [0, 10, 25, 50, 75, 90, 100]:
            print(f"  {p}th: {np.percentile(test_failure_rates, p):.4f}")
        
        hist, bin_edges = np.histogram(test_failure_rates, bins=10, range=(0, 1))
        print("\nHistogram of failure probabilities:")
        for i in range(10):
            print(f"  [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}): {hist[i]} tests ({hist[i]/len(test_failure_rates)*100:.2f}%)")

