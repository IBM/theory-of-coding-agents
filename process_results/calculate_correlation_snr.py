"""
Computes the rank correlation (Spearman's rho, Kendall's tau) and Signal-to-Noise Ratio (SNR) for different execution-based code selection strategies.

Methodology:
- Signal (μ): The exact score of a code evaluated on the full, unbootstrapped empirical grid of tests.
- Noise (σ²): The variance of that code's score across 50 bootstrap resamples (with replacement) of the grid.
- Rank correlations measure how well the full-grid scores align with the hidden oracle ground-truth.

Caveats & Tradeoffs:
1. Bias-Variance Tradeoff: Hard execution is unbiased (high rank correlation) but highly sensitive to test noise (low SNR). Soft execution suffers slight calibration bias (lower correlation) but provides massive variance reduction (high SNR).
2. Plug-in Principle: The SNR calculation mathematically treats the finite, empirically generated grid as the "true" population distribution to anchor the signal.
3. Strict Intersection Support & Geometric Mean: Codes that perfectly pass or fail all generated tests exhibit zero bootstrap variance. To ensure a mathematically fair comparison, SNR is calculated strictly on the intersection support (i.e. only on codes that exhibit non-zero variance across ALL evaluated strategies simultaneously). Because SNRs span multiple orders of magnitude, we aggregate using the geometric mean.
4. Final Estimator SNR vs. Pairwise Similarity SNR: Theorem 3.10 theoretically bounds the SNR dominance for pairwise functional similarity. The final selection heuristics evaluated here (MBR and MaxPass) compute scores that are linear aggregations of these functional similarities against reference sets or target outputs. Thus, measuring the SNR of the final estimator score here serves as the practical, observable manifestation of the theoretical variance reduction proven on the underlying similarity kernel.
"""

import os, sys
import numpy as np
import warnings
from scipy.stats import spearmanr, kendalltau, ConstantInputWarning
warnings.filterwarnings('ignore', category=ConstantInputWarning)
# Add the parent directory to PYTHONPATH to import utils and methods
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_jsonl
from scripts.process_results import get_k_codes

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = 'qwen3-235b-A22b-instruct'
datasets = ['big_code_bench_hard', 'leet_code', 'qiskit_human_eval']
seeds = ['seed-10', 'seed-11', 'seed-12', 'seed-13', 'seed-14']
generation_mode = 'execute_codes_on_tests'
max_rounds = 10
strategies = ['max_pass_hard', 'max_pass_soft', 'mbr_hard', 'mbr_soft']

num_bootstraps = 50

for dataset in datasets:
    print(f"\n======================================")
    print(f"Dataset: {dataset}")
    print(f"======================================")

    all_results = {}

    for seed in seeds:
        print(f"Processing {seed}...")
        result_dir = os.path.join(base_dir, 'results', dataset, model, seed, generation_mode)
        task_files = sorted([f for f in os.listdir(result_dir) if f.startswith('result_') and f.endswith('.jsonl')])

        num_tasks = len(task_files)
        print(f"Found {num_tasks} tasks.")
        scores_data = {s: {r: {i: {} for i in range(num_tasks)} for r in range(max_rounds)} for s in strategies} # maps r_code to (heuristic_score, oracle_score, var_h_boot)

        for i, task_file in enumerate(task_files):
            exec_file = os.path.join(result_dir, task_file)
            execs = read_jsonl(exec_file)

            for r in range(1, max_rounds): # number of samples available to the estimator (for SNR) and number of codes among which to select (for rank metrics), starting at 1 since pearson/kendall correlation are undefined at 0
                r_execs = [e for e in execs if e['r_code'] <= r and (e['r_tests'] <= r or e['r_tests'] == -1)]
                b_scores = {s: {c_idx: [] for c_idx in range(r + 1)} for s in strategies} # bootstrapped scores for snr estimation

                for b in range(num_bootstraps):
                    T_indices = np.random.choice(r + 1, size=r + 1, replace=True)
                    C_indices = np.random.choice(r + 1, size=r + 1, replace=True)

                    b_execs = []
                    for new_c, orig_c in enumerate(C_indices): # new index (subsampled grid) vs old index (full grid)
                        for new_t, orig_t in enumerate(T_indices):
                            # relabels code and test indices for bootstrap
                            for e in r_execs:
                                if e['r_code'] == orig_c and e['r_tests'] == orig_t:
                                    new_e = dict(e) # shallow copy
                                    new_e['r_code'] = new_c
                                    new_e['r_tests'] = new_t
                                    b_execs.append(new_e)
                                    break

                        # relabels oracle test indices for bootstrap
                        for e in r_execs:
                            if e['r_code'] == orig_c and e['r_tests'] == -1:
                                new_e = dict(e) # shallow copy
                                new_e['r_code'] = new_c
                                new_e['r_tests'] = -1
                                b_execs.append(new_e)
                                break

                    # Only retain one example of the score for each code per bootstrap
                    for strategy in strategies:
                        b_codes = get_k_codes(b_execs, r, strategy, -1, i) # retrieves all codes with their scores attached based on the bootstrapped samples
                        recorded_for_this_b = set()
                        for code_info in b_codes:
                            new_c = code_info['r_code']
                            score = code_info['score']
                            orig_c = C_indices[new_c]
                            
                            if orig_c not in recorded_for_this_b:
                                b_scores[strategy][orig_c].append(score)
                                recorded_for_this_b.add(orig_c)

                # computes non-bootstrapped scores for correlations and also bootstrapped variance for each code
                for strategy in strategies:
                    scored_codes = get_k_codes(execs, r, strategy, -1, i) # retrieves all codes with their scores attached based on the full sample set

                    for code_info in scored_codes:
                        gt_evals = [res for res in execs if res['r_tests'] == -1 and res['r_code'] == code_info['r_code']]
                        assert len(gt_evals) == 1, f"Expected 1 gt_eval for code {code_info['r_code']}, got {len(gt_evals)}"
                        oracle_score = gt_evals[0]['evaluation']['score']
                        heuristic_score = code_info['score']

                        b_list = b_scores[strategy][code_info['r_code']]
                        var_h_boot = np.var(b_list) if len(b_list) > 1 else np.nan

                        scores_data[strategy][r][i][code_info['r_code']] = (heuristic_score, oracle_score, var_h_boot)

        results = {strategy: {r: {'task_rhos': [], 'task_taus': [], 'task_snrs': []} for r in range(1, max_rounds)} for strategy in strategies}

        for r in range(1, max_rounds):
            for i in range(num_tasks):
                # 1. Compute correlations independently per strategy
                for strategy in strategies:
                    # To compute correlation, order doesn't technically matter as long as h and o are paired,
                    # but sorting by r_code ensures consistency
                    c_indices = sorted(scores_data[strategy][r][i].keys())
                    h_scores = [scores_data[strategy][r][i][c][0] for c in c_indices]
                    o_scores = [scores_data[strategy][r][i][c][1] for c in c_indices]
                    
                    rho, _ = spearmanr(h_scores, o_scores)
                    tau, _ = kendalltau(h_scores, o_scores)
                    if not np.isnan(rho) and not np.isnan(tau):
                        results[strategy][r]['task_rhos'].append(rho)
                        results[strategy][r]['task_taus'].append(tau)

                # 2. Compute strict intersection mask for SNR (only codes with valid variance in ALL strategies)
                valid_codes = []
                for c_idx in range(r + 1):
                    valid = True
                    for strategy in strategies:
                        v = scores_data[strategy][r][i][c_idx][2]
                        if np.isnan(v) or v <= 1e-10:
                            valid = False
                            break
                    if valid:
                        valid_codes.append(c_idx)

                # 3. Append SNR for the strict intersection subset
                for strategy in strategies:
                    for c_idx in valid_codes:
                        h = scores_data[strategy][r][i][c_idx][0]
                        v = scores_data[strategy][r][i][c_idx][2]
                        results[strategy][r]['task_snrs'].append((h ** 2) / v)

        # 4. Finalize results
        for strategy in strategies:
            for r in range(1, max_rounds):
                rhos = results[strategy][r]['task_rhos']
                taus = results[strategy][r]['task_taus']
                snrs = [s for s in results[strategy][r]['task_snrs'] if s > 1e-10] # Filter exactly 0 to prevent log(0), so we remove both infinite and 0 SNRs from the geometric mean
                
                results[strategy][r] = {
                    'spearman_rho': np.mean(rhos) if rhos else np.nan,
                    'kendall_tau': np.mean(taus) if taus else np.nan,
                    'snr': np.exp(np.mean(np.log(snrs))) if snrs else np.nan
                }

        all_results[seed] = results

    avg_results = {s: {} for s in strategies}
    for strategy in strategies:
        for m in range(1, max_rounds):
            avg_rho = np.nanmean([all_results[s][strategy][m]['spearman_rho'] for s in seeds])
            avg_tau = np.nanmean([all_results[s][strategy][m]['kendall_tau'] for s in seeds])
            avg_snr = np.nanmean([all_results[s][strategy][m]['snr'] for s in seeds])

            avg_results[strategy][m] = {
                'spearman_rho': avg_rho,
                'kendall_tau': avg_tau,
                'snr': avg_snr
            }

    print("\n=== Averaged Results Across Seeds ===")
    for strategy, m_data in avg_results.items():
        print(f"\nStrategy: {strategy}")
        for m, stats in m_data.items():
            print(f"  r={m}: Spearman's rho={stats['spearman_rho']:.4f}, Kendall's tau={stats['kendall_tau']:.4f}, SNR={stats['snr']:.4f}")
