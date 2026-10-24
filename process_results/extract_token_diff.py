import os
import json
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = 'qwen3-235b-A22b-instruct'
datasets = ['big_code_bench_hard', 'leet_code', 'qiskit_human_eval']
seeds = ['seed-10', 'seed-11', 'seed-12', 'seed-13', 'seed-14']

pairs = [('tfs', 'tf'), ('cfs', 'cf'), ('inds', 'ind')]

max_rounds = 10
target_round = max_rounds - 1

def get_tokens(dataset, method):
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    
    # Prefix execute_thompson_sampling_
    full_method = f"execute_thompson_sampling_{method}"
    model_dir = os.path.join(base_dir, 'results', dataset, model)
    for seed in seeds:
        file_path = os.path.join(model_dir, seed, full_method, 'pass@1_last.jsonl')
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data['round'] == target_round:
                        prompt_tokens.append(data.get('prompt_tokens', 0))
                        completion_tokens.append(data.get('completion_tokens', 0))
                        total_tokens.append(data.get('total_tokens', 0))
        except FileNotFoundError:
            pass
            
    return {
        'prompt': np.mean(prompt_tokens) if prompt_tokens else np.nan,
        'completion': np.mean(completion_tokens) if completion_tokens else np.nan,
        'total': np.mean(total_tokens) if total_tokens else np.nan
    }

print("Token differences (Insight Reformulation (Compactification) - Summary Concatenation) at round 9 (averaged across seeds)")
print(f"| {'Factorization':<15} | {'Insight Mechanism':<25} | {'Dataset':<20} | {'Conditioned (Prompt)':<20} | {'Generated (Completion)':<22} | {'Total':<20} |")
print(f"| {':'+'-'*14} | {':'+'-'*24} | {':'+'-'*19} | {':'+'-'*19} | {':'+'-'*21} | {':'+'-'*19} |")

fact_map = {
    'tfs': 'test first', 'tf': 'test first',
    'cfs': 'code first', 'cf': 'code first',
    'inds': 'independent', 'ind': 'independent'
}

mech_map = {
    'tfs': 'summary concatenation', 'tf': 'insight reformulation',
    'cfs': 'summary concatenation', 'cf': 'insight reformulation',
    'inds': 'summary concatenation', 'ind': 'insight reformulation'
}

for s_method, i_method in pairs: 
    for dataset in datasets:
        s_tokens = get_tokens(dataset, s_method)
        i_tokens = get_tokens(dataset, i_method)
        
        fact = fact_map[s_method]
        
        print(f"| {fact:<15} | {mech_map[s_method]:<25} | {dataset:<20} | {s_tokens['prompt']:<20.0f} | {s_tokens['completion']:<22.0f} | {s_tokens['total']:<20.0f} |")
        print(f"| {fact:<15} | {mech_map[i_method]:<25} | {dataset:<20} | {i_tokens['prompt']:<20.0f} | {i_tokens['completion']:<22.0f} | {i_tokens['total']:<20.0f} |")
        
        diff_prompt = i_tokens['prompt'] - s_tokens['prompt']
        diff_completion = i_tokens['completion'] - s_tokens['completion']
        diff_total = i_tokens['total'] - s_tokens['total']
        
        pct_prompt = (diff_prompt / s_tokens['prompt'] * 100) if s_tokens['prompt'] else np.nan
        pct_completion = (diff_completion / s_tokens['completion'] * 100) if s_tokens['completion'] else np.nan
        pct_total = (diff_total / s_tokens['total'] * 100) if s_tokens['total'] else np.nan
        
        diff_prompt_str = f"{diff_prompt:+.0f} ({pct_prompt:>+7.2f}%)" if not np.isnan(diff_prompt) else "nan (nan%)"
        diff_comp_str = f"{diff_completion:+.0f} ({pct_completion:>+7.2f}%)" if not np.isnan(diff_completion) else "nan (nan%)"
        diff_total_str = f"{diff_total:+.0f} ({pct_total:>+7.2f}%)" if not np.isnan(diff_total) else "nan (nan%)"
        
        print(f"| {fact:<15} | {'DIFF (insight-summary)':<25} | {dataset:<20} | {diff_prompt_str:<20} | {diff_comp_str:<22} | {diff_total_str:<20} |")
        print("|" + "-"*135 + "|")
