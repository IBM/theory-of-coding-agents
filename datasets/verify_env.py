from datasets import CodeDataset
from utils.evaluation.code_evaluator import evaluate_code
from tqdm import tqdm
import pprint

if __name__ == "__main__":
    for dataset_name, dataset_file in [('qiskit_human_eval_X', 'datasets/qiskit_human_eval_X.jsonl'),
                                       ('qiskit_human_eval', 'datasets/qiskit_human_eval.jsonl'),
                                       ('leet_code_minus', 'datasets/leet_code.jsonl'),
                                       ('human_eval', 'datasets/human_eval.jsonl'), 
                                       ('mbpp', 'datasets/mbpp.jsonl'),
                                       ('big_code_bench_hard', 'datasets/big_code_bench_v0.1.4.jsonl'),
                                       ('big_code_bench', 'datasets/big_code_bench_v0.1.4.jsonl')]:#,
        ds = CodeDataset(dataset_name, dataset_file)
        print(f"verify {dataset_name}")
        for i in tqdm(ds.data_range):
            data_row = ds.get_data(i)
            eval = evaluate_code(data_row['solution'], data_row['tests'], data_args=data_row['data_args'], time_limit=180)
            assert eval['score'] == 1.0, f"DATASET ENTRY {i}: {pprint.pformat(data_row)}\n" + f"EVALUATION: {pprint.pformat(eval)}"
