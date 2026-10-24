from utils import read_jsonl, create_dirs, init_log, create_or_clear_file, append_jsonl, print_log
from utils.evaluation.code_evaluator import evaluate_code
from typing import *
from tqdm import tqdm
import os
from utils.running_utils import load_env

def RunCodesOnTests(
    index: int,
    data: Dict,
    code_generations: List[Dict],
    test_generations: List[Dict],
    result_file: str,
    log_file: str,
    time_limit: float
) -> None:
    """
    Executes generated codes on the generated tests.

    Args:
        index: The index of the problem being solved. Used for logging/file naming.
        data: A dictionary containing problem data, including 'prompt' and 'data_args'.
        code_generations: A list of unconditionally generated code strings.
        test_generations: A list of test generation dictionaries.
        result_file: Path to the file where generation results will be saved (JSONL format).
        log_file: Path to the log file for this run.
        time_limit: The maximum time allowed for code execution during evaluation.
    """
    code_generations = [c['code'] for c in code_generations] # List[str]
    test_generations = [[test['test'] for test in test_generation['tests']] for test_generation in test_generations] # List[List[str]]

    data_args = data['data_args']

    # load partial result in case some evaluations were already conducted
    result = read_jsonl(result_file)
    if len(result) > 0:
        r = result[-1]['r'] + 1
    else:
        create_or_clear_file(result_file)
        create_or_clear_file(log_file)
        r = 0 # round counter

    td = tqdm(initial=r, total=len(code_generations)*(len(test_generations)+1))
    td.set_description(f'''[{index}]''')

    # compares codes with self-generated tests
    for r_code, code in enumerate(code_generations):
        for r_tests, tests in enumerate(test_generations):
            if any(res['r_code'] == r_code and res['r_tests'] == r_tests for res in result): # already evaluated this combination
                continue
            res = evaluate_code(
                code=code,
                test_cases=tests,
                data_args=data_args,
                time_limit=time_limit,
            )
            item = {
                'r': r,
                'r_code': r_code,
                'r_tests': r_tests,
                'code': code,
                'tests': tests,
                'evaluation': res,
            }
            td.update(1)
            r += 1
            append_jsonl(result_file, item)
            print_log('execution result [code]', code, 1)
            print_log('execution result [test]', tests, 1)
            print_log('execution result [statistics]', {key: value for key,value in res.items() if key!='feedbacks'}, 1)
            print_log('execution result [feedbacks]', res['feedbacks'], 1)

    ## compares codes with ground-truth tests
    tests = data['tests']
    for r_code, code in enumerate(code_generations):
        if any(res['r_code'] == r_code and res['r_tests'] == -1 for res in result):
            continue
        res = evaluate_code(
            code=code,
            test_cases=tests,
            data_args=data_args,
            time_limit=time_limit,
        )
        item = {
            'r': r,
            'r_code': r_code,
            'r_tests': -1, # indicates ground-truth tests
            'code': code,
            'tests': tests,
            'evaluation': res,
        }
        td.update(1)
        r += 1
        append_jsonl(result_file, item)
        print_log('test result [code]', code, 1)
        print_log('test result [test]', tests, 1)
        print_log('test result [statistics]', {key: value for key,value in res.items() if key!='feedbacks'}, 1)
        print_log('test result [feedbacks]', res['feedbacks'], 1)

if __name__ == '__main__':
    env = load_env()

    dataset = env['dataset']
    args = env['args']
    config = env['config']
    config_key = env['config_key']
    result_dir = env['result_dir']
    log_dir = env['log_dir']

    assert config_key.startswith('execute'), f'config_key {config_key} does not start with `execute`'

    run_config = config[config_key]

    codes_dir = os.path.join(result_dir.rstrip(config_key), run_config['code_dir'])
    tests_dir = os.path.join(result_dir.rstrip(config_key), run_config['test_dir'])
    create_dirs(result_dir)
    create_dirs(log_dir)
    
    for i in dataset.data_range:
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        log_file = os.path.join(log_dir, f'log_{i}.log')
        init_log(log_file)

        tests_file = os.path.join(tests_dir, f'result_{i}.jsonl')
        test_generations = read_jsonl(tests_file)[:run_config['max_test_gens']]

        codes_file = os.path.join(codes_dir, f'result_{i}.jsonl')
        code_generations = read_jsonl(codes_file)[:run_config['max_code_gens']]
        print(tests_file)
        print(codes_file)
        print(test_generations)
        print(code_generations)
        RunCodesOnTests(
            index=i,
            data=dataset.get_data(i),
            code_generations=code_generations,
            test_generations=test_generations,
            result_file=result_file,
            log_file=log_file,
            time_limit=args.time_limit
        )
    print(f"DONE. Logs are in {log_dir}")