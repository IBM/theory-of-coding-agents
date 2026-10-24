from utils.generators import TestGenerator
from utils import append_jsonl, read_jsonl, create_or_clear_file, init_log, load_env
from tqdm import tqdm
from typing import Dict
import os

def GenTests(
        index: int,
        test_generator: TestGenerator,
        data: Dict,
        result_file: str,
        log_file: str,
        run_config: Dict,
) -> None:
    """
    Generates test cases using a test generator and logs the results.

    This function generates a specified number of test cases using the provided
    `test_generator` and configuration parameters. It appends the generated tests
    to a result file in JSONL format and logs progress using a tqdm progress bar.
    If the result or log files do not exist, they are created or cleared.

    Args:
        index (int): Index of the current test generation process, used for display. Should be set to the index of the provided dataset entry.
        test_generator (TestGenerator): An instance capable of generating tests.
        data (Dict): Dictionary containing the prompt among other information on the dataset entry
            Expected keys:
                - 'prompt' (str): The function signature and docstring to guide test generation.
        result_file (str): Path to the file where generated test results are stored.
        log_file (str): Path to the log file for tracking progress or errors.
        run_config (Dict): Configuration for test generation. Expected keys:
            - 'generations' (int): Total number of test generations to perform.
            - 'temperature' (float): Sampling temperature for generation.
            - 'max_tokens' (int): Maximum number of tokens to generate.

    Returns:
        None
    """
    generations = run_config['rounds'] 
    temperature = run_config['temperature']
    max_tokens = run_config['max_tokens']

    # load partial result in case some generations were already conducted
    result = read_jsonl(result_file)
    if len(result) > 0:
        r = result[-1]['r'] + 1
    else:
        create_or_clear_file(log_file)
        create_or_clear_file(result_file)
        r = 0 # rounds of test generation

    td = tqdm(initial=r, total=generations)
    td.set_description(f'''[{index}]''')

    prompt = data['prompt']

    while r < generations:
        gen = test_generator.generate_tests(
            task_description=prompt,
            generation_mode='initialize',
            max_tokens=max_tokens,
            temperature=temperature
        )

        all_tests = [{'test': t} for t in gen['tests']]
        append_jsonl(result_file, {
            'r': r,
            'tests': all_tests,
            'token_counts': gen['token_counts']
        })
        td.update(1)
        r += 1


if __name__ == '__main__':
    env = load_env()

    dataset = env['dataset']
    model = env['model']
    args = env['args']
    log_dir = env['log_dir']
    result_dir = env['result_dir']
    config = env['config']

    test_generator = TestGenerator(model)

    for i in dataset.data_range:
        data = dataset.get_data(i)

        # create files
        log_file = os.path.join(log_dir, f'log_{i}.log')
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        init_log(log_file)

        run_config = config[args.config_key]
        
        GenTests(
            index=i,
            test_generator=test_generator,
            data=data,
            result_file=result_file,
            log_file=log_file,
            run_config=run_config,
        )
    print(f"DONE. Logs are in {log_dir}")
