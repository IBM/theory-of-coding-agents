from utils.generators import CodeGenerator
from utils import append_jsonl, read_jsonl, create_or_clear_file, init_log, create_dirs
from tqdm import tqdm
from typing import Dict
from utils.running_utils import load_env
import os

def GenCodes(
        index: int,
        code_generator: CodeGenerator,
        data: Dict,
        result_file: str,
        log_file: str,
        run_config: Dict,
) -> None:
    """
    Generates implementations using a code generator and logs the results.

    This function generates a specified number of implementations using the provided
    `code_generator` and configuration parameters. It appends the generated codes
    to a result file in JSONL format and logs progress using a tqdm progress bar.
    If the result or log files do not exist, they are created or cleared.

    Args:
        index (int): Index of the current code generation process, used for display. Should be set to the index of the provided dataset entry.
        code_generator (CodeGenerator): An instance capable of generating implementations.
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

    max_tokens = run_config['max_tokens']
    temperature = run_config['temperature']
    sampling_nums = run_config['rounds']

    prompt = data['prompt']

    # load partial result in case some generations were already conducted
    result = read_jsonl(result_file)
    if len(result) > 0:
        r = result[-1]['r'] + 1
    else:
        create_or_clear_file(log_file)
        create_or_clear_file(result_file)
        r = 0 # round counter

    td = tqdm(initial=r, total=sampling_nums)
    td.set_description(f'''[{index}]''')

    while r < sampling_nums:
        gen = code_generator.generate_code(
            task_description=prompt,
            generation_mode='initialize',
            max_tokens=max_tokens,
            temperature=temperature
        )
        item = {
            'r': r,
            'code': gen['code'],
            'token_counts': gen['token_counts']
        }
        append_jsonl(result_file, item)
        td.update(1)
        r += 1


if __name__ == '__main__':
    env = load_env()

    model = env['model']
    dataset = env['dataset']
    args = env['args']
    config = env['config']
    config_key = env['config_key']
    log_dir = env['log_dir']
    result_dir = env['result_dir']

    assert config_key.startswith('gen_codes'), f'config_key {config_key} does not start with `gen_codes`'

    code_generator = CodeGenerator(model)

    run_config = config[config_key]

    create_dirs(result_dir)
    create_dirs(log_dir)

    for i in dataset.data_range:
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        log_file = os.path.join(log_dir, f'log_{i}.log')
        init_log(log_file)

        GenCodes(
            index=i,
            code_generator=code_generator,
            data=dataset.get_data(i),
            result_file=result_file,
            log_file=log_file,
            run_config=run_config,
        )
    print(f"DONE. Logs are in {log_dir}")