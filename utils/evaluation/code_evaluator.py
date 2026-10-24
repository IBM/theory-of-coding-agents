from typing import List, Dict, Union, Any
import os, uuid
from utils import create_dirs, delete_dirs, write_file, read_json
from .eval_utils import add_cov_message, execute_wrapper, coverage

_evaluate_code_cache = dict()
_get_line_cov_cache = dict()

def evaluate_code(
        code: str,
        test_cases: Union[str, List[str]],
        data_args: Dict[str, Any],
        time_limit: float = 60.0,
        use_cache: bool = False,
) -> Dict:
    """
    Evaluates a given code solution against a set of test cases in a pytest environment.

    This function handles the execution and evaluation of a code solution by:
    - Setting up temporary directories for test execution
    - Running pytests sequentially
    - Aggregating results and calculating performance metrics
    It further implements caching, i.e., if enabled, a given code-test case combination is only executed once.

    Args:
        code (str): The main code to be evaluated.
        data_args (Dict[str, Any]): Context provided to program and test case, depending on the task/dataset at hand.
        test_cases (List[str]): List of test cases to run against the code.
        time_limit (float): Maximum time allowed for each test case execution in seconds.
        use_cache (bool): Whether to enable caching of execution (with keys set to concatenation of code and each test case)
    Returns:
        Dict[str, Union[float, List[bool], Dict[int, str]]]: A dictionary containing:
            - 'score': The proportion of passed tests (0.0 to 1.0), considering each `test_case` string as one test even if it contains multiple unit tests.
            - 'passed': Number of successfully passed tests, considering each `test_case` string as one test even if it contains multiple unit tests.
            - 'total': Total number of test cases, considering each `test_case` string as one test even if it contains multiple unit tests.
            - 'feedbacks': List of feedback dictionaries for each test case, considering each `test_case` string as one test even if it contains multiple unit tests.
                    the feedback dictionaries have keys 'id', 'test', 'passed', 'message'
            - 'passed_count': Total number of passed tests, accounting for multiple unit tests per `test_case` string.
            - 'total_count': Total number of tests, accounting for multiple unit tests per `test_case` string.
            - 'status': List of boolean values indicating pass/fail status for each test, considering each `test_case` string as one test even if it contains multiple unit tests.

    Example:
        >>> evaluate_code(
        ...     code="def add(a: Union[int, float], b: Union[int, float]): return a + b",
        ...     test_cases=["def test_add(): assert add(1, 2) == 3"],
        ...     data_args={
        ...         "context_program": "from typing import Union",
        ...         "context_test_program: ""
        ...     },
        ...     time_limit=5.0
        ... )
        {'score': 1.0, 'passed': 1, 'total': 1, 'feedbacks': [], 'passed_count': 1, 'total_count': 1, 'status': [True]}
    """
    # handle edge cases
    if type(test_cases) == str:
        test_cases = [test_cases]
    n_test_cases = len(test_cases)
    if n_test_cases == 0:
        print("[System]: there are no tests provided", flush=True)
        return {
            'score': 0.0,
            'passed': False,
            'invalid': 0,
            'total': 0,
            'feedbacks': [],
            'passed_count': 0,
            'total_count': 0,
            'status': [],
        }
    
    # prepare code
    context_program = data_args['context_program']
    context_test_program = data_args['context_test_program']
    code_total = context_program + '\n\n\n' + code + '\n\n\n' + context_test_program + '\n\n\n'

    run_dir = f'tmp/{uuid.uuid1()}'
    create_dirs(run_dir)
    
    # save code to disk or retrieve result from cache
    new_cache_keys = {} # the new cache keys that should be later added into the cache
    file_names = {} # the file names storing the code of cache misses
    case_dirs = {} # directories where the files are stored
    results = {} # all results, both from the cache and novel executions
    for i in range(n_test_cases):
        code_test_combined = code_total + test_cases[i]
        cache_key = code_test_combined
        if use_cache and cache_key in _evaluate_code_cache.keys():
            results[i] = _evaluate_code_cache[cache_key]
        else:
            file_name = f'test_{i}.py'
            case_dirs[i] = os.path.join(run_dir, f'case_{i}')
            create_dirs(case_dirs[i])
            write_file(os.path.join(case_dirs[i], file_name), code_test_combined)
            file_names[i] = file_name
            new_cache_keys[i] = cache_key

    if len(file_names) > 0:
        for args in [(data_args['env'], i, case_dirs[i], file_names[i], time_limit) for i in file_names.keys()]:
            execute_wrapper(args)
        for i in file_names.keys():
            case_dir = case_dirs[i]
            return_path = os.path.join(case_dir, f'return_{i}.json')
            if os.path.exists(return_path):
                res = read_json(return_path)
                results[res['id']] = res
                if use_cache:
                    _evaluate_code_cache[new_cache_keys[res['id']]] = res

    delete_dirs(run_dir)

    # parse results
    passed = 0
    status = []
    passed_count = 0
    total_count = 0
    invalid_count = 0
    feedbacks = [{
        'id': i,
        'test': test_cases[i]
    } for i in range(n_test_cases)]

    for i in range(n_test_cases):
        if i not in results:
            #raise TimeoutError("Pytest json report failed to generate")
            results[i] = {'id': i, 'passed': False, 'passed_i': 0, 'total_i': 0, 'invalid': 1, 'output': '[System]: Pytest process terminated (SIGTERM). The test exceeded allocated system resources or was externally killed.'}
        result = results[i]
        if result['passed']:
            status.append(True)
            passed += 1
            feedbacks[i]['passed'] = True
        else:
            status.append(False)
            feedbacks[i]['passed'] = False
        feedbacks[i]['message'] = result['output'].strip()

        passed_count += result['passed_i']
        total_count += result['total_i'] 
        invalid_count += 1 if result['invalid'] else 0

    # calculate score
    score = 0 if n_test_cases == 0 else passed / n_test_cases

    ret = {
        'score': score,
        'passed': passed,
        'invalid': invalid_count,
        'total': n_test_cases,
        'feedbacks': feedbacks,
        'passed_count': passed_count, # treats each test of the result separately
        'total_count': total_count, # treats each test of the result separately
        'status': status,
    }

    return ret

def get_line_cov_feedback(
        code: str,
        test_cases: List[str],
        data_args: Dict[str, Any],
        time_limit: int = 60,
        add_block_delimiters: bool = True,
        use_cache: bool = False,
) -> Dict[str, Union[str, float]]:
    """
    Generates line-by-line coverage feedback for a given code snippet by executing test cases.

    This function evaluates the coverage of the provided code using pytest with coverage
    reporting. It calculates which lines are covered and which are not, then returns both
    the overall coverage fraction and a formatted string indicating which lines of the code 
    were not covered. Only a single pytest instance is invoked, with all test_cases concatenated.

    Args:
        code (str): The source code to be evaluated for coverage.
        test_cases (List[str]): A list of strings, each representing a test case.
        data_args (Dict[str, Any]): A dictionary containing context information such as
            'context_program' and 'context_test_program', which are prepended to the code
            during execution.
        time_limit (int): Maximum time allowed for test execution in seconds.
            Defaults to 60.
        add_block_delimiters (boolo): if set to true, returns coverage feedback inside block delimiters ``` ... ```
        use_cache (bool): whether to enable caching of execution (with keys set to concatenation of code and each test case)
    Returns:
        Dict[str, Union[str, float]]: A dictionary containing:
            - "coverage" (float): The fraction of lines covered by tests.
            - "feedback" (str): A formatted string showing each line of code with coverage
              indicators ('+' for covered, '-' for uncovered).
    """
    # handle edge cases
    if type(test_cases) == str:
        test_cases = [test_cases]
    if len(test_cases) == 0:
        return {'coverage': 0, 'feedback': "[System]: There are no tests provided."}

    code = code.strip()
    context_program = data_args['context_program']
    context_test_program = data_args['context_test_program']
    code_total = context_program + '\n\n\n' + code
    test_total = context_test_program + 'from solution import *' + '\n\n\n' + '\n\n'.join(test_cases)
    start_line = context_program.count('\n') + 4

    cache_key = code_total + '\n\n\n' + test_total
    if use_cache and cache_key in _get_line_cov_cache.keys():
        result = _get_line_cov_cache[cache_key]
    else:
        try:
            run_dir = f'tmp/{uuid.uuid1()}'
            create_dirs(run_dir)
            write_file(os.path.join(run_dir, 'solution.py'), code_total)
            write_file(os.path.join(run_dir, 'test_solution.py'), test_total)
            result = coverage(data_args['env'], id=-1, run_dir=run_dir, file_name='solution.py', time_limit=time_limit)
            delete_dirs(run_dir)
            if use_cache:
                _get_line_cov_cache[cache_key] = result
        except:
            return {
                'coverage': 0, 
                'feedback': "[System]: Line Coverage Report Failed. This may be caused by syntax errors in the candidate code or tests."
            } 

    if result['missing_lines'] is None: # Coverage report not found or failed to generate. Just returns the code without annotation.
        return {
            'coverage': 0, 
            'feedback': "[System]: Line Coverage Report Failed. This may be caused by syntax errors in the candidate code or tests."
        } 

    missing_lines = [line-start_line for line in result['missing_lines']]
    covered_lines = [line-start_line for line in result['covered_lines']]
    all_lines = missing_lines + covered_lines

    coverage_fraction = len(covered_lines) / len(all_lines) if len(all_lines) > 0 else 0
    feedback = add_cov_message(code, covered_lines, missing_lines, add_block_delimiters=add_block_delimiters)

    return {
        'coverage': coverage_fraction,
        'feedback': feedback
    }