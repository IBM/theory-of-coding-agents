from utils.generators import CodeGenerator, TestGenerator, ReasoningGenerator
from utils.evaluation.code_evaluator import evaluate_code, get_line_cov_feedback
from typing import List, Dict
from utils import read_jsonl, write_jsonl, create_or_clear_file, create_dirs, init_log
from tqdm import tqdm
from utils.running_utils import load_env
import os
from copy import deepcopy

class ThompsonSampling:
    """
    Orchestrates the Thompson Sampling process for iterative code and test generation.

    This class manages the lifecycle of generating code and tests, executing them to
    gather observations (feedback), and using those observations to condition future
    generations. It supports different factorization strategies (code-first, test-first,
    independent) and utilizes separate generators for code, tests, and reasoning.
    """
    def __init__(
        self,
        code_generator: CodeGenerator,
        test_generator: TestGenerator,
        reasoning_generator: ReasoningGenerator,
        run_config: Dict,
        time_limit: float,
    ) -> None:
        """
        Initializes the ThompsonSampling orchestrator.

        Args:
            code_generator (CodeGenerator): Instance responsible for generating code solutions.
            test_generator (TestGenerator): Instance responsible for generating test cases.
            reasoning_generator (ReasoningGenerator): Instance responsible for compressing and reformulating observations.
            run_config (Dict): Configuration dictionary containing settings for rounds, factorization, and generation parameters (max_tokens, temperature) for code, tests, and reasoning.
            time_limit (float): The maximum time allowed for code execution during evaluation.
        """
        self.code_generator = code_generator
        self.test_generator = test_generator
        self.reasoning_generator = reasoning_generator
        self.run_config = run_config
        self.time_limit = time_limit
        self.iterator_rounds = self.run_config['rounds']
        self.code_config = self.run_config['gen_codes']
        self.test_config = self.run_config['gen_tests']
        self.reasoning_config = self.run_config['gen_reasoning']

    def sample_code(self,
                    task_description:str,
                    tests:List[Dict],
                    observation_history:List[Dict],
                    token_counts:Dict):
        """
        Generates a code solution based on the task, existing tests, and history.

        This method extracts observations from the history and invokes the code generator
        to produce a solution ("sample_posterior"). It updates the shared token counts.

        Args:
            task_description (str): The natural language description or prompt for the coding task.
            tests (List[Dict]): A list of dictionaries containing test cases to guide the code generation.
            observation_history (List[Dict]): A list of past observations/feedback from previous rounds.
            token_counts (Dict): A mutable dictionary to track 'prompt_tokens' and 'completion_tokens'.

        Returns:
            Tuple[str, Any]: A tuple containing the generated code string and the trajectory log probability.
        """
        observation_history = [observation['observation'] for observation in observation_history]
        tests = [test['test'] for test in tests]
        gen = self.code_generator.generate_code(task_description, 'sample_posterior', tests, observations=observation_history, max_tokens=self.code_config['max_tokens'], temperature=self.code_config['temperature'])
        token_counts['prompt_tokens'] += gen['token_counts']['prompt_tokens']
        token_counts['completion_tokens'] += gen['token_counts']['completion_tokens']
        return gen['code'], gen['trajectory_logprob']
    
    def sample_tests(self,
                     task_description:str,
                     code:str,
                     observation_history:List[Dict],
                     token_counts:Dict,
                     ground_truth_tests: List):
        """
        Generates test cases or returns ground truth tests if available.

        If `ground_truth_tests` are provided, they are wrapped and returned immediately.
        Otherwise, the test generator is invoked to create new tests based on the task
        description, the current code implementation, and observation history.

        Args:
            task_description (str): The natural language description or prompt for the coding task.
            code (str): The current code implementation to generate tests for.
            observation_history (List[Dict]): A list of past observations/feedback from previous rounds.
            token_counts (Dict): A mutable dictionary to track 'prompt_tokens' and 'completion_tokens'.
            ground_truth_tests (List): A list of pre-defined tests. If not empty, generation is skipped.

        Returns:
            Tuple[List[Dict], Any]: A tuple containing a list of test dictionaries (e.g., [{'test': ...}])
            and the trajectory log probability (or None if ground truth is used).
        """
        if ground_truth_tests:
            return [{'test': test} for test in ground_truth_tests], 0
        observation_history = [observation['observation'] for observation in observation_history]
        gen = self.test_generator.generate_tests(task_description, 'sample_posterior', code, observations=observation_history, max_tokens=self.test_config['max_tokens'], temperature=self.test_config['temperature'])
        token_counts['prompt_tokens'] += gen['token_counts']['prompt_tokens']
        token_counts['completion_tokens'] += gen['token_counts']['completion_tokens']
        return [{'test': test} for test in gen['tests']], gen['trajectory_logprob']

    def get_observations(self,
                         task_description:str,
                         code:str,
                         tests:List[str],
                         observation_history:List[Dict],
                         data_args:Dict,
                         token_counts:Dict):
        """
        Executes code against tests and generates compressed feedback observations.

        This method evaluates the provided code using the tests. It gathers feedback based
        on the configuration (line coverage, execution results, or both) and uses the
        reasoning generator to compress/reformulate this feedback into a concise observation
        for future steps.

        Args:
            task_description (str): The natural language description of the task.
            code (str): The code implementation to evaluate.
            tests (List[str]): A list of test case strings to execute.
            observation_history (List[Dict]): The history of previous observations.
            data_args (Dict): Execution arguments and context required by the evaluator.
            token_counts (Dict): A mutable dictionary to track token usage during reasoning generation.

        Returns:
            List[Dict]: A list of observation dictionaries containing feedback settings and the
            compressed observation text.
        """
        cov = get_line_cov_feedback(code=code, test_cases=tests, data_args=data_args, time_limit=self.time_limit) 
        eval = evaluate_code(code=code, test_cases=tests, data_args=data_args, time_limit=self.time_limit)
        match self.reasoning_config['feedback']:
            case 'line_coverage':
                feedback = cov['feedback']
            case 'tests':
                feedback = "\n\n".join(eval['feedbacks'])
            case 'line_coverage_and_tests':
                feedback = cov['feedback'] + "\n\n\n" + "\n\n".join([feedback['message'] for feedback in eval['feedbacks']])
            case _:
                feedback = ''
        observation_history = [observation['observation'] for observation in observation_history]
        gen = self.reasoning_generator.compress_observation(task_description, self.reasoning_config['compression'], self.reasoning_config["reformulate"], code, tests, feedback, observation_history, max_tokens=self.reasoning_config['max_tokens'], temperature=self.reasoning_config['temperature'])
        items = [{'feedback': self.reasoning_config['feedback'], 'compression': self.reasoning_config['compression'], 'reformulate': self.reasoning_config['reformulate'], 'observation': obs} for obs in gen['observations']]
        token_counts['prompt_tokens'] += gen['token_counts']['prompt_tokens']
        token_counts['completion_tokens'] += gen['token_counts']['completion_tokens']
        return items

    def run(
            self,
            index: int,
            data: Dict,
            result_file: str,
            log_file: str
    ) -> None:
        """
        Executes the main Thompson Sampling loop for a specific dataset entry.

        This function manages the iterative process of generating code and tests for a specific
        problem instance. It handles resumption from partial results, controls the loop for
        the configured number of rounds, applies the factorization strategy (code-first,
        test-first, or independent), and persists results to disk.

        Args:
            index (int): The index of the current dataset entry, used for logging and progress display.
            data (Dict): The dataset entry containing the problem definition.
                Expected keys:
                    - 'prompt' (str): The task description/docstring.
                    - 'data_args' (Dict): Arguments required for code evaluation.
                    - 'tests' (List, optional): Ground truth tests if 'provide_tests' config is True.
            result_file (str): The file path where JSONL results are appended after each round.
            log_file (str): The file path for logging run details.

        Returns:
            None
        """
        task_description = data['prompt']
        data_args = data['data_args']

        # load partial result in case some generations & evaluations were already conducted
        result = read_jsonl(result_file)
        if len(result) > 0:
            observation_history = result[-1]['observation_history'] # history of observations that is used to condition the test generation
            r = result[-1]['r'] + 1 # round counter
        else:
            create_or_clear_file(result_file)
            create_or_clear_file(log_file)
            observation_history = []
            r = 0 # round counter

        td = tqdm(initial=r, total=self.iterator_rounds)
        td.set_description(f'''[{index}]''')

        while r < self.iterator_rounds:
            token_counts = {'prompt_tokens': 0, 'completion_tokens': 0}
            ground_truth_tests = data['tests'] if self.run_config['provide_tests'] else None
            match self.run_config['factorization']:
                case 'code_first':
                    code, trajectory_logprob_code = self.sample_code(task_description, [], observation_history, token_counts)
                    test_list, trajectory_logprob_tests = self.sample_tests(task_description, code, observation_history, token_counts, ground_truth_tests)
                case 'test_first':
                    test_list, trajectory_logprob_tests = self.sample_tests(task_description, "", observation_history, token_counts,  ground_truth_tests)
                    code, trajectory_logprob_code = self.sample_code(task_description, test_list, observation_history, token_counts)
                case 'independent':
                    code, trajectory_logprob_code = self.sample_code(task_description, [], observation_history, token_counts)
                    test_list, trajectory_logprob_tests = self.sample_tests(task_description, "", observation_history, token_counts, ground_truth_tests)

            observation_history = self.get_observations(task_description, code, [test['test'] for test in test_list], observation_history, data_args, token_counts)
            result.append({
                'r': r,
                'observation_history': deepcopy(observation_history),
                'code': code,
                'tests': test_list,
                'token_counts': token_counts,
                'trajectory_logprob': trajectory_logprob_tests + trajectory_logprob_code
                })
            #self.add_scores(result, data_args)
            td.update(1)
            r += 1
            write_jsonl(result_file, result)

if __name__ == '__main__':
    env = load_env()

    model = env['model']
    dataset = env['dataset']
    args = env['args']
    config = env['config']
    config_key = env['config_key']
    result_dir = env['result_dir']
    log_dir = env['log_dir']

    assert config_key.startswith('thompson_sampling'), f'config_key {config_key} does not start with thompson_sampling'

    code_generator = CodeGenerator(model)
    test_generator = TestGenerator(model)
    reasoning_generator = ReasoningGenerator(model)

    run_config = config[config_key]

    create_dirs(result_dir)
    create_dirs(log_dir)

    ts = ThompsonSampling(
        code_generator=code_generator,
        test_generator=test_generator,
        reasoning_generator=reasoning_generator,
        run_config=run_config,
        time_limit=args.time_limit,
    )

    for i in dataset.data_range:
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        log_file = os.path.join(log_dir, f'log_{i}.log')
        init_log(log_file)

        ts.run(
            index=i,
            data=dataset.get_data(i),
            result_file=result_file,
            log_file=log_file,
        )
    print(f"DONE. Logs are in {log_dir}")