from utils import extract_unit_tests, print_log, add_block, deduplicate_and_subsample
from typing import List, Dict, Tuple, Any
from utils.model_api import ModelBase
from .prompts import get_test_system_prompt
from utils.evaluation.eval_utils import MAX_OUTPUT_LEN

class TestGenerator:
    """This class handles the generation of tests using the specified ModelBase."""

    def __init__(self, model: ModelBase) -> None:
        """
       Initializes the TestGenerator with a ModelBase instance.

       Args:
           model: The ModelBase instance to be used for test generation.
       """
        self.model = model

    def _generate(self, 
            system_prompt: str, 
            user_prompt: str,
            max_tokens: int,
            temperature: float,
            attempts: int = 10,
            top_p: float = 0.95
    ) -> Tuple[dict, str, List[str]]:
        """
        Generates test cases based on the provided system prompt, user_prompt, and generation parameters.
        Tracks the prompts and output with a logger.

        Args:
            system_prompt (str): The system prompt that describes the task.
            user_prompt (str): The user prompt that provides the system with the required information.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature for controlling the randomness of the generated text.
            attempts (int, optional): The number of generation attempts if no test cases are retrieved. Defaults to 10.

        Returns:
            Tuple[dict, str, List[str]]: A tuple containing the token counts, the extracted output, and the extracted tests.
        """
        
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        token_counts = {'prompt_tokens': 0, 'completion_tokens': 0}
        for _ in range(attempts):
            gen = self.model.generate_chat(messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            token_counts['prompt_tokens'] += gen['tokens_count']['prompt_tokens']
            token_counts['completion_tokens'] += gen['tokens_count']['completion_tokens']
            trajectory_logprob = gen['trajectory_logprob']
            output = gen['output']
            test_cases = extract_unit_tests(output)
            print_log('generate test cases [system]', system_prompt, 1)
            print_log('generate test cases [user]', user_prompt, 1)
            print_log('generate test cases [assistant]', output, 1)
            print_log('generate test cases [extracted test cases]', '\n\n'.join(test_cases if test_cases is not None else []), 1)
            if test_cases:
                break
        if not test_cases:
            test_cases = [f"ERROR: Failed to extract test cases from\n\n```\n{output}\n```"[:MAX_OUTPUT_LEN]]
            if len(test_cases[0]) == MAX_OUTPUT_LEN:
                test_cases[0] += "\n... [Output Truncated]"
            print_log('test cases [extracted test cases]', test_cases[0], 1)
        return token_counts, trajectory_logprob, output, test_cases        

    def generate_tests(self,
            task_description: str,
            generation_mode: str,
            code: str = '',
            existing_tests: List[str] = [],
            max_existing_tests: int = -1,
            test_feedback: str = '',
            observations: List[str] = [],
            max_tests_per_generation: int = -1,
            max_tokens: int = 8192,
            temperature: float = 1.0,
            top_p: float = 0.95
    ) -> Dict[List[str], Any]:
        """
        Generates test cases.

        Args:
            task_description (str): The task description.
            generation_mode (str): Specifies the system prompt. Must be one of the following:
              - 'initialize': Primes test case generation without any conditioning on current test population or other observations.
              - 'sample_posterior': Primes test case generation by conditioning on observational data (in-context Thompson sampling).
              # - 'extend': Primes additional test case generation, conditioning on existing tests and test feedback (if provided).
            code (str, optional): A code to condition on for the sample_posterior' mode.
            existing_tests (List[str], optional): A list of existing test cases to condition on for the 'extend' mode.
            max_existing_tests (int, optional): The maximum number of existing tests to condition on (via subsampling). If set to -1, all tests are used.
            test_feedback (str, optional): The test feedback to condition on for the 'extend' mode.
            observations (List[str], optional): The observations to condition on for the 'sample_posterior' mode.
            max_tests_per_generation (int, optional): The maximum number of tests to return. If set to -1, all generated tests are returned.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 8192.
            temperature (float, optional): The sampling temperature. Defaults to 1.0.

        Returns:
            dict: a dictionary containing the generated test cases (key `tests`) and the number of tokens used (key `token_counts`).

        Raises:
           NotImplementedError: If an unsupported generation mode is specified.
        """
        existing_tests = "\n\n".join(deduplicate_and_subsample([add_block(test) for test in existing_tests], max_existing_tests)) if existing_tests else None
        observation_string = "\n".join([s.strip() for s in observations]) if observations else None

        system_prompt = get_test_system_prompt(generation_mode+('_with_code' if code else '_without_code'))
        task_prompt = f'''### Task Description\n{task_description}\n\n'''
        history = f'''### Observations from Previous Rounds\n{observation_string}\n\n''' if observation_string else ''
        code = f'''### Candidate Implementation\n{add_block(code)}\n\n''' if code else ''
        test_cases = f'''### Tests\n{existing_tests}\n\n''' if existing_tests else ''
        test_feedback = f'''### Test Feedback for Candidate Implementation\n{test_feedback}\n\n''' if test_feedback else ''


        user_prompt = task_prompt + history + code + test_cases + test_feedback + '### Analysis and Test Suite'

        token_counts, trajectory_logprob, _, test_cases = self._generate(system_prompt, user_prompt, max_tokens, temperature, 10, top_p)
        test_cases = deduplicate_and_subsample(test_cases, max_tests_per_generation)

        return {'tests': test_cases, 'token_counts': token_counts, 'trajectory_logprob': trajectory_logprob}