from utils.model_api import ModelBase
from utils import extract_longest_block, print_log, add_block, extract_blocks, deduplicate_and_subsample
from typing import Dict, Any, Tuple, List, Union
from .prompts import get_code_system_prompt
from utils.evaluation.eval_utils import MAX_OUTPUT_LEN

class CodeGenerator:
    """This class handles the generation of code using the specified ModelBase."""

    def __init__(self, model: ModelBase) -> None:
        """
        Initialize the CodeGenerator with a specific model.

        Parameters:
            model (ModelBase): The model instance used for code generation.
        """
        self.model = model

    def _generate(self, 
            system_prompt: str, 
            user_prompt: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
            attempts: int = 10,
    ) -> Tuple[dict, str, str]:
        """
        Generates code based on the provided system prompt, user_prompt, and generation parameters.
        Tracks the prompts and output with a logger.

        Args:
            system_prompt (str): The system prompt that describes the task.
            user_prompt (str): The user prompt that provides the system with the required information.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature for controlling the randomness of the generated text.
            attempts (int, optional): The number of generation attempts if no code is retrieved. Defaults to 10.

        Returns:
            Tuple[dict, str, List[str]]: A tuple containing the token counts, the extracted output, and the extracted code.
        """
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        token_counts = {'prompt_tokens': 0, 'completion_tokens': 0}
        for _ in range(attempts):
            gen = self.model.generate_chat(messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            token_counts['prompt_tokens'] += gen['tokens_count']['prompt_tokens']
            token_counts['completion_tokens'] += gen['tokens_count']['completion_tokens']
            trajectory_logprob = gen['trajectory_logprob']
            output = gen['output']
            code = extract_longest_block(output)
            print_log('generate code [system]', system_prompt, 1)
            print_log('generate code [user]', user_prompt, 1)
            print_log('generate code [assistant]', output, 1)
            print_log('generate code [extracted code]', code if code is not None else "", 1)
            if code:
                break
        if not code:
            code = f'ERROR: Failed to extract code from\n\n```\n{output}\n```'[:MAX_OUTPUT_LEN]
            if len(code) == MAX_OUTPUT_LEN:
                code += "\n... [Output Truncated]"
            print_log('code [extracted code]', code, 1)
        gen['tokens_count'] = token_counts
        return token_counts, trajectory_logprob, output, code
    
    def generate_code(self,
            task_description: str,
            generation_mode: str,
            tests:List[str] = [],
            max_tests: int = -1,
            past_code: Union[str, List[str]] = '',
            test_feedback: str = '',
            observations: List[str] = [],
            max_tokens: int = 8192,
            temperature: float = 1.0,
            top_p: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generates code.

        Args:
            task_description (str): The task description.
            generation_mode (str): Specifies the system prompt. Must be one of the following:
              - 'initialize': Primes code generation without any conditioning on test suite, test feedback, or other observations.
              - 'sample_posterior': Primes test case generation by conditioning on observational data (in-context Thompson sampling).
              # - 'repair': Primes code generation for conditioning on prior code, tests, and test feedback.
              # - 'mutate': Primes code generation for conditioning on prior code via mutation.
              # - 'crossover': Primes code generation for conditioning on prior codes via crossover.
            past_code (Union[str, List[str]]), optional): A previous code (or codes) to condition on for the 'repair', 'mutate', and 'crossover' modes.
            tests (List[str], optional): A list of test cases to condition on for the 'repair' and 'sample_posterior' modes.
            max_tests (int, optional): The maximum number of tests to condition on (via subsampling). If set to -1, all tests are used.
            test_feedback (str, optional): The test feedback to condition on for the 'repair' mode.
            observations (List[str], optional): The observations to condition on for the 'sample_posterior' mode.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 8192.
            temperature (float, optional): The sampling temperature. Defaults to 1.0.

        Returns:
            dict: a dictionary containing the generated code (key `code`) and the number of tokens used (key `token_counts`).

        Raises:
           NotImplementedError: If an unsupported generation mode is specified.
        """
        if isinstance(past_code, list):
            n_past_codes = len(past_code)
            past_code = "\n\n".join(add_block(pc) for pc in past_code)
        else:
            past_code = add_block(past_code) if past_code else None
            n_past_codes = 1 # or smaller, used for correct grammar in prompt
        tests = "\n\n".join(deduplicate_and_subsample([add_block(test) for test in tests], max_tests)) if tests else None
        observation_string = "\n".join([s.strip() for s in observations]) if observations else None
        
        system_prompt = get_code_system_prompt(generation_mode+('_with_tests' if tests else '_without_tests'))
        task_prompt = f'''### Task Description\n{task_description}\n\n'''
        history = f'''### Observations from Previous Rounds\n{observation_string}\n\n''' if observation_string else ''
        past_code = f'''### Candidate Implementation{'s' if n_past_codes > 1 else ''}\n{past_code}\n\n''' if past_code else ''
        test_cases = f'''### Tests\n{tests}\n\n''' if tests else ''
        test_feedback = f'''### Test Feedback for Candidate Implementation\n{test_feedback}\n\n''' if test_feedback else ''

        user_prompt = task_prompt + history + past_code + test_cases + test_feedback + '### Analysis and Implementation'

        token_counts, trajectory_logprob, _, code = self._generate(system_prompt, user_prompt, max_tokens, temperature, top_p)

        return {'code': code, 'token_counts': token_counts, 'trajectory_logprob': trajectory_logprob}