from utils import extract_observations, print_log, add_block
from typing import List, Tuple
from utils.model_api import ModelBase
import re
from .prompts import get_compression_system_prompt

class ReasoningGenerator:
    """This class handles general reasoning using the specified ModelBase."""

    def __init__(self, model: ModelBase) -> None:
        """
       Initializes the ReasoningGenerator with a ModelBase instance.

       Args:
           model: The ModelBase instance to be used for reasoning.
       """
        self.model = model

    def _generate(self, 
            system_prompt: str, 
            user_prompt: str,
            max_tokens: int,
            temperature: float,
            top_p: float = 0.95
        ) -> Tuple[dict, str]:
        """
        Generates reasoning traces based on the provided system prompt, user_prompt, and generation parameters.
        Tracks the prompts and output with a logger.

        Args:
            system_prompt (str): The system prompt that describes the task.
            user_prompt (str): The user prompt that provides the system with the required information.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature for controlling the randomness of the generated text.

        Returns:
            Tuple[dict, str]: A tuple containing the token counts and the extracted output
        """
        
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        gen = self.model.generate_chat(messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        token_counts = gen['tokens_count']
        output = gen['output']
        print_log('generate reasoning [system]', system_prompt, 1)
        print_log('generate reasoning [user]', user_prompt, 1)
        print_log('generate reasoning [assistant]', output, 1)

        return token_counts, output        

    def compress_observation(self, 
            task_description: str,
            compression_mode: str,
            reformulate: bool,
            code: str, 
            tests:List[str], 
            test_feedback:str, 
            observations:List[str], 
            max_tokens: int = 8192,
            temperature: float = 1.0,
            top_p: float = 0.95):
        """
        Compresses the observation to update the model belief.

        Parameters:
            task_description (str): The task description.
            compression_mode (str): Specifies the system prompt. Must be one of the following:
             - 'insights': Compresses the test feedback into insights.
             - 'summary': Compresses the test feedback into a summary.
             # - 'raw': Does not compress the test feedback at all.
            reformulate (str): Whether to also compress past observations again. If set to false, past `observations` are not added to the context and the new observations are appended.
            code (str): The candidate implementation.
            tests (List[str]): Candidate test cases.
            test_feedback (str): The test feedback.
            observations (str): The past observations, already in compressed form.
            max_tokens (int, optional): The maximum number of tokens per code generation. Defaults to 8192.
            temperature (float, optional): The temperature for controlling the randomness of the generated text. Defaults to 1.0.

        Returns:
            Dict: A dictionary containing all observations thus far (including past observations) and the token count of this round of reasoning. Keys:
                - 'observations'
                - 'tokens_count'
        """
        tests = "\n\n".join([add_block(test) for test in tests])
        observation_string = "\n".join([s.strip() for s in observations]) if observations and reformulate else None
        
        task_prompt = f'''### Task Description\n{task_description}\n\n'''
        code = f'''### Candidate Implementation\n{add_block(code)}\n\n'''
        test_cases = f'''### Tests\n{tests}\n\n'''
        test_feedback = f'''### Test Feedback for Candidate Implementation\n{test_feedback}\n\n'''
        history = f'''### Observations from Previous Rounds\n{observation_string}\n\n''' if observation_string else ''

        if compression_mode == 'raw':
            assert not reformulate, 'raw compression mode cannot be combined with reformulate'
            extracted_observations = [re.sub(r'###', r'#####', code+test_cases+test_feedback)]
            extracted_observations = [f"####Round {len(observations)} Observations:\n" + "\n".join(extracted_observations) + "\n"]
            token_counts = {'prompt_tokens': 0, 'completion_tokens': 0}
        else:
            user_prompt = task_prompt + code + test_cases + test_feedback + history + "### Analysis and Observations"
            system_prompt = get_compression_system_prompt(compression_mode + ('_reformulate' if reformulate else ''))

            token_counts, output = self._generate(system_prompt, user_prompt, max_tokens, temperature, top_p=top_p)

            extracted_observations = extract_observations(output)
            if extracted_observations:
                print_log('generate reasoning [extracted observations]', '\n\n'.join(extracted_observations), 1)
            if not reformulate: # mark each round (since they are independently appended), simplify to one long observation and ensure double spacing in between
                extracted_observations = [f"**Round {len(observations)} Observations:**\n" + "\n".join(extracted_observations) + "\n"]
        extracted_observations = [re.sub(r'\n{2,}', '\n', obs) for obs in extracted_observations] # remove unwanted double spacing (the double spacing in between observation chunks in the absence of reformulation will be added later so it is not filtered)
        total_observations = extracted_observations if reformulate else observations + extracted_observations

        return {'observations': total_observations, 'token_counts': token_counts}