import json, regex, os
from typing import Dict, List, Tuple

env_path_prefix = os.environ.get('MICROMAMBA_ENVS')
if not env_path_prefix:
    raise ValueError("Please set the 'MICROMAMBA_ENVS' environment variable!")

class CodeDataset:
    """
    CodeDataset class for loading and managing dataset files in JSONL format.

    This class is designed to load datasets from JSONL files and provide access to specific data entries.
    It supports filtering based on selected indices and slicing through start and end positions. The class handles
    different types of datasets with specialized processing for each dataset type.

    Attributes:
        name (str): Name of the dataset.
        dataset_file (str): Path to the JSONL file containing the dataset.
        selected_ids (List[int]): List of indices to select from the dataset.
        start (int): Start index for data range.
        end (int): End index for data range.
        data (List[Dict]): Loaded dataset as a list of dictionaries.
        data_range (List[int]): List of valid indices after applying filters.

    Methods:
        __init__: Initializes the dataset by loading data from a JSONL file and applying start, end, and selected indices filters.
        get_data: Retrieves a specific data entry by index with dataset-specific processing.
    """
    data_range: List[int] = []

    def __init__(
            self,
            name: str,
            dataset_file: str,
            selected_ids: List[int] = [],
    ):
        """
        Initializes the CodeDataset by loading data from a JSONL file and applying filters.

        Args:
            name (str): Name of the dataset.
            dataset_file (str): Path to the JSONL file containing the dataset.
            selected_ids (List[int], optional): List of indices to select from the dataset. Defaults to [].
        """
        self.name = name
        print(f'-------------------- load dataset {dataset_file} --------------------', flush=True)
        self.dataset_file = dataset_file
        self.data = []
        with open(dataset_file, 'r') as file:
            line = file.readline()
            while line != '':
                self.data.append(json.loads(line))
                line = file.readline()
        # select which parts of the dataset the currently active process has access to
        self.data_range = [i for i in range(0, len(self.data))]
        if len(selected_ids) == 0:
            print("The full dataset is considered because selected_ids is empty.")
            selected_ids = self.data_range.copy()
        if self.name == 'qiskit_human_eval' or self.name == 'qiskit_human_eval_X': # impossible instances that require real-world quantum computer hardware are removed from dataset
            selected_ids = [i for i in selected_ids if i not in [43, 97, 98, 122, 129, 133, 134, 146]]
        if self.name == 'big_code_bench_hard': # only hard subset is retained
            selected_ids = [i for i in selected_ids if i in [13, 15, 17, 19, 34, 37, 82, 89, 92, 93, 99, 100, 101, 108, 120, 123, 124, 129, 139, 147, 161, 162, 177, 184, 187, 199, 208, 211, 214, 227, 239, 241, 267, 273, 274, 287, 302, 308, 310, 313, 324, 326, 341, 346, 360, 367, 368, 374, 399, 401, 409, 417, 418, 424, 443, 445, 454, 458, 461, 477, 486, 492, 501, 502, 503, 509, 511, 513, 526, 528, 530, 532, 553, 560, 567, 574, 579, 582, 583, 587, 590, 592, 593, 594, 618, 636, 637, 654, 655, 657, 678, 720, 723, 752, 760, 763, 765, 771, 777, 785, 800, 826, 845, 854, 857, 865, 870, 879, 897, 906, 914, 915, 916, 917, 928, 942, 945, 952, 955, 963, 964, 969, 971, 985, 988, 990, 998, 999, 1003, 1004, 1006, 1008, 1012, 1013, 1015, 1019, 1020, 1022, 1039, 1040, 1042, 1053, 1057, 1077, 1085, 1124, 1129, 1137]]
        self.data_range = [i for i in self.data_range if i in selected_ids]

    def get_data(self, i: int) -> Dict:
        """
        Retrieves a specific data entry by index with dataset-specific processing.

        Args:
            i (int): Index of the data entry to retrieve.
        Returns:
            Dict: A dictionary containing the processed data for the specified index. It has the following structure:
                - index
                - prompt
                - solution
                - tests
                - data_args
                    - context_program
                    - context_test_program

        Raises:
            IndexError: If the provided index is out of the valid range.
        """
        d = self.data[i]

        if self.name == 'human_eval':
            return {
                'prompt': f'```python\n{d["prompt"]}\n```', # prompt for code completion
                'solution': d['prompt'] + d['canonical_solution'],
                'tests': [regex.sub('candidate', d['entry_point'], regex.sub(r'def check\(.*', f"def test_{d['entry_point']}():", d['test']))], # single test with multiple asserts
                'data_args': {
                    'context_program': "", # executed before the program
                    'context_test_program': "", # executed before the test case
                    'env': f"{env_path_prefix}/human_eval" # micromamba environment
                } 
            }
        if self.name == 'qiskit_human_eval' or self.name == 'qiskit_human_eval_X':
            return {
                'prompt': f'```python\n{d["prompt"]}\n```', # prompt for code completion
                'solution': d['prompt'] + d['canonical_solution'],
                'tests': [regex.sub('candidate', d['entry_point'], regex.sub(r'def check\(.*', f"def test_{d['entry_point']}():", d['test']))], # single test with multiple asserts
                'data_args': {
                    'context_program': "", # executed before the program
                    'context_test_program': "", # executed before the test case
                    'env': f"{env_path_prefix}/qiskit_human_eval" # micromamba environment
                } 
            }
        
        elif self.name == 'big_code_bench' or self.name == 'big_code_bench_hard':
            return {
                'prompt': f'```python\n{d["complete_prompt"]}\n```', # prompt for code completion
                'solution': d['code_prompt'] + d['canonical_solution'],
                'tests': [d['test']], # entire test class
                'data_args': {
                    'context_program': "", # executed before the program
                    'context_test_program': "import os, time\nos.environ['TZ'] = 'UTC'\ntime.tzset()", # executed before the test case
                    'env': f"{env_path_prefix}/big_code_bench" # micromamba environment
                } 
            }
        
        elif self.name == "mbpp":
            imports_and_signature = regex.search(r'^.*?^(\s*def .+?:)', d["code"], flags=regex.MULTILINE | regex.DOTALL).group(0)
            return {
                'prompt': f'{d["text"]}\nImplementation template:\n```python\n{imports_and_signature}\n```',
                'solution': d["code"], # 
                'tests': [f"def test_case_{i}():\n    {t}" for i, t in enumerate(d["test_list"])], # assert statements converted into pytests
                'data_args': {
                    'context_program': "", # executed before the program
                    'context_test_program': d["test_setup_code"], # executed before the test case
                    'env': f"{env_path_prefix}/mbpp" # micromamba environment
                } 
            }
        elif self.name == "leet_code" or self.name == 'leet_code_minus':
            if self.name == 'leet_code':
                prompt = f'{d["problem_description"]}\nThe implementation may use the following utilities if needed:\n```python\n{d["prompt"]}\n```\nImplementation template:\n```python\n{d["starter_code"]}\n```'
                context_program = d['prompt']
                solution = d["completion"]
            elif self.name == 'leet_code_minus': # without additional external utilities provided <- this is the configuration reported in the paper
                prompt = f'{d["problem_description"]}\nImplementation template:\n```python\n{d["starter_code"]}\n```'
                context_program = ""
                solution = d['prompt'] + "\n" + d["completion"]
            sanitized_entry_point = regex.sub(r'[^0-9a-zA-Z_]', '_', d['entry_point'])
            return {
                'prompt': prompt,
                'solution': solution,
                'tests': [regex.sub('candidate', d['entry_point'], regex.sub(r'def check\(.*', f"def test_{sanitized_entry_point}():", d['test']))],
                'data_args': {
                    'context_program': context_program, # executed before the program
                    'context_test_program': "", # executed before the test case
                    'env': f"{env_path_prefix}/leet_code" # micromamba environment
                } 
            }
        else:
            raise NotImplementedError(f'self.name={self.name}')
