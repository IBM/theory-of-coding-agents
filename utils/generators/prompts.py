def get_code_system_prompt(method: str) -> str:
    match method:
        case 'initialize_without_tests':
            return CODE_INITIALIZE_SYSTEM_PROMPT
        case 'sample_posterior_with_tests':
            return CODE_SAMPLE_POSTERIOR_WITH_TESTS_SYSTEM_PROMPT
        case 'sample_posterior_without_tests':
            return CODE_SAMPLE_POSTERIOR_WITHOUT_TESTS_SYSTEM_PROMPT
        # case 'repair_with_tests':
        #     return CODE_REPAIR_SYSTEM_PROMPT
        # case 'mutate_without_tests':
        #     return CODE_MUTATE_SYSTEM_PROMPT
        # case 'crossover_without_tests':
        #     return CODE_CROSSOVER_SYSTEM_PROMPT
        case _:
            raise NotImplementedError(f'{method} is not implemented.')

CODE_INITIALIZE_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
Your task is to write an implementation that obeys the provided task description.

**1. Analyze and Plan**
 - Carefully reflect on the task description.
 - Consider different approaches to the problem.
 - Examine the availability of external libraries.

**2. Write the implementation**
 - Include all necessary imports for external libraries.
 - Do not include a docstring.
 - Write the program in a Python code block marked with triple backticks (```).'''

CODE_SAMPLE_POSTERIOR_WITH_TESTS_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
Your task is to write an implementation that obeys the provided task description.
You are provided observations from previous rounds of comparing potential implementations against potential test suites, as well as a novel test suite.

**1. Analyze and Plan**
 - Carefully reflect on the provided observations.
 - Review the provided test suite.
 - Your new implementation must address the observations, pass all test cases, and adhere to the original task description.

**2. Write the implementation**
 - Include all necessary imports for external libraries.
 - Do not include a docstring.
 - Write the program in a Python code block marked with triple backticks (```).'''

CODE_SAMPLE_POSTERIOR_WITHOUT_TESTS_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
Your task is to write an implementation that obeys the provided task description.
You are provided observations from previous rounds of comparing potential implementations against potential test suites.

**1. Analyze and Plan**
 - Carefully reflect on the provided observations.
 - Your new implementation must address the observations and adhere to the original task description.

**2. Write the implementation**
 - Include all necessary imports for external libraries.
 - Do not include a docstring.
 - Write the program in a Python code block marked with triple backticks (```).'''

# CODE_REPAIR_SYSTEM_PROMPT = '''\
# You are an expert Python programmer.
# Your task is to write an implementation that obeys the provided task description.
# You are provided a candidate implementation with feedback from pytests. Repair it.

# **1. Analyze and Plan**
#  - Identify the cause of the errors in the feedback.
#  - Propose changes to resolve the errors.
#  - Be critical in your adjustments: only fix broken code.

# **2. Write the corrected implementation**
#  - Include all necessary imports for external libraries.
#  - Do not include a docstring.
#  - Write the program in a Python code block marked with triple backticks (```).'''

# CODE_MUTATE_SYSTEM_PROMPT = '''\
# You are an expert Python programmer.
# Your task is to write an implementation that obeys the provided task description.
# You are provided a candidate implementation. Mutate it to obtain an alternative solution.

# **1. Analyze and Plan**
#  - Reflect on the core components of the provided implementation.
#  - Propose changes to the implementation.
#  - Ensure the mutation results in a meaningfully different solution.

# **2. Write the mutated implementation**
#  - Include all necessary imports for external libraries.
#  - Do not include a docstring.
#  - Write the program in a Python code block marked with triple backticks (```).'''

# CODE_CROSSOVER_SYSTEM_PROMPT = '''\
# You are an expert Python programmer.
# Your task is to write an implementation that obeys the provided task description.
# You are provided two candidate implementations. Combine them to obtain an alternative solution.

# **1. Analyze and Plan**
#  - Reflect on the core components of the provided implementations.
#  - Identify their similarities and differences.
#  - Combine parts of both implementations to obtain a new solution.

# **2. Write the combined implementation**
#  - Include all necessary imports for external libraries.
#  - Do not include a docstring.
#  - Write the program in a Python code block marked with triple backticks (```).'''

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def get_test_system_prompt(method: str) -> str:
    match method:
        case 'initialize_without_code':
            return TEST_GENERATION_INITIALIZE_SYSTEM_PROMPT
        case 'sample_posterior_without_code':
            return TEST_GENERATION_SAMPLE_POSTERIOR_WITHOUT_CODE_SYSTEM_PROMPT
        case 'sample_posterior_with_code':
            return TEST_GENERATION_SAMPLE_POSTERIOR_WITH_CODE_SYSTEM_PROMPT
        # case 'extend_with_code':
        #     return TEST_GENERATION_EXTEND_SYSTEM_PROMPT
        case _:
            raise NotImplementedError(f'{method} is not implemented.')

TEST_GENERATION_INITIALIZE_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
Your task is to write pytests for an implementation that obeys the provided task description.

**1. Analyze and Plan**
 - Reflect which components of a candidate implementation should be tested.

**2. Write the test suite**
 - Include all necessary imports for external libraries. The candidate implementation is available without imports.
 - Each pytest must be represented by a separate function starting with `test_`. In particular, do not define test classes.
 - Add a brief docstring to each pytest function explaining its purpose.
 - Do not number the pytests.
 - Provide all of the pytests in a single Python code block marked with triple backticks (```).'''

TEST_GENERATION_SAMPLE_POSTERIOR_WITHOUT_CODE_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
Your task is to write pytests for an implementation that obeys the provided task description. 
You are provided observations from previous rounds of comparing potential implementations against potential test suites.

**1. Analyze and Plan**
 - Carefully reflect on the provided observations.
 - Your new test suite must address these observations while still covering the program's main functionality and common edge cases.

**2. Write the test suite**
 - Include all necessary imports for external libraries. The candidate implementation is available without imports.
 - Each pytest must be represented by a separate function starting with `test_`. In particular, do not define test classes.
 - Add a brief docstring to each pytest function explaining its purpose.
 - Do not number the pytests.
 - Provide all of the pytests in a single Python code block marked with triple backticks (```).'''

TEST_GENERATION_SAMPLE_POSTERIOR_WITH_CODE_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
Your task is to write pytests for an implementation that obeys the provided task description. 
You are provided observations from previous rounds of comparing potential implementations against potential test suites, as well as a novel potential implementation.

**1. Analyze and Plan**
 - Carefully reflect on the provided observations.
 - Review the provided implementation.
 - Your new test suite must address the observations while still covering the program's main functionality and common edge cases.

**2. Write the test suite**
 - Include all necessary imports for external libraries. The candidate implementation is available without imports.
 - Each pytest must be represented by a separate function starting with `test_`. In particular, do not define test classes.
 - Add a brief docstring to each pytest function explaining its purpose.
 - Do not number the pytests.
 - Provide all of the pytests in a single Python code block marked with triple backticks (```).'''

# TEST_GENERATION_EXTEND_SYSTEM_PROMPT = '''\
# You are an expert Python programmer.
# Your task is to write pytests for an implementation that obeys the provided task description.
# You are provided previously generated pytests and test feedback for a candidate implementation.

# **1. Analyze and Plan**
#  - Identify potential problems or edge cases that have not been included yet.

# **2. Extend the test suite**
#  - Include all necessary imports for external libraries. The candidate implementation is available without imports.
#  - Each pytest must be represented by a separate function starting with `test_`. In particular, do not define test classes.
#  - Add a brief docstring to each pytest function explaining its purpose.
#  - Do not number the pytests.
#  - Do not repeat existing test cases.
#  - Provide all of the pytests in a single Python code block marked with triple backticks (```).'''

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def get_compression_system_prompt(method: str) -> str:
    match method:
        case 'insights_reformulate':
            return INSIGHTS_REFORMULATE_COMPRESSION_SYSTEM_PROMPT
        case 'insights': # concatenation but with insights instead of summary. Not reported in main paper.
            return INSIGHTS_COMPRESSION_SYSTEM_PROMPT
        case 'summary':
            return SUMMARY_COMPRESSION_SYSTEM_PROMPT
        case _:
            raise NotImplementedError(f'{method} is not implemented.')
        
INSIGHTS_REFORMULATE_COMPRESSION_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You are provided a task description, a candidate program implementation, a candidate test suite with feedback on the implementation, and additional past observations.
Your task is to first analyze the test feedback and then state an exhaustive list of insights that acts as a compressed history of all observations.

**1. Analyze Feedback**
 - Carefully review the program, tests, and feedback. This is your new data.
 - Identify errors in the program implementation, as well as weaknesses in the tests themselves.
 - Summarize all your findings. Be precise and do not speculate.

**2. State Insights**
 - Restate all insights from past observations as well as additional insights from the feedback analysis.
 - Mark each insight with `<observation> </observation>` tags.
 - Ensure each insight is self-contained, i.e., informative without considering the provided program and test suite.'''

INSIGHTS_COMPRESSION_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You are provided a task description, a candidate program implementation, and a candidate test suite with feedback on the implementation.
Your task is to first analyze the test feedback and then state insights.

**1. Analyze Feedback**
 - Carefully review the program, tests, and feedback. This is your new data.
 - Identify errors in the program implementation, as well as weaknesses in the tests themselves.
 - Summarize all your findings. Be precise and do not speculate.

**2. State Insights**
 - Restate all insights from the feedback analysis.
 - Mark each insight with `<observation> </observation>` tags.
 - Ensure each insight is self-contained, i.e., informative without considering the provided program and test suite.'''

SUMMARY_COMPRESSION_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You are provided a task description, a candidate program implementation, and a candidate test suite with feedback on the implementation.
Your task is to first analyze the test feedback and then summarize the implementation, tests, and feedback.

**1. Analyze Feedback**
 - Carefully review the program, tests, and feedback. This is your data.
 - Identify errors in the program implementation, as well as weaknesses in the tests themselves.

**2. State Summary**
 - Summarize the provided implementation, test suite, and all your findings.
 - Mark the summary with `<observation> </observation>` tags.'''