from typing import List
import re, random

def extract_blocks(content: str) -> List[str]:
    """
    Extracts all blocks of text enclosed within triple backticks (```...```) from the given string.
    
    Args:
        content (str): A string that may contain one or more code blocks delimited by triple backticks.
    
    Returns:
        List[str]: A list of strings, each representing the content of a code block found in the input. The backtick markers are excluded from the returned blocks.
    """
    if not content:
        return []
    blocks = []
    if '```' in content:
        lines = content.strip().split('\n')
        i = 0 # line index to step through code
        block = ''
        while i < len(lines):
            while i < len(lines): # search for opening backticks
                if lines[i].strip().startswith('```'):
                    i += 1
                    break
                i += 1
            block = ''
            while i < len(lines): # collect lines until closing backticks
                if lines[i].strip().startswith('```'):
                    i += 1
                    blocks.append(block.strip())
                    block = ''
                    break
                block += lines[i] + '\n'
                i += 1
        if block != '': # edge case where the last block was missing closing backticks
            blocks.append(block.strip())
    return blocks

def extract_longest_block(content: str) -> str:
    """
    Extracts the longest code block from the given content. If no code block is marked, returns None.

    Args: 
        content (str): A string that may contain one or more code blocks delimited by triple backticks.

    Returns:
        str: The content of the longest code block found, with leading and trailing whitespace removed. If no code block is found, returns None."""
    if not content:
        return None
    blocks = extract_blocks(content)
    return max(blocks, key=len) if len(blocks) > 0 else None

def add_block(content: str, lang: str = 'python') -> str:
    """
    Wraps the given content in a code block formatted with triple backticks and an optional language specifier.
    
    Args:
        content (str): The code or text to be enclosed in a code block.
        lang (str): The language identifier for syntax highlighting. Defaults to 'python'.
    
    Returns:
        str: A string containing the content wrapped in a Markdown-style code block.
    """
    return f'''\
```{lang.strip()}
{content.strip()}
```'''

def extract_unit_tests(output: str) -> List[str]:
    """
    Extracts unit test functions from a string output, including all imports.

    The function first extracts the longest code block from the output string and
    then uses a regular expression to find all functions starting with `test_`.
    Moreover, it uses regular expressions to find imports and prepends them to the tests.

    Args:
        output: The string containing the code, potentially wrapped in other text.

    Returns:
        List[str]: All detected unit test function with imports prepended. Each function includes the `def` signature and its body. If no code could be extracted, returns None.
    """
    code = extract_longest_block(output)

    if code:
        import_pattern = r'^\s*(?:import\s+[\w\s,.]+|from\s+[\w.]+\s+import\s+[\w\s,.*]+)\s*$'
        imports = re.findall(import_pattern, code, flags=re.MULTILINE)
        imports_str = '\n'.join(imp.strip() for imp in imports) + '\n\n' if imports else ''

        pattern = r'(^def\s+test_\w+\s*\(.*?\)\s*:.*?)(?=^def|\Z)'
        matches = re.findall(pattern, code, flags=re.MULTILINE | re.DOTALL)
    return [imports_str + match.strip() for match in matches] if code else None

def extract_observations(output: str) -> List[str]:
    """
    Extracts all observations that are delimited by `<observation> </observation>`.

    Args:
        output: The string containing the observation, potentially wrapped in other text.

    Returns:
        List[str]: All detected observation.
    """
    pattern = r"<observation>(.*?)</observation>"
    matches = re.findall(pattern, output, re.DOTALL)
    return [match.strip() for match in matches]

def indent_text(text: str, spaces: int = 4) -> str:
    """Indent each line of a multiline string by a given number of spaces.

    Args:
        text (str): The input multiline string.
        spaces (int, optional): Number of spaces to indent each line. Defaults to 4.

    Returns:
        str: The indented multiline string.
    """
    indentation = ' ' * spaces
    return '\n'.join(f"{indentation}{line}" for line in text.splitlines())

def deduplicate_and_subsample(
        strings: List[str],
        max_strings: int):
    """
    Deduplicates and subsamples a list of strings.

    Args:
        strings (List[str]): A list of strings.
        max_strings (int): The maximum number of strings. If set to -1, all strings are returned.
    Returns:
        List[str]: The deduplicated and randomly subsampled (without replacement) strings.
    """
    strings = list(set(strings))
    if max_strings > -1 and len(strings) > max_strings:
        strings = random.sample(strings, max_strings)
        return strings
    else:
        return strings