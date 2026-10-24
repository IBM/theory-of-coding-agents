from typing import List, Dict
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, wait_random
from .models import ModelBase

@retry(wait=wait_exponential(multiplier=2, min=10, max=5120)+wait_random(0,10), stop=stop_after_attempt(10))
def gpt_chat(
        client: OpenAI,
        model: str,
        model_system_prompt: str,
        messages: List[Dict],
        tools: List[Dict] = [],
        max_tokens: int = 1024,
        stop_strs: List[str] = [],
        temperature: float = 0.8,
        top_p: float= 0.95
) -> Dict:
    args = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop_strs,
        timeout=3600, # 1h
        logprobs=True,
    )
    if len(tools) > 0:
        args['tools'] = tools
    if model_system_prompt:
        args['messages'] = [{"role": "system", "content": model_system_prompt}] + messages
    #print(args, flush=True)
    response = client.chat.completions.create(**args)
    #print(response, flush=True
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        trajectory_logprob = sum(token.logprob for token in response.choices[0].logprobs.content)
    else:
        trajectory_logprob = None

    return {
        'output': response.choices[0].message.content,
        'message': response.choices[0].message.model_dump(),
        'tokens_count': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        },
        'trajectory_logprob': trajectory_logprob
    }

class APIModels(ModelBase):
    def __init__(self, name: str, model_path: str = 'gpt-3.5-turbo', model_system_prompt: str = "", **args):
        self.name = name
        self.model_path = model_path
        self.client = OpenAI(**args)
        self.model_system_prompt = model_system_prompt

    def generate_chat(
            self,
            messages: List[Dict],
            tools: List[Dict] = [],
            max_tokens: int = 1024,
            stop_strs: List[str] = [],
            temperature: float = 0.8,
            top_p: float= 0.95
    ) -> Dict:
        """
        Generate a chat completion using the OpenAI API.

        This method sends a list of messages to the OpenAI chat completion API and returns
        the response along with token usage information. It includes automatic retry logic
        with exponential backoff in case of API failures.

        Args:
            messages (List[Dict]): List of message dictionaries for the conversation.
            tools (List[Dict], optional): List of tool definitions for function calling.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
            stop_strs (List[str], optional): List of strings that will stop generation.
            temperature (float, optional): Model generation temperature. Defaults to 0.8.
        Returns:
            Dict: A dictionary containing:
                - 'output': The generated text content
                - 'message': The complete message object from the API response
                - 'tokens_count': Dictionary with prompt_tokens and completion_tokens counts
                - 'trajectory_logprob': Float describing the log probability of generating the output sequence

        Raises:
            Exception: Propagates any exceptions from the OpenAI client or retry logic.
        """
        return gpt_chat(
            client=self.client,
            model=self.model_path,
            model_system_prompt=self.model_system_prompt,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            temperature=temperature,
            top_p=top_p
        )
