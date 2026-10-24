from utils import load_config, create_dirs
import os, random, argparse, numpy #, torch
from datasets import CodeDataset
from utils.model_api import ModelBase, model_factory
from typing import *

def read_args(add_args: List[Dict] = []) -> argparse.Namespace:
    """
    Reads and parses command-line arguments.

    Sets up a base set of arguments for running code generation/evaluation
    and allows for additional arguments to be passed dynamically.

    Args:
        add_args: A list of dictionaries, where each dictionary defines an
            additional argument to be added to the parser. Each dict must
            contain a 'name' key and can contain any other keyword arguments
            for `parser.add_argument`. Defaults to an empty list.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_key', type=str, default='sampling', help='The config_key that retrieves the configuration for the desired strategy from the config file.')
    parser.add_argument('--config', type=str, default='configs/config.json', help='The config file that determines the setup for each of the possible strategies.')

    parser.add_argument('--time_limit', type=int, default=180, help='The time limit given to the code evaluator.')

    parser.add_argument('--api_key', type=str, default='NA', help='The OpenAI API key (can be set to NA when using vLLM).')
    parser.add_argument('--base_url', type=str, default='', help='The base URL where the model is hosted (can be set to http://localhost:8000/v1 when using vLLM).')

    parser.add_argument('--selected_ids', nargs='+', type=int, default=[], help='The selected rows from the dataset. Can be used to split the evaluation across multiple nodes.')

    parser.add_argument('--seed', type=int, default=0, help='The random seed for all modules.')

    for a in add_args:
        name = a['name']
        a.pop('name')
        parser.add_argument(name, **a)
    return parser.parse_args()


def load_env(
        config_key: str = '',
        build_config_key: Callable = None,
        add_args: List[Dict] = [],
        requires_model: bool = True
) -> Dict[str, Union[
        argparse.Namespace,
        ModelBase,
        str,
        Dict[str, Any],
        CodeDataset,
        str,
        str
]]:
    """
    Loads the environment, including arguments, configuration, model, and dataset.

    This function orchestrates the setup for a run by parsing arguments,
    loading the configuration file, and loading the dataset. It also sets up
    required output and log directories.

    Args:
        config_key: The config_key that retrieves the configuration for the picked method. This can
            be overridden by the command-line argument `--config_key`.
            Defaults to an empty string.
        build_config_key: A callable function that accepts the parsed
            arguments and returns the run type as a string. Used if
            `config_key` is empty and no `--config_key` is provided.
            Defaults to None.
        add_args: Additional command-line arguments to be parsed.
            See `read_args` for details. Defaults to an empty list.
        requires_model: Whether to establish connection to a model.
            Defaults to True.


    Returns:
        A dictionary containing the initialized environment components:
        - args (`argparse.Namespace`): The parsed command-line arguments.
        - model (`ModelBase` or None): The initialized model object or None.
        - config_key (str): The config_key that retrieves the configuration for the picked method.
        - config (Dict[str, Any]): The loaded configuration dictionary.
        - dataset (`CodeDataset`): The initialized dataset object.
        - result_dir (str): The path to the results directory.
        - log_dir (str): The path to the logs directory.

    Raises:
        AssertionError: If `config_key` cannot be determined (is an empty string).
    """
    args = read_args(add_args)

    random.seed(args.seed)
    numpy.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(args.seed) 

    if args.config_key != '':
        config_key = args.config_key
    elif build_config_key != None:
        config_key = build_config_key(args)
    assert config_key != '', '''config_key can't be none'''

    config = load_config(args.config)

    if requires_model:
        # init_cuda()
        extra_args = {}
        if args.api_key != '':
            extra_args['api_key'] = args.api_key
        if args.base_url != '':
            extra_args['base_url'] = args.base_url
        model = model_factory(**config['model'], **extra_args)
    else:
        model = None

    dataset = CodeDataset(**config['dataset'], selected_ids=args.selected_ids)

    result_dir = os.path.join(config['result_dir'], f'seed-{args.seed}', config_key)
    log_dir = os.path.join(config['log_dir'], f'seed-{args.seed}', config_key)
    create_dirs(result_dir)
    create_dirs(log_dir)

    if args.api_key != '':
        args.api_key = 'xxx'

    return {
        'args': args,
        'model': model,
        'config_key': config_key,
        'config': config,
        'dataset': dataset,
        'result_dir': result_dir,
        'log_dir': log_dir
    }