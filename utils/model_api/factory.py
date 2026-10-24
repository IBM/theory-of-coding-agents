from .models import ModelBase
from .api_models import APIModels

def model_factory(
        model_type: str,
        name: str,
        model_path: str,
        model_system_prompt: str,
        **args
) -> ModelBase:
    """
    Factory function to create and return a model instance based on the specified type.

    This function creates model instances of different types (currently only 'api' is supported)
    by calling the appropriate constructor with the provided arguments.

    Args:
        model_type (str): The type of model to create. Currently supports only 'api' which corresponds to the OpenAI API.
        name (str): The name of the model.
        model_path (str): The path to the model file or directory.
        **args: Additional keyword arguments to pass to the model constructor.

    Returns:
        ModelBase: An instance of the requested model type.

    Raises:
        NotImplementedError: If an unsupported model_type is provided.

    Example:
        >>> model = model_factory('api', 'my_model', '/path/to/model')
    """
    if model_type == 'api':
        model = APIModels(name, model_path, model_system_prompt, **args)
    else:
        raise NotImplementedError

    print(f'{model_type}: {model_path} loaded.', flush=True)
    return model
