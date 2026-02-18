import importlib


def load_config(model_name: str):
    module = importlib.import_module(f"configs.{model_name}_config")
    return module.MODEL_CLASS, module.MODEL_CONFIG


def load_model_class(model_name: str):
    module = importlib.import_module(f"configs.{model_name}_config")
    return module.MODEL_CLASS
