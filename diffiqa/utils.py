
import os
import yaml
import random
import importlib
from typing import Tuple

import numpy as np

import torch
from torchvision.transforms import Compose


def seed_all(seed: int) -> None:
    """ Seeds python.random, torch and numpy,

    Args:
        seed (int): Desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(module: str, weights: str) -> torch.nn.Module:
    """ Loads the model architecture using specified module and weights location

    Args:
        module (str): Name of the module and function which loads the model architecture.
        weights (str): Location of the pretrained model weights.

    Returns:
        torch.nn.Module: Pretrained FR model.
    """

    module_name, func_name = module.rsplit(".", 1)

    module = importlib.import_module(module_name)
    importlib.reload(module)

    model_func = getattr(module, func_name)
    model : torch.nn.Module = model_func()

    model.load_state_dict(torch.load(weights))

    return model


def load_config_file(config_loc: str) -> dict:
    """ Constructs an argument class from the configuration located at <config_loc>.

    Args:
        config_loc (str): Location of the configuration file.

    Returns:
        dict: Dictionary of configuration variables. 
    """
    
    assert os.path.exists(config_loc), f" Given config path ({config_loc}) does not exist!"
    with open(config_loc, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def construct_transformation(transformation_arguments: dict) -> Compose:
    """ Constructs a composition of transformations given by the transformation arguments.

    Args:
        transformation_arguments (dict): Arguments of the transformation.

    Returns:
        Compose: Torchvision Compose object of given transformations.
    """

    transforms_list = []
    idx = 1
    while True:
        try:
            trans_args = transformation_arguments[f"trans_{idx}"] #getattr(transformation_arguments, f"trans_{idx}")
            module_name, function_name = trans_args["module"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            trans_function = getattr(module, function_name)
            if "params" in trans_args:
                transforms_list.append(trans_function(**trans_args["params"]))
            else:
                transforms_list.append(trans_function())
        except KeyError:
            break
        idx += 1

    return Compose(transforms_list)


def construct_full_model(config: dict) -> Tuple[torch.nn.Module, Compose]:
    """ Construct a torch.nn.Module model and its transform from the config provided in config_loc.

    Args:
        config_loc (str): Location of the config file containing information about the model.

    Returns:
        Tuple[torch.nn.Module, Compose]: Returns the constructed model and given transformation.
    """

    args = config["fr_model"]

    model = load_model(args["module"], args["weights"])
    trans = construct_transformation(args["transformations"])

    return model, trans


def isimagefile(loc: str, exts=[".png", ".jpg", ".jpeg", ".tiff", ".bmp"]):
    """ Helper function to determine if files are indeed images.

    Args:
        loc (str): Location of file to be checked.
        exts (list, optional): Possible file types for image files. Defaults to [".png", ".jpg", ".jpeg", ".tiff", ".bmp"].

    Returns:
        bool: Returns True if file is image type or False otherwise.
    """
    for ext in exts:
        if loc.endswith(ext):
            return True
    return False


if __name__ == "__main__":

    ...
