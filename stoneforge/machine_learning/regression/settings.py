import json
import pickle
import os
from typing import Annotated

LR_METHODS = [
    "linear_regression",
    "decision_tree_regression",
    "support_vector_regression",
    "random_forest_regression",
    "lightgbm_regression",
    # "xgboost_regression",
    # "catboost_regression"
]


def _methods():
    """Returns the list of available machine learning methods.

    Returns
    -------
    ML_METHODS : list of str
        A list of strings representing the names of available machine learning methods.
    """
    return LR_METHODS


# NOTE: pass this saving operations to the data manager in the future
def _json_saves(
    filepath : Annotated[str, "Path to the file where settings will be saved"],
    name: Annotated[str, "Name of the file to save the settings"]) -> None:
    """Saves the machine learning settings or hyperparameters into a JSON file.'

    Parameters
    ----------
    file : str
        The filepath where the settings will be saved.
    name : str
        The name of the file to save the settings, without extension.
    """
    with open(f'{name}_settings.json', 'w') as write_file:
        json.dump(filepath, write_file)


def settings(
    method: Annotated[str, "Machine learning method"] = "linear_regression",
    filepath: Annotated[str, "Path to the file where settings will be saved"] = ".", 
    **kwargs):
    """This saves the settings or hyperparameters for a machine learning method into the machine.
    The idea is to be reusable for distinct data.

    Parameters
    ----------
    method : str, optional
        Name of the machine learning method to be used. Should be one of the following:
            - 'linear_regression'
            - 'decision_tree_regression'
            - 'support_vector_regression'
            - 'random_forest_regression'
            - 'lightgbm_regression'
            
    filepath : str, optional
        Path to the file where settings will be saved. If not provided, it defaults to the current directory (".").
        if False or None, it will return the settings as a serialized object in the current directory (".").
        
    Example
    -------
    >>> settings(method="linear_regression", filepath="./lr_project")
        
    Warnings
    --------
    This function is designed to save settings for specific machine learning methods. If the method is not supported,
    """
    
    methods_requiring_save = {
        "linear_regression",
        "decision_tree_regression",
        "support_vector_regression",
        "random_forest_regression",
        "lightgbm_regression",
        # "xgboost_regression",
        # "catboost_regression"
    }

    if method not in methods_requiring_save:
        return _methods()  # fallback for unsupported methods

    if filepath:
        _json_saves(kwargs, os.path.join(filepath, method))
    else:
        return pickle.dumps(kwargs)