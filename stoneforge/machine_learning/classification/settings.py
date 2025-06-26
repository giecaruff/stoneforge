import json
import pickle
import os
from typing import Annotated

ML_METHODS = [
    "gaussian_naive_bayes",
    "decision_tree_classifier",
    "support_vector_machine",
    "logistic_regression",
    "k_neighbors_classifier",
    "random_forest_classifier",
#    "x_g_boost_classifier",
]

def _methods():
    """Returns the list of available machine learning methods.

    Returns
    -------
    ML_METHODS : list of str
        A list of strings representing the names of available machine learning methods.
    """
    return ML_METHODS

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
    method: Annotated[str, "Machine learning method"] = "gaussian_naive_bayes",
    filepath: Annotated[str, "Path to the file where settings will be saved"] = ".", 
    **kwargs):
    """This saves the settings or hyperparameters for a machine learning method into the machine.
    The idea is to be reusable for distinct data.

    Parameters
    ----------
    method : str, optional
        Name of the machine learning method to be used. Should be one of the following:
            - 'gaussian_naive_bayes'
            - 'decision_tree_classifier'
            - 'support_vector_machine'
            - 'logistic_regression'
            - 'k_neighbors_classifier'
            - 'random_forest_classifier'
            
    filepath : str, optional
        Path to the file where settings will be saved. If not provided, it defaults to the current directory (".").
        if False or None, it will return the settings as a serialized object in the current directory (".").
        
    Example
    -------
    >>> settings(method="gaussian_naive_bayes", filepath="./nb_project", var_smoothing=1e-5)
        
    Warnings
    --------
    This function is designed to save settings for specific machine learning methods. If the method is not supported,
    """
    
    methods_requiring_save = {
        "gaussian_naive_bayes",
        "decision_tree_classifier",
        "support_vector_machine",
        "logistic_regression",
        "k_neighbors_classifier",
        "random_forest_classifier",
        # "x_g_boost_classifier",  # Uncomment when needed
    }

    if method not in methods_requiring_save:
        return _methods()  # fallback for unsupported methods

    if method == "logistic_regression":
        kwargs['solver'] = 'liblinear'  # due to scikit-learn limitation

    if filepath:
        _json_saves(kwargs, os.path.join(filepath, method))
    else:
        return pickle.dumps(kwargs)
