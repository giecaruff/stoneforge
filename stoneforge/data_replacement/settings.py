import numpy as np
import numpy.typing as npt
import json
import pickle
import warnings
import os

# TODO: if method is not listed do not execute and gives error to the user

LR_METHODS = [
    "linear_regression_simple",
    "linear_regression_polynomial",
    "decision_tree_regression",
    "support_vector_regression",
    "random_forest_regression",
    'lightgbm_regression',
    'xgboost_regression',
    'catboost_regression'
]

def methods():
    return LR_METHODS

# To remove in further versions
def saves(file, name):
    with open(name+'_settings.json', 'w') as write_file:
        json.dump(file, write_file)

# Will replace json 'saves'
def saves2(file, name):
    with open(name+'_settings.pkl', 'wb') as write_file:
        pickle.dump(file, write_file)

def settings(method: str = "linear_regression_simple", path = ".", verbose = False, **kwargs):
    if verbose:
        print(method)

    if method == "linear_regression_simple" or method  =="linear_regression_polynomial":
        new_settings = {}
        if not 'degree' in kwargs:
            
            saves({'degree':1}, os.path.join(path, 'polinomial'))
            new_settings = kwargs
        else:
            saves({'degree':kwargs['degree']}, os.path.join(path, 'polinomial'))
            new_settings = {}
            for k in kwargs:
                if k != 'degree':
                    new_settings[k] = kwargs[k]
    
        saves(new_settings, os.path.join(path, method))

    if method == "support_vector_regression":
        saves(kwargs, os.path.join(path, method))

    if method == "decision_tree_regression":
        saves(kwargs, os.path.join(path, method))

    if method == "random_forest_regression":
        saves(kwargs, os.path.join(path, method))

    if method == "xgboost_regression":
        saves(kwargs, os.path.join(path, method))

    if method == "lightgbm_regression":
        saves(kwargs, os.path.join(path, method))

    if method == "catboost_regression":
       saves(kwargs, os.path.join(path, method))

    if method == "":
        print("Method not found")
        return methods()

    saves(LR_METHODS, os.path.join(path, 'all_methods'))



