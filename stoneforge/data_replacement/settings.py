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

def settings(method: str = "linear_regression_simple", path = ".", verbose = False, **kwargs):
    if verbose:
        print(method)

    if method == "linear_regression_simple" or method  =="linear_regression_polynomial":
        new_settings = {}
        overall_settings = {}
        if not 'degree' in kwargs:
            
            #if path:
            #    saves({'degree':1}, os.path.join(path, 'polinomial'))
            #else:
            #    _polinomial = pickle.dumps({'degree':1})
            new_settings = kwargs
            overall_settings['polinomial'] = {'degree':1}
        else:
            #if path:
            #    saves({'degree':kwargs['degree']}, os.path.join(path, 'polinomial'))
            #else:
            #    _polinomial = pickle.dumps({'degree':kwargs['degree']})
            overall_settings['polinomial'] = {'degree':kwargs['degree']}            
            new_settings = {}
            for k in kwargs:
                if k != 'degree':
                    new_settings[k] = kwargs[k]
        overall_settings['settings'] = new_settings
        if path:
            saves(overall_settings, os.path.join(path, method))
        else:
            return pickle.dumps(overall_settings)

    if method == "support_vector_regression":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "decision_tree_regression":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "random_forest_regression":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "xgboost_regression":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "lightgbm_regression":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "catboost_regression":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)
    
    #if not serialize:
    #    saves(LR_METHODS, os.path.join(path, 'all_methods'))

    else:
        return methods()



