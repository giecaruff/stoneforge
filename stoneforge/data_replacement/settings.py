import numpy as np
import numpy.typing as npt
import json
import warnings

# TODO: if method is not listed do not execute and gives error to the user

LR_METHODS = [
    "decision_tree_regression",
    "support_vector_regression",
    "linear_regression_simple",
    "linear_regression_polynomial",
    "random_forest_regression",
    'lightgbm_regression',
]

def methods():
    return LR_METHODS

def saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)

def settings(method: str = "linear_regression_simple", path = ".", **kwargs):

    if method == "linear_regression_simple" or method  =="linear_regression_polynomial":
        new_settings = {}
        if not 'degree' in kwargs:
            saves({'degree':1}, path+"\\polinomial_settings")
            new_settings = kwargs
        else:
            saves({'degree':kwargs['degree']}, path+"\\polinomial_settings")
            new_settings = {}
            for k in kwargs:
                if k != 'degree':
                    new_settings[k] = kwargs[k]
    
        saves(new_settings, path+"\\linear_regression_settings")

    if method == "support_vector_regression":
        saves(kwargs, path+"\\support_vector_settings")

    if method == "decision_tree_regression":
        saves(kwargs, path+"\\decision_tree_regression")

    if method == "random_forest_regression":
        saves(kwargs, path+"\\random_forest_settings")

    if method == "":
        print("Method not found")
        return methods()

    #if method == "xgboost_regression":
    #    saves(kwargs, path+"\\xgboost_regression_settings")

    #if method == "lightgbm_regression":
    #    saves(kwargs, path+"\\lightgbm_settings")

    #if method == "catboost_regression":
    #    saves(kwargs, path+"\\catboost_settings")

    saves(LR_METHODS, path+'\\all_methods')



