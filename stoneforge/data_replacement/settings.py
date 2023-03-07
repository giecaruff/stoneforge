import numpy as np
import numpy.typing as npt
import json
import warnings

def saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)

def settings(method: str = "simple_linear_regression", path = ".", **kwargs):

    if method == "simple_linear_regression":
        saves(kwargs, path+"\\simple_linear_regression_settings")

    if method == "DecisionTreeClassifier":
        saves(kwargs, path+"\\decision_tree_classifier_settings")

