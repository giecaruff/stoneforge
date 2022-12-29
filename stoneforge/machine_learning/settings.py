import numpy as np
import numpy.typing as npt
import json
import warnings

def saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)

def settings(method: str = "GaussianNB", path = "", **kwargs):

    if method == "GaussianNB":
        saves(kwargs, path+"\\gaussian_naive_bayes_settings")
    else:
        print("No designed method")
        pass