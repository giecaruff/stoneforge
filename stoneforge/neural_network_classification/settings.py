import numpy as np
import numpy.typing as npt
import json
import warnings

ML_METHODS = [
    "MLPClassifier"]

def saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)


def settings( path = ".", n_neurons=100, n_layes=1, activation='relu'):
    kwargs = {'hidden_layer_sizes':(n_neurons, n_layes),
               'activation':activation}

    saves(kwargs, path+"\\multi_layer_perceptron__settings")