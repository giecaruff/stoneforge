import numpy as np
import numpy.typing as npt
import pickle
import json
import warnings

from sklearn.neural_network import MLPClassifier


def saves(file, name):
    with open(name + ".pkl", "wb") as write_file:
        pickle.dump(file, write_file)


#Multi-layer Perceptron
def multi_layer_perceptron(X: npt.ArrayLike, y: npt.ArrayLike, path, gs = False, **kwargs) -> np.ndarray:

    f = open(path + '\\multi_layer_perceptron_settings.json')

    settings = json.load(f)

    mlp = MLPClassifier(**settings)

    mlp.fit(X, y, **kwargs)
    
    saves(mlp, path+"\\multi_layer_perceptron_fit_property")
