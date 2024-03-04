import numpy as np
import numpy.typing as npt
import pickle
import json
import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def saves(file, name):
    with open(name + ".pkl", "wb") as write_file:
        pickle.dump(file, write_file)


def multi_layer_perceptron(X: npt.ArrayLike, y: npt.ArrayLike, path, gs = False, **kwargs) -> np.ndarray:

    if gs:
        parameters = {'hidden_layer_sizes': [(64,),(128,), (64, 64), (128, 64)],
        'activation': ['relu', 'tanh', 'identity'], 
        'solver': ['lbfgs', 'sgd', 'adam'],
        'warm_start':[True, False],
        'random_state':[99]
}

        multilayer = MLPClassifier()
        bestmlp = GridSearchCV(multilayer,parameters,scoring='accuracy')
        bestmlp.fit(X,y)
        settings = bestmlp.best_params_

    if not gs:
        f = open(path + '\\multi_layer_perceptron_settings.json')
        settings = json.load(f)
        if not settings:
            settings['random_state'] = 99
    
    mlpc = MLPClassifier(**settings)
    mlpc.fit(X, y, **kwargs)
    
    saves(mlpc, path+"\\multi_layer_perceptron_fit_property")


_fit_methods = {
    "MLPClassifier": multi_layer_perceptron}

def fit(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "MLPClassifier", path = ".", **kwargs):

    if method == "MLPClassifier":
        fun = _fit_methods[method]

    scaler = StandardScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    saves(scaler, path+"\\scaler")

    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    saves(le, path+"\\LabelEncoded")
        
    fun(X_norm, y_encoded, path, **kwargs)
