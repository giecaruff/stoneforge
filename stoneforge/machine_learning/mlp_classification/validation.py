import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


def saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)

def multi_layer_perceptron(X: npt.ArrayLike, y: npt.ArrayLike, path, n_splits, random_state, **kwargs) -> np.ndarray:

    f = open(path + '\\multi_layer_perceptron.json')

    settings = json.load(f)

    naive = MLPClassifier(**settings)

    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = random_state)

    result = cross_val_score(naive, X, y, cv = kfold)
    mean_result = {}
    mean_result['mean_accuracy'] = round(result.mean() * 100,3)
    if path:
        saves(mean_result,path + '\\mean_accuracy')
    if not path:
        return mean_result


_fit_methods = {
    "MLPClassifier": multi_layer_perceptron}


def validation(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "MLPClassifier", path = ".", n_splits = 30, random_state = 5, **kwargs):

    X_norm = StandardScaler().fit_transform(X)

    if method == "MLPClassifier":
        fun = _fit_methods[method]

    X_norm = StandardScaler().fit_transform(X)
        
    fun(X_norm, y, path, n_splits, random_state, **kwargs)


