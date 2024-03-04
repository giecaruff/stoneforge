import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from . import fit
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


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
    mean_result['mean_absolute_percentage_error'] = round(result.mean() * 100,3)
    if path:
        saves(mean_result,path + '\\mean_absolute_percentage_error')
    if not path:
        return mean_result


_fit_methods = {
    "MLPRegressor": multi_layer_perceptron}


def validation(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "MLPRegressor", path = ".", n_splits = 30, random_state = 5, **kwargs):

    X_norm = StandardScaler().fit_transform(X)

    if method == "MLPClassifier":
        fun = _fit_methods[method]

    X_norm = StandardScaler().fit_transform(X)
        
    fun(X_norm, y, path, n_splits, random_state, **kwargs)


