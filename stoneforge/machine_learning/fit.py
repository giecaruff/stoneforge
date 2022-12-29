import numpy as np
import numpy.typing as npt
import pickle
import json
import warnings

from sklearn.naive_bayes import GaussianNB

def saves(file, name):
    with open(name + ".pkl", "wb") as write_file:
        pickle.dump(file, write_file)


def gaussian_naive_bayes(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    f = open(path + '\\gaussian_naive_bayes_settings.json')

    settings = json.load(f)

    naive = GaussianNB(**settings)

    naive.fit(X, y, **kwargs)
    
    saves(naive, path+"\\gaussian_naive_bayes_fit_property")


_fit_methods = {
    "GaussianNB": gaussian_naive_bayes
}

def fit(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "GaussianNB", path = "", **kwargs):


    if method == "GaussianNB":
        fun = _fit_methods[method]
    else:
        print("Indicates the machine learning method")
        
    fun(X, y, path, **kwargs)