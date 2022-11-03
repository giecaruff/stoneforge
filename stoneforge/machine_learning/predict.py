import numpy as np
import numpy.typing as npt
import pickle
import warnings

from sklearn.naive_bayes import GaussianNB


def gaussian_naive_bayes(x: npt.ArrayLike, path = "", **kwargs) -> np.ndarray:

    naive = pickle.load(open(path+"\\gaussian_naive_bayes_fit_property.pkl", 'rb'))

    return naive.predict(x, **kwargs)


_predict_methods = {
    "GaussianNB": gaussian_naive_bayes
}

def predict(x: npt.ArrayLike, method: str = "GaussianNB", path = "", **kwargs):

    if method == "GaussianNB":
        fun = _predict_methods[method]
        
    return fun(x, path, **kwargs)