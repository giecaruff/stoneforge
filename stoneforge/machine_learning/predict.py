import numpy as np
import numpy.typing as npt
import pickle
import warnings

#from sklearn.naive_bayes import GaussianNB


def gaussian_naive_bayes(x: npt.ArrayLike, path = "", **kwargs) -> np.ndarray:

    naive = pickle.load(open(path+"gaussian_naive_bayes_fit_property.pkl", 'rb'))

    return naive.predict(x, **kwargs)

def decision_tree_classifier(x: npt.ArrayLike, path = "", **kwargs) -> np.ndarray:

    d_treec = pickle.load(open(path+"decision_tree_classifier_fit_property.pkl", 'rb'))

    return d_treec.predict(x, **kwargs)


_predict_methods = {
    "GaussianNB": gaussian_naive_bayes,
    "DecisionTreeClassifier": decision_tree_classifier
}

def predict(x: npt.ArrayLike, method: str = "GaussianNB", path = "", **kwargs):

    if method == "GaussianNB":
        fun = _predict_methods[method]
    if method == "DecisionTreeClassifier":
        fun = _predict_methods[method]
        
    return fun(x, path, **kwargs)