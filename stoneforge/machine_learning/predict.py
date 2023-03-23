import numpy as np
import numpy.typing as npt
import pickle
import warnings

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from autosklearn.classification import AutoSklearnClassifier



def gaussian_naive_bayes(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    naive = pickle.load(open(path+"\\gaussian_naive_bayes_fit_property.pkl", 'rb'))

    return naive.predict(x, **kwargs)


def decision_tree_classifier(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    d_treec = pickle.load(open(path+"\\decision_tree_classifier_fit_property.pkl", 'rb'))

    return d_treec.predict(x, **kwargs)


def support_vector_machine(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    svm = pickle.load(open(path+"\\support_vector_machine_fit_property.pkl", 'rb'))
    
    return svm.predict(x, **kwargs)


def logistic_regression(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    logistic = pickle.load(open(path+"\\logistic_regression_fit_property.pkl", 'rb'))

    return logistic.predict(x,**kwargs)


def k_nearest_neighbors(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    knn = pickle.load(open(path+"\\k_nearest_neighbors_fit_property.pkl", 'rb'))
    
    return knn.predict(x,**kwargs)

def random_florest(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    d_florest = pickle.load(open(path+"\\random_florest_fit_property.pkl", 'rb'))
    
    return d_florest.predict(x,**kwargs)

def xgboost(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    xg = pickle.load(open(path+"\\xgboost_fit_property.pkl", 'rb'))
    
    return xg.predict(x,**kwargs)

def catboost(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    xg = pickle.load(open(path+"\\catboost_fit_property.pkl", 'rb'))
    
    return cb.predict(x,**kwargs)

#def aautoml(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    #auto = pickle.load(open(path+"\\automl_fit_property.pkl", 'rb'))
    
    #return auto.predict(x,**kwargs)





_predict_methods = {
    "GaussianNB": gaussian_naive_bayes,
    "DecisionTreeClassifier": decision_tree_classifier,
    "SVM": support_vector_machine,
    "LogisticRegression": logistic_regression,
    "KNeighborsClassifier": k_nearest_neighbors,
    "RandomForestClassifier": random_florest,
    'XGBClassifier': xgboost,
    'CatBoost': catboost
    #'AutomlClassifier': automl 
    }


def predict(x: npt.ArrayLike, method: str = "GaussianNB", path = ".", **kwargs):

    if method == "GaussianNB":
        fun = _predict_methods[method]
    if method == "DecisionTreeClassifier":
        fun = _predict_methods[method]
    if method == "SVM":
        fun = _predict_methods[method]
    if method == "LogisticRegression":
        fun = _predict_methods[method]
    if method == "KNeighborsClassifier":
        fun = _predict_methods[method]
    if method == "RandomForestClassifier":
        fun= _predict_methods[method]
    if method == "XGBClassifier":
        fun= _predict_methods[method]
    if method == "CatBoost":
        fun= _predict_methods[method]
    #if method == "AutoML":
        #fun = _predict_methods[method]

    return fun(x, path, **kwargs)
