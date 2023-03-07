import numpy as np
import numpy.typing as npt
import pickle
import json
import warnings

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

def saves(file, name):
    with open(name + ".pkl", "wb") as write_file:
        pickle.dump(file, write_file)


#Simple Linear Regression
def simple_linear_regression_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs):

    f = open(path + '\\simple_linear_regression_settings.json')

    settings = json.load(f)

    slregression = LinearRegression(**settings)
    slregression.fit(X, y, **kwargs)

    saves(slregression, path+"\\simple_linear_regression_fit_property")
    

_fit_methods = {
    "simple_linear_regression": simple_linear_regression_replacement,
    }

def fit(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "simple_linear_regression", path = ".", **kwargs):


    if method == "simple_linear_regression":
        fun = _fit_methods[method]
    #if method == "DecisionTreeClassifier":
    #    fun = _fit_methods[method]

    
    fun(X, y, path, **kwargs)
