import numpy as np
import numpy.typing as npt
import pickle
import json
import warnings


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost.core import CatBoostRegressor


from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

def saves(file, name):
    with open(name + ".pkl", "wb") as write_file:
        pickle.dump(file, write_file)


#Simple Linear Regression
def linear_regression_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs):

    f1 = open(path + '\\linear_regression_settings.json')
    f2 = open(path + '\\polinomial_settings.json')

    settings = json.load(f1)
    pol_settings = json.load(f2)

    pol_degree = PolynomialFeatures(degree=pol_settings['degree'])
    X_poly = pol_degree.fit_transform(X)

    slregression = LinearRegression(**settings)
    slregression.fit(X_poly, y, **kwargs)

    saves(slregression, path+"\\linear_regression_fit_property")
    





def suporte_vector_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs):

    f = open(path + '\\support_vector_settings.json')

    settings = json.load(f)

    svn = SVC(**settings)

    svn.fit(X, y, **kwargs)
    
    saves(svn, path+"\\suporte_vector_fit_property")



def decision_tree_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs):

    f = open(path + '\\decision_tree_settings.json')

    settings = json.load(f)

    decisontree = DecisionTreeClassifier(**settings)

    decisontree.fit(X, y, **kwargs)
    
    saves(decisontree, path+"\\decision_tree_fit_property")


def random_florest_replecement(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs):

    f = open(path + '\\random_florest_settings.json')

    settings = json.load(f)

    random =  RandomForestClassifier(**settings)

    random.fit(X, y, **kwargs)
    
    saves(random, path+"\\random_florest_fit_property")


def xgboost_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs):

    f = open(path + '\\xgboost_settings.json')

    settings = json.load(f)

    xgboost =  XGBClassifier(**settings)

    xgboost.fit(X, y, **kwargs)
    
    saves(xgboost, path+"\\xgboost_fit_property") 


def lightgbm_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs):

    f = open(path + '\\lightgbm_florest_settings.json')

    settings = json.load(f)

    light =  lgb(**settings)

    light.fit(X, y, **kwargs)
    
    saves(light, path+"\\lightgbm_replacement_fit_property")  



def catboost_replecement(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs):

    f = open(path + '\\catboost_florest_settings.json')

    settings = json.load(f)

    cat =  CatBoostRegressor(**settings)

    cat.fit(X, y, **kwargs)
    
    saves(cat, path+"\\catboost_florest_fit_property")


_fit_methods = {
    "linear_regression": linear_regression_replacement,
    "support_vector": suporte_vector_replacement,
    "decisoon_tree": decision_tree_replacement,
    "random_florest": random_florest_replecement,
    "xgboost": xgboost_replacement,
    "light": lightgbm_replacement,
    "cat": catboost_replecement
    }


def fit(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "linear_regression", path = ".", **kwargs):

    if method == "linear_regression":
        fun = _fit_methods[method]
        fun = _fit_methods[method]
    if method == "support_vector":
        fun = _fit_methods[method]
    if method == "decisoon_tree":
        fun = _fit_methods[method]
    if method == "random_florest":
        fun = _fit_methods[method]
    if method == "xgboost":
        fun = _fit_methods[method]
    if method == "light":
        fun = _fit_methods[method]
    if method == "cat":
        fun = _fit_methods[method]

    
    fun(X, y, path, **kwargs)
