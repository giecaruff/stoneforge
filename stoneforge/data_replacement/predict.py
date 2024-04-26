import numpy as np
import numpy.typing as npt
import pickle
import warnings
import json
import os

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost.core import CatBoostRegressor

"linear_regression_simple",
"linear_regression_polynomial",
"decision_tree_regression",
"support_vector_regression",
"random_forest_regression",
'lightgbm_regression',
'xgboost_regression',
'catboost_regression'

def fit_load(path, method):
    full_path = os.path.join(path, method + "_fit_property.pkl")
    with open(full_path, 'rb') as f:
        return pickle.load(f)

def linear_regression(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    method = 'linear_regression_simple'

    slregression = fit_load(path, method)
    f = open(os.path.join(path, 'polinomial_settings.json'))
    pol_settings = json.load(f)

    pol_degree = PolynomialFeatures(degree=pol_settings['degree'])
    x_poly = pol_degree.fit_transform(x)

    return slregression.predict(x_poly , **kwargs)


def support_vector_regression(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    method = 'support_vector_regression'

    svnegression = fit_load(path, method)

    return svnegression.predict(x, **kwargs)


def decision_tree_regression(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    method = 'decision_tree_regression'

    d_treec_regression = fit_load(path, method)

    return d_treec_regression.predict(x, **kwargs)


def random_florest_regression(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    method = 'random_forest_regression'

    randomregression = fit_load(path, method)
    
    return randomregression.predict(x, **kwargs)


def xgboost_regression(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    method = 'xgboost_regression'

    xgboostregression = fit_load(path, method)

    return xgboostregression.predict(x,**kwargs)


def lightgbm_regression(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    method = 'lightgbm_regression'

    lightregression = fit_load(path, method)
    
    return lightregression.predict(x,**kwargs)


def catboost_regression(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    method = 'catboost_regression'

    catregression = fit_load(path, method)
    
    return catregression.predict(x,**kwargs)


_predict_methods = {
    "linear_regression_simple": linear_regression,
    "linear_regression_polynomial": linear_regression,
    "support_vector_regression": support_vector_regression,
    "decision_tree_regression": decision_tree_regression,
    "random_forest_regression": random_florest_regression,
    "xgboost_regression": xgboost_regression,
    "lightgbm_regression": lightgbm_regression,
    "catboost_regression": catboost_regression,
    }


def predict(x: npt.ArrayLike, method: str = "linear_regression_simple", path = ".", **kwargs):

    if method == "linear_regression_simple":
        fun = _predict_methods[method]
    if method == "linear_regression_polynomial":
        fun = _predict_methods[method]
    if method == "support_vector_regression":
        fun = _predict_methods[method]
    if method == "decision_tree_regression":
        fun = _predict_methods[method]
    if method == "random_forest_regression":
        fun = _predict_methods[method]
    if method == "xgboost_regression":
        fun = _predict_methods[method]
    if method == "lightgbm_regression":
        fun= _predict_methods[method]
    if method == "catboost_regression":
        fun= _predict_methods[method]

    scaler = pickle.load(
        open(os.path.join(path,method+"_scaler.pkl"), 'rb'))
    scalerp = pickle.load(
        open(os.path.join(path,method+"_scalerp.pkl"), 'rb'))
    #y_scalerp = pickle.load(open(path+"\\y_scalerp.pkl", 'rb'))
    #scaler.fit(x)

    x_norm = scaler.transform(x)
    x_norm = scalerp.transform(x_norm)

    y = fun(x_norm, path, **kwargs)
    #y = fun(x, path, **kwargs)

    #y_norm = scaler.transform(y)
    #y_norm = scalerp.transform(y_norm)

    return y
