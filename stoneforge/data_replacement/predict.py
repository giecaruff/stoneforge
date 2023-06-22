import numpy as np
import numpy.typing as npt
import pickle
import warnings
import json

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost.core import CatBoostRegressor


def linear_regression_replacement(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    slregression = pickle.load(open(path+"\\linear_regression_fit_property.pkl", 'rb'))
    f = open(path + '\\polinomial_settings.json')
    pol_settings = json.load(f)

    pol_degree = PolynomialFeatures(degree=pol_settings['degree'])
    x_poly = pol_degree.fit_transform(x)

    return slregression.predict(x_poly , **kwargs)


def support_vector_replacement(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    svnegression = pickle.load(open(path+"\\support_vector_fit_property.pkl", 'rb'))

    return svnegression.predict(x, **kwargs)


def decision_tree_replacement(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    d_treec_regression = pickle.load(open(path+"\\decision_tree_fit_property.pkl", 'rb'))

    return d_treec_regression.predict(x, **kwargs)


def random_florest_replecement(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    randomregression = pickle.load(open(path+"\\random_florest_fit_property.pkl", 'rb'))
    
    return randomregression.predict(x, **kwargs)


def xgboost_replacement(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    xgboostregression = pickle.load(open(path+"\\xgboost_fit_property.pkl", 'rb'))

    return xgboostregression.predict(x,**kwargs)


def lightgbm_replacement(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    lightregression = pickle.load(open(path+"\\lightgbm_fit_property.pkl", 'rb'))
    
    return lightregression.predict(x,**kwargs)

def catboost_replecement(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    catregression = pickle.load(open(path+"\\catboost_fit_property.pkl", 'rb'))
    
    return catregression.predict(x,**kwargs)



_predict_methods = {
    "linear_regression_simple": linear_regression_replacement,
    "linear_regression_polynomial": linear_regression_replacement,
    "suporte_vector_regression": support_vector_replacement,
    "decision_tree_regression": decision_tree_replacement,
    "random_florest_regression": random_florest_replecement,
    "xgboost_regression": xgboost_replacement,
    "lightgbm_regression": lightgbm_replacement,
    "catboost_regression": lightgbm_replacement,
    }


def predict(x: npt.ArrayLike, method: str = "linear_regression_simple", path = ".", **kwargs):

    if method == "linear_regression_simple":
        fun = _predict_methods[method]
    if method == "linear_regression_polynomial":
        fun = _predict_methods[method]
    if method == "suporte_vector_regression":
        fun = _predict_methods[method]
    if method == "decision_tree_regression":
        fun = _predict_methods[method]
    if method == "random_florest_regression":
        fun = _predict_methods[method]
    if method == "xgboost_regression":
        fun = _predict_methods[method]
    if method == "lightgbm_regression":
        fun= _predict_methods[method]
    if method == "catboost_regression":
        fun= _predict_methods[method]

    scaler = pickle.load(open(path+"\\scaler.pkl", 'rb'))
    scalerp = pickle.load(open(path+"\\scalerp.pkl", 'rb'))
    #y_scalerp = pickle.load(open(path+"\\y_scalerp.pkl", 'rb'))
    #scaler.fit(x)

    x_norm = scaler.transform(x)
    x_norm = scalerp.transform(x_norm)

    y = fun(x_norm, path, **kwargs)

    #y_norm = scaler.transform(y)
    #y_norm = scalerp.transform(y_norm)

    return y
