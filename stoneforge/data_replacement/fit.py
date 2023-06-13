import numpy as np
import numpy.typing as npt
import pickle
import json
import warnings


from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as LGBMRegressor
from catboost.core import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler



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
    

#Suporte Vector 
def suporte_vector_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs):

    f = open(path + '\\support_vector_settings.json')

    settings = json.load(f)
    if not settings:
        settings['random_state'] = 99

    svn = SVC(**settings)

    svn.fit(X, y, **kwargs)
    
    saves(svn, path+"\\suporte_vector_fit_property")



#Decison Tree
def decision_tree_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path,gs=True, **kwargs):

    if gs:
        parameters = {'criterion': ['gini', 'entropy'],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        decisiontree = DecisionTreeRegressor()

        bestdt = GridSearchCV(decisiontree,parameters,scoring='accuracy')
        bestdt.fit(X,y)
        settings = bestdt.best_params_

    if not gs:
        f = open(path + '\\decision_tree_replacement.json')
        settings = json.load(f)
        if not settings:
            settings['random_state'] = 99
    
    d_treer = DecisionTreeRegressor(**settings)
    d_treer.fit(X, y, **kwargs)
    
    saves(d_treer, path+"\\decision_tree_fit_property")


#Random Florest
def random_florest_replecement(X: npt.ArrayLike, y: npt.ArrayLike, path, gs=True,**kwargs):

    if gs:
        parameters = {'criterion': ['gini', 'entropy'],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        randomflorest = RandomForestRegressor()

        bestrf = GridSearchCV(randomflorest,parameters,scoring='accuracy')
        bestrf.fit(X,y)
        settings = bestrf.best_params_

    if not gs:
        f = open(path + '\\random_florest_settings.json')
        setting = json.load(f)
        if not settings:
            settings['random_state'] = 99
    
    d_florestc = RandomForestRegressor(**settings)
    d_florestc.fit(X, y, **kwargs)
    
    saves(d_florestc, path+"\\random_florest_fit_property")


#XgBoost
def xgboost_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path,gs=False, **kwargs) -> np.ndarray:

    if gs:
        parameters =  {'n_estimators': [100],
        'learning_rate': [0.5],
        'max_depth':[5,10,15,30,50,70,100]}

        xgb = XGBRegressor()

        bestxgbc = GridSearchCV(xgb,parameters,scoring='accuracy')
        bestxgbc.fit(X,y)
        #bestxgb = bestxgb.best_params_
        settings = bestxgbc.best_params_

    if not gs:
        f = open(path + '\\xgboost_regression_settings.json')
        settings = json.load(f)
    
    xg = XGBRegressor(**settings)
    xg.fit(X, y, **kwargs)

    saves(xg, path+"\\xgboost_fit_property")

#LightGBM
def lightgbm_replacement(X: npt.ArrayLike, y: npt.ArrayLike, path, gs=False, **kwargs):

    if gs:
        parameters =  {'n_estimators': [100],
        'learning_rate': [0.5],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        lghtr = LGBMRegressor()

        bestlight = GridSearchCV(lghtr,parameters,scoring='accuracy')
        bestlight.fit(X,y)
        #bestxgb = bestxgb.best_params_
        settings = bestlight.best_params_

    if not gs:
        f = open(path + '\\lightgbm_florest_settings.json')
        settings = json.load(f)
        if not settings:
            settings['random_state'] = 99
    
    xg = LGBMRegressor(**settings)
    xg.fit(X, y, **kwargs)

    saves(xg, path+"\\lightgbm_replacement_fit_property")

#CatBoost
def catboost_replecement(X: npt.ArrayLike, y: npt.ArrayLike, path,gs=False, **kwargs) -> np.ndarray:

    if gs:
        parameters =  {'n_estimators': [100,150,200],
        'learning_rate': [0.3,0.5,0.7],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        cat = CatBoostRegressor()

        bestcatr = GridSearchCV(cat,parameters,scoring='accuracy')
        bestcatr.fit(X,y)
        #bestcat = bestcat.best_params_
        settings = bestcatr.best_params_

    if not gs:
        f = open(path + '\\catboost_settings.json')
        settings = json.load(f)
        if not settings:
            settings['random_state'] = 99
    
    cb = CatBoostRegressor(**settings)
    cb.fit(X, y, **kwargs)

    saves(cb, path+"\\catboost_fit_property")



_fit_methods = {
    "linear_regression": linear_regression_replacement,
    "linear_regression": linear_regression_replacement,
    "suporte_vector_regression": suporte_vector_replacement,
    "decision_tree_regression": decision_tree_replacement,
    "random_florest_regression": random_florest_replecement,
    "xgboost_regression": xgboost_replacement,
    "lightgbm_regression": lightgbm_replacement,
    "catboost_regression": catboost_replecement
    }


def fit(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "linear_regression", path = ".", **kwargs):

    if method == "linear_regression_simple":
        fun = _fit_methods[method]
    if method == "linear_regression_polynomial":
        fun = _fit_methods[method]
    if method == "suporte_vector_regression":
        fun = _fit_methods[method]
    if method == "decision_tree_regression":
        fun = _fit_methods[method]
    if method == "random_florest_regression":
        fun = _fit_methods[method]
    if method == "xgboost_regression":
        fun = _fit_methods[method]
    if method == "lightgbm_regression":
        fun = _fit_methods[method]
    if method == "catboost_regression":
        fun = _fit_methods[method]


    scaler = MinMaxScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    saves(scaler, path+"\\scaler")
    
    #Normalization adjusts the values of a variable to a specific range.

    scalerp = StandardScaler()
    scalerp.fit(X_norm)
    X_norm = scaler.transform(X_norm)
    saves(scalerp, path+"\\scalerp")
    
    #Standardization transforms the data in such a way that it has a mean of zero and a standard deviation of one.

    # TODO: repass to preprocessing due to the sobreposition of processes

    #scaler = MinMaxScaler()
    #scaler.fit(y)
    #y_norm = scaler.transform(y)
    #saves(scaler, path+"\\y_scaler")
        
    #scalerp = StandardScaler()
    #scalerp.fit(y_norm)
    #y_norm = scaler.transform(y_norm)
    #saves(scalerp, path+"\\y_scalerp")
    

    fun(X_norm, y, path, **kwargs)
