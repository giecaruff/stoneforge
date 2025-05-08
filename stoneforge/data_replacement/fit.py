import numpy.typing as npt
import pickle
import json
import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
#from catboost.core import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

LR_METHODS = [
    "linear_regression_simple",
    "linear_regression_polynomial",
    "decision_tree_regression",
    "support_vector_regression",
    "random_forest_regression",
    'lightgbm_regression',
    'xgboost_replacement',
    'catboost_replacement'
]

def saves(file, path, method, suffix = "_fit_property.pkl", sz = False):
    full_path = os.path.join(path, method + suffix)
    if not sz:
        with open(full_path, "wb") as write_file:
            pickle.dump(file, write_file)
    else:
        return pickle.dump(file)

def load_settings(path, method):
    with open(os.path.join(path, method + "_settings.json")) as f:
        return json.load(f)


#Simple Linear Regression
def linear_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs):

    method = 'linear_regression_simple'

    if path:
        overall_settings = load_settings(path, method)
    else:
        overall_settings = pickle.loads(settings)
        #pol_settings = pickle.loads(settings[0])
        #_settings = pickle.loads(settings[1])

    _settings = overall_settings['settings']
    pol_settings = overall_settings['polinomial']


    pol_degree = PolynomialFeatures(degree=pol_settings['degree'])
    X_poly = pol_degree.fit_transform(X)

    slregression = LinearRegression(**_settings)
    slregression.fit(X_poly, y, **kwargs)

    overall_fit_settings = {'serialized_model':slregression, 'polinomial':pol_settings}

    if not path:
        #serialized_model = pickle.dumps(slregression)
        #return settings[0], serialized_model
        return pickle.dumps(overall_fit_settings)
    else:
        saves(slregression, path, method)
    

#Suporte Vector 
def support_vector_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs):

    method = 'support_vector_regression'

    if path:
        settings = load_settings(path, method)
    else:
        settings = pickle.loads(settings)

    svn = SVR(**settings)
    

    if y.ndim == 2:
        svn_values = {}
        svn_values['2dy'] = {}
        n = y.shape[1]

        for i in range(n):
            svn.fit(X, y[:,i], **kwargs)
            svn_values['2dy'][i] = svn

        if not path:
            serialized_model = pickle.dumps(svn_values)
            return serialized_model
        else:
            saves(svn, path, method)
    else:
        svn.fit(X, y, **kwargs)
        svn_values = {}
        svn_values['1dy'] = svn
    if not path:
        serialized_model = pickle.dumps(svn_values)
        return serialized_model
    else:
        saves(svn_values, path, method)


#Decison Tree
def decision_tree_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs):

    method = 'decision_tree_regression'

    if gs:
        parameters = {'criterion': ['gini', 'entropy'],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        decisiontree = DecisionTreeRegressor()

        bestdt = GridSearchCV(decisiontree,parameters,scoring='accuracy')
        bestdt.fit(X,y)
        settings = bestdt.best_params_

    if not gs:
        if path:
            settings = load_settings(path, method)
            settings['random_state'] = 99
        else:
            settings = pickle.loads(settings)
    
    d_treer = DecisionTreeRegressor(**settings)

    d_treer.fit(X, y, **kwargs)
    if not path:
        serialized_model = pickle.dumps(d_treer)
        return serialized_model
    else:
        saves(d_treer, path, method)


#Random Forest
def random_forest_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs):

    method = 'random_forest_regression'

    if gs:
        parameters = {'criterion': ['gini', 'entropy'],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        randomflorest = RandomForestRegressor()

        bestrf = GridSearchCV(randomflorest,parameters,scoring='accuracy')
        bestrf.fit(X,y)
        settings = bestrf.best_params_

    if not gs:
        if path:
            settings = load_settings(path, method)
            settings['random_state'] = 99
        else:
            settings = pickle.loads(settings)
            settings['random_state'] = 99
    
    d_forestc = RandomForestRegressor(**settings)
    d_forestc.fit(X, y, **kwargs)
    if not path:
        serialized_model = pickle.dumps(d_forestc)
        return serialized_model
    else:
        saves(d_forestc, path, method)


#XgBoost
def xgboost_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs):

    method = 'xgboost_regression'

    if gs:
        parameters =  {'n_estimators': [100],
        'learning_rate': [0.5],
        'max_depth':[5,10,15,30,50,70,100]}

        xgb = XGBRegressor()

        bestxgbc = GridSearchCV(xgb,parameters,scoring='accuracy')
        bestxgbc.fit(X,y)
        _settings = bestxgbc.best_params_

    if not gs:
        if path:
            _settings = load_settings(path, method)
        else:
            _settings = pickle.loads(settings)
    
    xg = XGBRegressor(**_settings)
    xg.fit(X, y, **kwargs)
    if not path:
        serialized_model = pickle.dumps(xg)
        return serialized_model
    else:
        saves(xg, path, method)


#LightGBM
def lightgbm_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs):

    method = 'lightgbm_regression'

    if y.ndim == 2:
        lgbm_values = {}
        lgbm_values['2dy'] = {}

        ii = 0
        for _y in y.T:

        # =================================================== #
        # this should be better implemented

            if gs:
                parameters =  {'n_estimators': [100],
                'learning_rate': [0.5],
                'max_depth':[5,10,15,30,50,70,100],
                'random_state':[99]}

                lghtr = lgb.LGBMRegressor()

                bestlight = GridSearchCV(lghtr,parameters,scoring='accuracy')
                bestlight.fit(X,_y)
                _settings = bestlight.best_params_

            if not gs:
                if path:
                    _settings = load_settings(path, method)
                    _settings['random_state'] = 99
                    _settings['verbose'] = -1
                else:
                    _settings = pickle.loads(settings)
                    _settings['random_state'] = 99
                    _settings['verbose'] = -1
            
            lgbm = lgb.LGBMRegressor(**_settings)
            lgbm.fit(X, _y, **kwargs)
            lgbm_values['2dy'][ii] = lgbm
            ii += 1

        # =================================================== #
    else:
        # =================================================== #
        lgbm_values = {}
        lgbm_values['1dy'] = {}

        if gs:
            parameters =  {'n_estimators': [100],
            'learning_rate': [0.5],
            'max_depth':[5,10,15,30,50,70,100],
            'random_state':[99]}

            lghtr = lgb.LGBMRegressor()

            bestlight = GridSearchCV(lghtr,parameters,scoring='accuracy')
            bestlight.fit(X,y)
            settings = bestlight.best_params_

        if not gs:
            if path:
                _settings = load_settings(path, method)
                _settings['random_state'] = 99
                _settings['verbose'] = -1
            else:
                _settings = pickle.loads(settings)
                _settings['random_state'] = 99
                _settings['verbose'] = -1
        
        lgbm = lgb.LGBMRegressor(**_settings)
        lgbm.fit(X, y, **kwargs)
        lgbm_values['1dy'] = lgbm

    if not path:
        serialized_model = pickle.dumps(lgbm_values)
        return serialized_model
    else:
        saves(lgbm_values, path, method)


#CatBoost
def catboost_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs):

    method = 'catboost_regression'

    if y.ndim == 2:
        cb_values = {}
        cb_values['2dy'] = {}

        ii = 0
        for _y in y.T:

            # =================================================== #

            if gs:
                parameters =  {'n_estimators': [100,150,200],
                'learning_rate': [0.3,0.5,0.7],
                'max_depth':[5,10,15,30,50,70,100],
                'random_state':[99]}

                cat = CatBoostRegressor()

                bestcatr = GridSearchCV(cat,parameters,scoring='accuracy')
                bestcatr.fit(X,_y)
                _settings = bestcatr.best_params_

            if not gs:
                if path:
                    _settings = load_settings(path, method)
                    _settings['random_state'] = 99
                    _settings['silent'] = True
                else:
                    _settings = pickle.loads(settings)
                    _settings['random_state'] = 99
                    _settings['silent'] = True
            
            cb = CatBoostRegressor(**_settings)
            cb.fit(X, _y, **kwargs)
            ii += 1
            cb_values['2dy'][ii] = cb

            # =================================================== #

    else:

        # =================================================== #
        cb_values = {}
        cb_values['1dy'] = {}
        if gs:
            parameters =  {'n_estimators': [100,150,200],
            'learning_rate': [0.3,0.5,0.7],
            'max_depth':[5,10,15,30,50,70,100],
            'random_state':[99]}

            cat = CatBoostRegressor()

            bestcatr = GridSearchCV(cat,parameters,scoring='accuracy')
            bestcatr.fit(X,y)
            _settings = bestcatr.best_params_

        if not gs:
            if path:
                _settings = load_settings(path, method)
                _settings['random_state'] = 99
                _settings['silent'] = True
            else:
                _settings = pickle.loads(settings)
                _settings['random_state'] = 99
                _settings['silent'] = True
            
            cb = CatBoostRegressor(**_settings)
            cb.fit(X, y, **kwargs)
            cb_values['1dy'] = cb

            # =================================================== #

    if not path:
        serialized_model = pickle.dumps(cb_values)
        return serialized_model
    else:
        saves(cb_values, path, method)

            


_fit_methods = {
    "linear_regression_simple": linear_regression,
    "linear_regression_polynomial": linear_regression,
    "support_vector_regression": support_vector_regression,
    "decision_tree_regression": decision_tree_regression,
    "random_forest_regression": random_forest_regression,
    "xgboost_regression": xgboost_regression,
    "lightgbm_regression": lightgbm_regression,
    "catboost_regression": catboost_regression
    }


def fit(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "linear_regression_simple", 
path = ".", gs=False, settings = False, **kwargs):

    if method == "linear_regression_simple":
        fun = _fit_methods[method]
    if method == "linear_regression_polynomial":
        fun = _fit_methods[method]
    if method == "support_vector_regression":
        fun = _fit_methods[method]
    if method == "decision_tree_regression":
        fun = _fit_methods[method]
    if method == "random_forest_regression":
        fun = _fit_methods[method]
    if method == "xgboost_regression":
        fun = _fit_methods[method]
    if method == "lightgbm_regression":
        fun = _fit_methods[method]
    if method == "catboost_regression":
        fun = _fit_methods[method]

    # ===================================== #

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    
    #Normalization adjusts the values of a variable to a specific range.

    # ===================================== #

    scalerp = StandardScaler()
    scalerp.fit(X_norm)
    X_norm = scalerp.transform(X_norm)

    if method == "scaler_regression":
        return pickle.dumps(scaler), pickle.dumps(scalerp)

    # ===================================== #
    
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
    
    if not path:
        serialized_model = fun(X_norm, y, path, gs, settings, **kwargs)
        return serialized_model
    else:
        saves(scaler, path, method+'_scaler', suffix = '.pkl')
        saves(scalerp, path, method+'_scalerp', suffix = '.pkl')
        fun(X_norm, y, path, gs, settings, **kwargs)
    #fun(X, y, path, **kwargs)
