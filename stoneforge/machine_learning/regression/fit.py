import os
import json
import pickle
import numpy as np
import numpy.typing as npt
from typing import Annotated

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Define supported models and their hyperparameter grids
MODELS = {
    "linear_regression": (LinearRegression, {}),
    "decision_tree_regression": (DecisionTreeRegressor, {
        "max_depth": [5, 10, 15, 30, 50, 70, 100],
        "random_state": [99]
    }),
    "support_vector_regression": (SVR, {}),
    "random_forest_regression": (RandomForestRegressor, {
        "n_estimators": [100],
        "max_depth": [10, 30, 50],
        "random_state": [99]
    }),
    "lightgbm_regression": (lgb.LGBMRegressor, {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [5, 10, 30],
        "random_state": [99]
    }),
}

# Save model or any object
def _saves(
    info : Annotated [dict, "Serialized or dictionary settings"],
    filepath : Annotated [str, "Path to the file"],
    name : Annotated [str, "File name"],
    suffix : Annotated [str, "Suffix of the file to be saved"] = "_fit_property.pkl",
    sz : Annotated [str, "Serialized option"]=False)-> None:
    """Saves model file in .json or .pkl format.
    
    Parameters
    ----------
    info : dict
        The settings or model to be saved, either as a dictionary or serialized object.
    filepath : str
        The path where the file will be saved.
    name : str
        The name of the file to save the settings, without extension.
    suffix : str, optional
        The suffix to append to the file name. Defaults to "_fit_property.pkl".
    sz : bool, optional
        If True, the info is serialized and returned as a .pkl format. Defaults to False, which saves the file.
        
    Warnings
    --------
    Don't change the suffix parameter unless you know what you are doing. The other functions recognize this suffix.
    """
    
    full_path = os.path.join(filepath, name + suffix)
    if sz:
        return pickle.dumps(info)
    with open(full_path, "wb") as f:
        pickle.dump(info, f)

# Load settings from JSON
def _load_settings(
    filepath : Annotated [str, "Path to the file"],
    method : Annotated [str, "machine learning method"] = "gaussian_naive_bayes") -> dict:
    """Loads settings from a JSON file for the specified machine learning method.
    
    Parameters
    ----------
    filepath : str
        The path to the directory where the settings file is located.
    method : str, optional
        The name of the machine learning method for which settings are to be loaded.
        
    Returns
    -------
    dict
        A dictionary containing the settings for the specified machine learning method.
    """
    
    with open(os.path.join(filepath, f"{method}_settings.json")) as f:
        return json.load(f)

# Core model training function
def _train_model(
    X : Annotated [np.array, "X feature data"],
    y : Annotated [np.array, "y target data"],
    method : Annotated [str, "Machine learning method"],
    filepath : Annotated [str, "Path to the file where the model will be saved"],
    gs : Annotated [bool, "Grid search for hyperparameter tuning"] = False,
    settings : Annotated [bool, "Settings for the model, if not provided will load from file"] = False,
    **kwargs) -> bytes | None:
    """Internal function that trains a machine learning model based on the specified method and saves the training into file.
    
    Parameters
    ----------
    X : np.array
        Feature data for training the model.
    y : np.array
        Target data for training the model.
    method : str
        The machine learning method to be used for training. Should be one of the following:
    filepath : str
        The path to the file where the trained model will be saved. If not provided, the model will be returned as a serialized object.
    gs : bool, optional
        If True, performs grid search for hyperparameter tuning. Defaults to False.
    settings : bool, optional
        If True, uses the provided settings for the model. If False, loads settings from a file if available.
    **kwargs : dict
        Additional keyword arguments to be passed to the model's fit method.
        
    Returns
    -------
    np.array or None
        If `filepath` is not provided, returns the serialized model as a byte string. Otherwise, saves the model to the specified file and returns None.
        
    Warnings
    --------
    If the `method` is not supported, a ValueError will be raised.
    If `gs` is True, the model will be trained using grid search for hyperparameter.
    Filenames are standardized to include the method name and a suffix "_fit_property.pkl" for consistency.
    """
    ModelClass, param_grid = MODELS[method]
    
    if method not in MODELS:
        raise ValueError(f"Unsupported method '{method}'. Available: {list(MODELS.keys())}")

    # Grid Search if requested
    if gs:
        model = ModelClass()
        grid = GridSearchCV(model, param_grid, scoring="r2")
        grid.fit(X, y)
        settings = grid.best_params_
    elif not settings:
        settings = _load_settings(filepath, method) if filepath else {}
    else:
        settings = pickle.loads(settings)

    # Set default random_state if applicable
    if "random_state" in ModelClass().get_params():
        settings["random_state"] = settings.get("random_state", 99)

    model = ModelClass(**settings)
    model.fit(X, y, **kwargs)

    return _saves(model, filepath, method) if filepath else pickle.dumps(model)

# Main interface
def fit(
    X : Annotated [np.array, "X feature data"],
    y : Annotated [np.array, "y target data"],
    method : Annotated [str, "Machine learning method"] = 'linear_regression',
    filepath : Annotated [str, "Path to the file where the model will be saved"] = ".",
    gs : Annotated [bool, "Grid search for hyperparameter tuning"] = False,
    settings : Annotated [bool, "Settings for the model, if not provided will load from file"] = False,
    **kwargs):
    """Fits a machine learning model to the provided data and saves the model to a file :footcite:t:`scikit-learn, dias2023`.
    
    Parameters
    ----------
    X : np.array
        Feature data for training the model.
    y : np.array
        Target data for training the model.
    method : str
        The machine learning method to be used for training. Should be one of the following:
        - 'linear_regression'
        - 'decision_tree_regression'
        - 'support_vector_regression'
        - 'random_forest_regression'
        - 'lightgbm_regression'
    filepath : str
        The path to the file where the trained model will be saved. If not provided, the model will be returned as a serialized object.
    gs : bool, optional
        If True, performs grid search for hyperparameter tuning. Defaults to False.
    settings : bool, optional
        If True, uses the provided settings for the model. If False, loads settings from a file if available.
    **kwargs : dict
        Additional keyword arguments to be passed to the model's fit method.
        
    Returns
    -------
    np.array or None
        If `filepath` is not provided, returns the serialized model as a byte string. Otherwise, saves the model to the specified file and returns None.
        
    Examples
    --------
    >>> X_fit = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_fit = np.array([0, 1, 0])
    >>> fit(X_fit, y_fit, method="linear_regression", filepath="./lr_project", gs=True)

    Warnings
    --------
    If the `method` is not supported, a ValueError will be raised.
    If `gs` is True, the model will be trained using grid search for hyperparameter.
    Filenames are standardized to include the method name and a suffix "_fit_property.pkl" for consistency.  
    """

    # Preprocessing
    scaler = MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X)

    scalerp = StandardScaler().fit(X_scaled)
    X_norm = scalerp.transform(X_scaled)

    # Shortcut to get preprocessors only
    if method == "scalers":
        return (
            pickle.dumps(scaler),
            pickle.dumps(scalerp),
        )

    # Save preprocessors
    if filepath:
        _saves(scaler, filepath, f"{method}_scaler", suffix=".pkl")
        _saves(scalerp, filepath, f"{method}_scalerp", suffix=".pkl")

    # Train and return or save model
    return _train_model(X_norm, y, method, filepath, gs, settings, **kwargs)


#XgBoost NOTE:(too big, must remain optional)

#def xgboost_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs):
#
#    method = 'xgboost_regression'
#
#    if gs:
#        parameters =  {'n_estimators': [100],
#        'learning_rate': [0.5],
#        'max_depth':[5,10,15,30,50,70,100]}
#
#        xgb = XGBRegressor()
#
#        bestxgbc = GridSearchCV(xgb,parameters,scoring='accuracy')
#        bestxgbc.fit(X,y)
#        _settings = bestxgbc.best_params_
#
#    if not gs:
#        if path:
#            _settings = load_settings(path, method)
#        else:
#            _settings = pickle.loads(settings)
#    
#    xg = XGBRegressor(**_settings)
#    xg.fit(X, y, **kwargs)
#    if not path:
#        serialized_model = pickle.dumps(xg)
#        return serialized_model
#    else:
#        saves(xg, path, method)


#LightGBM


#CatBoost
# def catboost_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs):

#     method = 'catboost_regression'

#     if y.ndim == 2:
#         cb_values = {}
#         cb_values['2dy'] = {}

#         ii = 0
#         for _y in y.T:

#             # =================================================== #

#             if gs:
#                 parameters =  {'n_estimators': [100,150,200],
#                 'learning_rate': [0.3,0.5,0.7],
#                 'max_depth':[5,10,15,30,50,70,100],
#                 'random_state':[99]}

#                 cat = CatBoostRegressor()

#                 bestcatr = GridSearchCV(cat,parameters,scoring='accuracy')
#                 bestcatr.fit(X,_y)
#                 _settings = bestcatr.best_params_

#             if not gs:
#                 if path:
#                     _settings = load_settings(path, method)
#                     _settings['random_state'] = 99
#                     _settings['silent'] = True
#                 else:
#                     _settings = pickle.loads(settings)
#                     _settings['random_state'] = 99
#                     _settings['silent'] = True
            
#             cb = CatBoostRegressor(**_settings)
#             cb.fit(X, _y, **kwargs)
#             ii += 1
#             cb_values['2dy'][ii] = cb

#             # =================================================== #

#     else:

#         # =================================================== #
#         cb_values = {}
#         cb_values['1dy'] = {}
#         if gs:
#             parameters =  {'n_estimators': [100,150,200],
#             'learning_rate': [0.3,0.5,0.7],
#             'max_depth':[5,10,15,30,50,70,100],
#             'random_state':[99]}

#             cat = CatBoostRegressor()

#             bestcatr = GridSearchCV(cat,parameters,scoring='accuracy')
#             bestcatr.fit(X,y)
#             _settings = bestcatr.best_params_

#         if not gs:
#             if path:
#                 _settings = load_settings(path, method)
#                 _settings['random_state'] = 99
#                 _settings['silent'] = True
#             else:
#                 _settings = pickle.loads(settings)
#                 _settings['random_state'] = 99
#                 _settings['silent'] = True
            
#             cb = CatBoostRegressor(**_settings)
#             cb.fit(X, y, **kwargs)
#             cb_values['1dy'] = cb

#             # =================================================== #

#     if not path:
#         serialized_model = pickle.dumps(cb_values)
#         return serialized_model
#     else:
#         saves(cb_values, path, method)

