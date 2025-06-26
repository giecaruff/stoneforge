import numpy as np
import numpy.typing as npt
import pickle
import json
import os
from typing import Annotated

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# By Jose Augusto Victorino Dias empirical expertise
MODELS = {
    "gaussian_naive_bayes": (GaussianNB, {}),
    "decision_tree_classifier": (DecisionTreeClassifier, {
        "criterion": ['gini', 'entropy'],
        "max_depth": [5, 10, 15, 30, 50, 70, 100],
    }),
    "support_vector_machine": (SVC, {}),
    "logistic_regression": (LogisticRegression, {
        "random_state": [99]
    }),
    "k_neighbors_classifier": (KNeighborsClassifier, {
        "n_neighbors": np.arange(3, 61, 2),
        "weights": ['uniform', 'distance'],
        "p": np.arange(1, 6)
    }),
    "random_forest_classifier": (RandomForestClassifier, {
        "criterion": ['gini', 'entropy'],
        "max_depth": [5, 10, 15, 30, 50, 70, 100],
        "random_state": [99]
    })
}

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

def _train_model(
    X : Annotated [np.array, "X feature data"],
    y : Annotated [np.array, "y target data"],
    method : Annotated [str, "Machine learning method"],
    filepath : Annotated [str, "Path to the file where the model will be saved"],
    gs : Annotated [bool, "Grid search for hyperparameter tuning"] = False,
    settings : Annotated [bool, "Settings for the model, if not provided will load from file"] = False,
    **kwargs):
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
    if method not in MODELS:
        raise ValueError(f"Method '{method}' is not supported. Available methods: {list(MODELS.keys())}")

    
    ModelClass, param_grid = MODELS[method]

    # Hyperparameter tuning
    if gs:
        model = ModelClass()
        clf = GridSearchCV(model, param_grid, scoring='accuracy')
        clf.fit(X, y)
        settings = clf.best_params_

    # Load or deserialize settings
    elif not settings:
        settings = _load_settings(filepath, method) if filepath else {}
    else:
        settings = pickle.loads(settings)

    model = ModelClass(**settings)
    model.fit(X, y, **kwargs)

    if not filepath:
        return pickle.dumps(model)
    else:
        _saves(model, filepath, method)

def fit(
    X : Annotated [np.array, "X feature data"],
    y : Annotated [np.array, "y target data"],
    method : Annotated [str, "Machine learning method"],
    filepath : Annotated [str, "Path to the file where the model will be saved"],
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
            - 'gaussian_naive_bayes'
            - 'decision_tree_classifier'
            - 'support_vector_machine'
            - 'logistic_regression'
            - 'k_neighbors_classifier'
            - 'random_forest_classifier'
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
    >>> fit(X_fit, y_fit, method="gaussian_naive_bayes", filepath="./nb_project", gs=True)

    Warnings
    --------
    If the `method` is not supported, a ValueError will be raised.
    If `gs` is True, the model will be trained using grid search for hyperparameter.
    Filenames are standardized to include the method name and a suffix "_fit_property.pkl" for consistency.  
    """

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    scalerp = StandardScaler()
    X_norm = scalerp.fit_transform(X_scaled)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Optionally return just the preprocessors
    if method == "scalers":
        return (pickle.dumps(scaler),
                pickle.dumps(scalerp),
                pickle.dumps(le))

    # Save preprocessors
    if filepath:
        _saves(scaler, filepath, f"{method}_scaler", suffix=".pkl")
        _saves(scalerp, filepath, f"{method}_scalerp", suffix=".pkl")
        _saves(le, filepath, f"{method}_y_encoded", suffix=".pkl")

    # Train model
    return _train_model(X_norm, y_encoded, method, filepath, gs, settings, **kwargs)


#XGBOOST
#def xgboost(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs) -> np.ndarray:
#
#    method = "x_g_boost_classifier"
#
#    if gs:
#        parameters =  {'n_estimators': [100],
#        'learning_rate': [0.5],
#        'max_depth':[5,10,15,30,50,70,100],
#        'random_state':[99]}
#
#        xgb = XGBClassifier()
#
#        bestxgb = GridSearchCV(xgb,parameters,scoring='accuracy')
#        bestxgb.fit(X,y)
#        settings = bestxgb.best_params_
#
#    if not gs:
#        if path:
#            settings = load_settings(path, method)
#        else:
#            settings = pickle.loads(settings)
#
#    settings['random_state'] = 99
#    xg = XGBClassifier(**settings)
#    xg.fit(X, y, **kwargs)
#
#    if not path:
#        serialized_model = pickle.dumps(xg)
#        return serialized_model
#    else:
#        saves(xg, path, method)
