import numpy as np
import numpy.typing as npt
import pickle
import os
from typing import Annotated

REGRESSION_METHODS = [
    "linear_regression",
    "decision_tree_regression",
    "support_vector_regression",
    "random_forest_regression",
    'lightgbm_regression',
    #'xgboost_regression',
    #'catboost_regression'
]

def _load_pickle(path: Annotated[str, "Path to a .pkl file"]) -> object:
    with open(path, 'rb') as f:
        return pickle.load(f)

def _fit_load(
    path: Annotated[str, "Path to saved models"],
    method: Annotated[str, "Machine learning method"]) -> object:
    return _load_pickle(os.path.join(path, f"{method}_fit_property.pkl"))

def _predict_model(
    x: np.array,
    method: Annotated[str, "Machine learning method"],
    filepath: Annotated[str, "Path to saved models"],
    fit_info: Annotated[bytes, "Serialized model object"],
    **kwargs) -> np.array:
    
    if method not in REGRESSION_METHODS:
        raise ValueError(f"Unsupported method '{method}'. Available: {REGRESSION_METHODS}")
    
    model = _fit_load(filepath, method) if filepath else pickle.loads(fit_info)
    
    return model.predict(x, **kwargs)

def predict(
    x: np.array,
    method: Annotated[str, "Machine learning method"] = "linear_regression",
    filepath: Annotated[str, "Path to saved models"] = ".",
    fit_info: Annotated[bool, "Serialized model object"] = False,
    scalers: Annotated[bool, "Optional tuple of (scaler, scalerp)"] = False,
    **kwargs) -> np.array:
    """Applies preprocessing and runs prediction using a previously trained model :footcite:t:`scikit-learn`.

    Parameters
    ----------
    x : np.array
        Input feature data.
    method : str
        ML method to use for prediction.
    path : str
        Directory with saved model and scalers.
    fit_info : bytes or False
        Serialized model if not using path.
    scalers : tuple or False
        Optional tuple (scaler, scalerp); otherwise loaded from path.
    kwargs : dict
        Extra arguments passed to the model’s `.predict()` method.

    Returns
    -------
    np.array
        Predicted labels (in original label format).
        
    Examples
    --------
    >>> predict(X_predict, method = "linear_regression", path = "./lr_project")
    """
    
    if method == "scalers":
        return 0  # or raise NotImplementedError("Method 'scalers' is a reserved keyword.")

    # Load preprocessing tools
    if not scalers:
        scaler = _load_pickle(os.path.join(filepath, f"{method}_scaler.pkl"))
        scalerp = _load_pickle(os.path.join(filepath, f"{method}_scalerp.pkl"))
    else:
        scaler, scalerp, = scalers

    # Normalize input
    x_norm = pickle.loads(scaler).transform(x)
    x_norm = pickle.loads(scalerp).transform(x_norm)

    return _predict_model(x_norm, method, filepath, fit_info, **kwargs)

# =============================================================== #


# def xgboost_regression(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:
#
#   if path:
#       method = 'xgboost_regression'
#       xgboostregression = fit_load(path, method)
#   else:
#       xgboostregression = pickle.loads(fit_info)
#
#    return xgboostregression.predict(x,**kwargs)


# def catboost_regression(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:

#     if path:
#         method = 'catboost_regression'
#         catregression = fit_load(path, method)
#     else:
#         catregression = pickle.loads(fit_info)
    
#     if '2dy' in catregression:
#         y_pred = []
#         for i in catregression['2dy']:
#             y_pred.append(catregression['2dy'][i].predict(x, **kwargs))
#         return np.array(y_pred).T
#     else:
#         return catregression['1dy'].predict(x, **kwargs)

