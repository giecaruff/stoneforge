import numpy as np
import numpy.typing as npt
import pickle
import os
from typing import Annotated

ML_METHODS = [
    "gaussian_naive_bayes",
    "decision_tree_classifier",
    "support_vector_machine",
    "logistic_regression",
    "k_neighbors_classifier",
    "random_forest_classifier",
#    "x_g_boost_classifier",
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
    model = _fit_load(filepath, method) if filepath else pickle.loads(fit_info)
    return model.predict(x, **kwargs)

def predict(
    x: np.array,
    method: Annotated[str, "Machine learning method"] = "gaussian_naive_bayes",
    filepath: Annotated[str, "Path to saved models"] = ".",
    fit_info: Annotated[bool, "Serialized model object"] = False,
    scalers: Annotated[bool, "Optional tuple of (scaler, scalerp, label_encoder)"] = False,
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
        Optional tuple (scaler, scalerp, label_encoder); otherwise loaded from path.
    kwargs : dict
        Extra arguments passed to the model’s `.predict()` method.

    Returns
    -------
    np.array
        Predicted labels (in original label format).
        
    Examples
    --------
    >>> predict(X_predict, method = "gaussian_naive_bayes", path = "./nb_project")
    """
    
    if method == "scalers":
        return 0  # or raise NotImplementedError("Method 'scalers' is a reserved keyword.")

    # Load preprocessing tools
    if not scalers:
        scaler = _load_pickle(os.path.join(filepath, f"{method}_scaler.pkl"))
        scalerp = _load_pickle(os.path.join(filepath, f"{method}_scalerp.pkl"))
        le = _load_pickle(os.path.join(filepath, f"{method}_y_encoded.pkl"))
    else:
        scaler, scalerp, le = scalers

    # Normalize input
    x_norm = pickle.loads(scaler).transform(x)
    x_norm = pickle.loads(scalerp).transform(x_norm)

    # Predict
    y_pred = _predict_model(x_norm, method, filepath, fit_info, **kwargs)

    # Decode labels
    return pickle.loads(le).inverse_transform(y_pred)
    
#def xgboost(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:
#    
#    if path:
#        method = 'support_vector_machine'
#        xg= fit_load(path, method)
#    else:
#        xg = pickle.loads(fit_info)
#
#    return xg.predict(x, **kwargs)

#def aautoml(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    #auto = pickle.load(open(path+"\\automl_fit_property.pkl", 'rb'))
    
    #return auto.predict(x,**kwargs)
