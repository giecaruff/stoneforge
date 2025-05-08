import numpy as np
import numpy.typing as npt
import pickle
import os


def fit_load(path, method):
    full_path = os.path.join(path, method + "_fit_property.pkl")
    with open(full_path, 'rb') as f:
        return pickle.load(f)
    
ML_METHODS = [
    "gaussian_naive_bayes",
    "decision_tree_classifier",
    "support_vector_machine",
    "logistic_regression",
    "k_neighbors_classifier",
    "random_forest_classifier",
    "x_g_boost_classifier",
]

def gaussian_naive_bayes(x: npt.ArrayLike, path, fit_info, **kwargs) -> np.ndarray:

    if path:
        method = 'gaussian_naive_bayes'
        naive = fit_load(path, method)
    else:
        naive = pickle.loads(fit_info)

    return naive.predict(x, **kwargs)


def decision_tree_classifier(x: npt.ArrayLike, path, fit_info, **kwargs) -> np.ndarray:

    if path:
        method = 'decision_tree_classifier'
        d_treec = fit_load(path, method)
    else:
        d_treec = pickle.loads(fit_info)

    return d_treec.predict(x, **kwargs)


def support_vector_machine(x: npt.ArrayLike, path, fit_info, **kwargs) -> np.ndarray:
    
    if path:
        method = 'support_vector_machine'
        svm= fit_load(path, method)
    else:
        svm = pickle.loads(fit_info)

    return svm.predict(x, **kwargs)


def logistic_regression(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:

    if path:
        method = 'logistic_regression'
        logistic= fit_load(path, method)
    else:
        logistic = pickle.loads(fit_info)

    return logistic.predict(x, **kwargs)


def k_nearest_neighbors(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:
    
    if path:
        method = 'support_vector_machine'
        knn= fit_load(path, method)
    else:
        knn = pickle.loads(fit_info)

    return knn.predict(x, **kwargs)


def random_florest(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:
    
    if path:
        method = 'support_vector_machine'
        d_forest= fit_load(path, method)
    else:
        d_forest = pickle.loads(fit_info)

    return d_forest.predict(x, **kwargs)


def xgboost(x: npt.ArrayLike, path, fit_info, **kwargs)-> np.ndarray:
    
    if path:
        method = 'support_vector_machine'
        xg= fit_load(path, method)
    else:
        xg = pickle.loads(fit_info)

    return xg.predict(x, **kwargs)

#def aautoml(x: npt.ArrayLike, path, **kwargs)-> np.ndarray:

    #auto = pickle.load(open(path+"\\automl_fit_property.pkl", 'rb'))
    
    #return auto.predict(x,**kwargs)

_predict_methods = {
    "gaussian_naive_bayes": gaussian_naive_bayes,
    "decision_tree_classifier": decision_tree_classifier,
    "support_vector_machine": support_vector_machine,
    "logistic_regression": logistic_regression,
    "k_neighbors_classifier": k_nearest_neighbors,
    "random_forest_classifier": random_florest,
    "x_g_boost_classifier": xgboost,
    #'AutomlClassifier': automl 
    }


def predict(x: npt.ArrayLike, method: str = "GaussianNB", path = ".", fit_info = False,
            scalers=False, **kwargs):

    if method == "gaussian_naive_bayes":
        fun = _predict_methods[method]
    if method == "decision_tree_classifier":
        fun = _predict_methods[method]
    if method == "support_vector_machine":
        fun = _predict_methods[method]
    if method == "logistic_regression":
        fun = _predict_methods[method]
    if method == "k_neighbors_classifier":
        fun = _predict_methods[method]
    if method == "random_forest_classifier":
        fun= _predict_methods[method]
    if method == "x_g_boost_classifier":
        fun= _predict_methods[method]
    if method == "cat_boost_classifier":
        fun= _predict_methods[method]
    #if method == "AutoML":
        #fun = _predict_methods[method]
    if method == "scalers":
        return 0
    
    if method != "scalers":

        if not scalers:
            scaler = pickle.load(
                open(os.path.join(path,method+"_scaler.pkl"), 'rb'))
            scalerp = pickle.load(
                open(os.path.join(path,method+"_scalerp.pkl"), 'rb'))
            le = pickle.load(
                open(os.path.join(path,method+"_y_encoded.pkl"), 'rb'))
            
        else:
            scaler,scalerp,le = scalers

        x_norm = pickle.loads(scaler).transform(x)
        x_norm = pickle.loads(scalerp).transform(x_norm)

        y = fun(x_norm, path, fit_info, **kwargs)

        y_encoded = pickle.loads(le).inverse_transform(y)
        return y_encoded
