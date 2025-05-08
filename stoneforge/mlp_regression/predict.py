import numpy as np
import numpy.typing as npt
import pickle


def multi_layer_perceptron(x: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    mlpc = pickle.load(open(path+"\\multi_layer_perceptron_fit_property.pkl", 'rb'))

    return mlpc.predict(x, **kwargs)

_predict_methods = {
    "MLPRegressor": multi_layer_perceptron}

def predict(x: npt.ArrayLike, method: str = "MLPRegressor", path = ".", **kwargs):

    if method == "MLPRegressor":
        fun = _predict_methods[method]



    scaler = pickle.load(open(path+"\\scaler.pkl", 'rb'))
    le = pickle.load(open(path+"\\LabelEncoded.pkl", 'rb'))
    #scaler.fit(x)

    x_norm = scaler.transform(x)
    y = fun(x_norm, path, **kwargs)
    y_decoded = le.inverse_transform(y)

    return y_decoded