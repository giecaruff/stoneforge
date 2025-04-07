import numpy as np
import numpy.typing as npt
import pickle
import json
import warnings
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

ML_METHODS = [
    "gaussian_naive_bayes",
    "decision_tree_classifier",
    "support_vector_machine",
    "logistic_regression",
    "k_neighbors_classifier",
    "random_forest_classifier",
    "x_g_boost_classifier",
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

#Naive Bayes
def gaussian_naive_bayes(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs) -> np.ndarray:
    
    method = "gaussian_naive_bayes"

    if path:
        settings = load_settings(path, method)
    else:
        settings = pickle.loads(settings)

    naive = GaussianNB(**settings)

    naive.fit(X, y, **kwargs)

    if not path:
        serialized_model = pickle.dumps(naive)
        return serialized_model
    else:
        saves(naive, path, method)


#Decision Tree 
def decision_tree_classifier(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs) -> np.ndarray:
    
    method = "decision_tree_classifier"

    if gs:
        parameters = {'criterion': ['gini', 'entropy'],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        decisiontree = DecisionTreeClassifier()

        bestdt = GridSearchCV(decisiontree,parameters,scoring='accuracy')
        bestdt.fit(X,y)
        settings = bestdt.best_params_

    if not gs:
        if path:
            settings = load_settings(path, method)
        else:
            settings = pickle.loads(settings)

    settings['random_state'] = 99
    d_treec = DecisionTreeClassifier(**settings)
    d_treec.fit(X, y, **kwargs)

    if not path:
        serialized_model = pickle.dumps(d_treec)
        return serialized_model
    else:
        saves(d_treec, path, method)


#Support Machine Vector
def support_vector_machine(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs) -> np.ndarray:

    method = "support_vector_machine"

    if path:
        settings = load_settings(path, method)
    else:
        settings = pickle.loads(settings)

    settings['random_state'] = 99    

    svm = SVC(**settings)

    svm.fit(X, y, **kwargs)

    if not path:
        serialized_model = pickle.dumps(svm)
        return serialized_model
    else:
        saves(svm, path, method)


#Logistic Regression
def logistic_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs) -> np.ndarray:

    method = "logistic_regression"

    if path:
        settings = load_settings(path, method)
    else:
        settings = pickle.loads(settings)

    settings['random_state'] = 99

    logistic = LogisticRegression(**settings)

    logistic.fit(X, y, **kwargs)

    if not path:
        serialized_model = pickle.dumps(logistic)
        return serialized_model
    else:
        saves(logistic, path, method)


# K-nearest Neighbors
def k_nearest_neighbors(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs) -> np.ndarray:

    method = "k_neighbors_classifier"

    if gs:
        parameters = {'n_neighbors': np.arange(3,61,2),
        'weights':['uniform', 'distance'],
        'p':np.arange(1,6)}

        knn = KNeighborsClassifier()

        bestknn = GridSearchCV(knn,parameters,scoring='accuracy')
        bestknn.fit(X,y)
        settings = bestknn.best_params_

    if not gs:
        if path:
            settings = load_settings(path, method)
        else:
            settings = pickle.loads(settings)
    
    knn = KNeighborsClassifier(**settings)
    knn.fit(X, y, **kwargs)
    
    if not path:
        serialized_model = pickle.dumps(knn)
        return serialized_model
    else:
        saves(knn, path, method)


#Random Forest
def random_florest(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs) -> np.ndarray:

    method = "random_forest_classifier"
    
    if gs:
        parameters = {'criterion': ['gini', 'entropy'],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        randomflorest = RandomForestClassifier()

        bestrf = GridSearchCV(randomflorest,parameters,scoring='accuracy')
        bestrf.fit(X,y)
        #bestrf = bestrf.best_params_
        settings = bestrf.best_params_

    if not gs:
        if path:
            settings = load_settings(path, method)
        else:
            settings = pickle.loads(settings)
    settings['random_state'] = 99
    
    d_florest = RandomForestClassifier(**settings)
    d_florest.fit(X, y, **kwargs)

    if not path:
        serialized_model = pickle.dumps(d_florest)
        return serialized_model
    else:
        saves(d_florest, path, method)


#XGBOOST
def xgboost(X: npt.ArrayLike, y: npt.ArrayLike, path, gs, settings, **kwargs) -> np.ndarray:

    method = "x_g_boost_classifier"

    if gs:
        parameters =  {'n_estimators': [100],
        'learning_rate': [0.5],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        xgb = XGBClassifier()

        bestxgb = GridSearchCV(xgb,parameters,scoring='accuracy')
        bestxgb.fit(X,y)
        #bestxgb = bestxgb.best_params_
        settings = bestxgb.best_params_


    if not gs:
        if path:
            settings = load_settings(path, method)
        else:
            settings = pickle.loads(settings)

    settings['random_state'] = 99
    xg = XGBClassifier(**settings)
    xg.fit(X, y, **kwargs)

    if not path:
        serialized_model = pickle.dumps(xg)
        return serialized_model
    else:
        saves(xg, path, method)

_fit_methods = {
    "gaussian_naive_bayes": gaussian_naive_bayes,
    "decision_tree_classifier": decision_tree_classifier,
    "support_vector_machine": support_vector_machine,
    "logistic_regression": logistic_regression,
    "k_neighbors_classifier": k_nearest_neighbors,
    "random_forest_classifier": random_florest,
    "x_g_boost_classifier": xgboost,
    #'AutomlClassifier': automl 
    }


def fit(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "gaussian_naive_bayes",
        path = ".", gs = False, settings = False, **kwargs):

    if method == "gaussian_naive_bayes":
        fun = _fit_methods[method]
    if method == "decision_tree_classifier":
        fun = _fit_methods[method]
    if method == "support_vector_machine":
        fun = _fit_methods[method]
    if method == "logistic_regression":
        fun = _fit_methods[method]
    if method == "k_neighbors_classifier":
        fun = _fit_methods[method]
    if method == "random_forest_classifier":
        fun = _fit_methods[method]
    if method == "x_g_boost_classifier":
        fun = _fit_methods[method]
    if method == "cat_boost_classifier":
        fun = _fit_methods[method]
    #if method == "AutoML":
        #fun = _fit_methods[method]

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)

    scalerp = StandardScaler()
    scalerp.fit(X_norm)
    X_norm = scaler.transform(X_norm)

    # TODO: repass to preprocessing due to the sobreposition of processes
    
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)

    if method == "scalers":
        return pickle.dumps(scaler), pickle.dumps(scalerp), pickle.dumps(le)

    if not path:
        return fun(X_norm, y_encoded, path, gs, settings, **kwargs)
    else:
        saves(scaler, path, method+'_scaler', suffix = '.pkl')
        saves(scalerp, path, method+'_scalerp', suffix = '.pkl')
        saves(le, path, method+'_y_encoded', suffix = '.pkl')
        fun(X_norm, y_encoded, path, gs, settings, **kwargs)
