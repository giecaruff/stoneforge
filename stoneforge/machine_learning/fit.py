import numpy as np
import numpy.typing as npt
import pickle
import json
import warnings

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

def saves(file, name):
    with open(name + ".pkl", "wb") as write_file:
        pickle.dump(file, write_file)


#Naive Bayes
def gaussian_naive_bayes(X: npt.ArrayLike, y: npt.ArrayLike, path, gs = False, **kwargs) -> np.ndarray:

    f = open(path + '\\gaussian_naive_bayes_settings.json')

    settings = json.load(f)

    naive = GaussianNB(**settings)

    naive.fit(X, y, **kwargs)
    
    saves(naive, path+"\\gaussian_naive_bayes_fit_property")


#Decision Tree 
def decision_tree_classifier(X: npt.ArrayLike, y: npt.ArrayLike, path, gs = False, **kwargs) -> np.ndarray:

    if gs:
        parameters = {'criterion': ['gini', 'entropy'],
        'max_depth':[5,10,15,30,50,70,100],
        'random_state':[99]}

        decisiontree = DecisionTreeClassifier()

        bestdt = GridSearchCV(decisiontree,parameters,scoring='accuracy')
        bestdt.fit(X,y)
        settings = bestdt.best_params_

    if not gs:
        f = open(path + '\\decision_tree_classifier_settings.json')
        settings = json.load(f)
        if not settings:
            settings['random_state'] = 99
    
    d_treec = DecisionTreeClassifier(**settings)
    d_treec.fit(X, y, **kwargs)
    
    saves(d_treec, path+"\\decision_tree_classifier_fit_property")


#Support Machine Vector
def support_vector_machine(X: npt.ArrayLike, y: npt.ArrayLike, path, gs = False, **kwargs) -> np.ndarray:

    f = open(path + '\\support_vector_machine_settings.json')

    settings = json.load(f)
    if not settings:
        settings['random_state'] = 99

    svm = SVC(**settings)

    svm.fit(X, y, **kwargs)

    saves(svm, path+"\\support_vector_machine_fit_property")


#Logistic Regression
def logistic_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, gs = False, **kwargs) -> np.ndarray:

    f = open(path + '\\logistic_regression_settings.json')

    settings = json.load(f)
    if not settings:
        settings['random_state'] = 99

    logistic = LogisticRegression(**settings)

    logistic.fit(X, y, **kwargs)

    saves(logistic, path+"\\logistic_regression_fit_property")


# K-nearest Neighbors
def k_nearest_neighbors(X: npt.ArrayLike, y: npt.ArrayLike, path, gs = False, **kwargs) -> np.ndarray:

    if gs:
        parameters = {'n_neighbors': np.arange(3,61,2),
        'weights':['uniform', 'distance'],
        'p':np.arange(1,6)}

        knn = KNeighborsClassifier()

        bestknn = GridSearchCV(knn,parameters,scoring='accuracy')
        bestknn.fit(X,y)
        settings = bestknn.best_params_

    if not gs:
        f = open(path + '\\k_nearest_neighbors_settings.json')
        settings = json.load(f)
    
    knn = KNeighborsClassifier(**settings)
    knn.fit(X, y, **kwargs)
    
    saves(knn, path+"\\k_nearest_neighbors_fit_property")


#Random Forest
def random_florest(X: npt.ArrayLike, y: npt.ArrayLike, path, gs =False, **kwargs) -> np.ndarray:
    
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
        f = open(path + '\\random_florest_settings.json')
        settings = json.load(f)
        if not settings:
            settings['random_state'] = 99
    
    d_florest = RandomForestClassifier(**settings)
    d_florest.fit(X, y, **kwargs)

    saves(d_florest, path+"\\random_florest_fit_property")


#XGBOOST
def xgboost(X: npt.ArrayLike, y: npt.ArrayLike, path, gs=False, **kwargs) -> np.ndarray:

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
        f = open(path + '\\xgboost_settings.json')
        settings = json.load(f)
    
    xg = XGBClassifier(**settings)
    xg.fit(X, y, **kwargs)

    saves(xg, path+"\\xgboost_fit_property")


#CatBoost
def catboost(X: npt.ArrayLike, y: npt.ArrayLike, path, gs=False, **kwargs) -> np.ndarray:

    if gs:
        parameters =  {'n_estimators': [100,150,200],
        'learning_rate': [0.3,0.5,0.7],
        'max_depth':[5,10,15,30,50,70,100],
        'random_seed':[99]}

        cat = CatBoostClassifier()

        bestcat = GridSearchCV(cat,parameters,scoring='accuracy')
        bestcat.fit(X,y)
        settings = bestcat.best_params_

    if not gs:
        f = open(path + '\\catboost_settings.json')
        settings = json.load(f)
        if not settings:
            settings['random_state'] = 99
    
    cb = CatBoostClassifier(**settings)
    cb.fit(X, y, **kwargs)

    saves(cb, path+"\\catboost_fit_property")


_fit_methods = {
    "GaussianNB": gaussian_naive_bayes,
    "DecisionTreeClassifier": decision_tree_classifier,
    "SVM": support_vector_machine,
    "LogisticRegression": logistic_regression,
    "KNeighborsClassifier": k_nearest_neighbors,
    "RandomForestClassifier": random_florest,
    'XGBClassifier': xgboost,
    'CatBoostClassifier': catboost
    #'AutomlClassifier': automl 
    }


def fit(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "GaussianNB", path = ".", **kwargs):

    if method == "GaussianNB":
        fun = _fit_methods[method]
    if method == "DecisionTreeClassifier":
        fun = _fit_methods[method]
    if method == "SVM":
        fun = _fit_methods[method]
    if method == "LogisticRegression":
        fun = _fit_methods[method]
    if method == "KNeighborsClassifier":
        fun = _fit_methods[method]
    if method == "RandomForestClassifier":
        fun = _fit_methods[method]
    if method == "XGBClassifier":
        fun = _fit_methods[method]
    if method == "CatBoostClassifier":
        fun = _fit_methods[method]
    #if method == "AutoML":
        #fun = _fit_methods[method]

    scaler = StandardScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    saves(scaler, path+"\\scaler")

    # TODO: repass to preprocessing due to the sobreposition of processes
    
    le = LabelEncoder()
    le.fit(y)
    y_encoded = le.transform(y)
    saves(le, path+"\\LabelEncoded")
        
    fun(X_norm, y_encoded, path, **kwargs)
