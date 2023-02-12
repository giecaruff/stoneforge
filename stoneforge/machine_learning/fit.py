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

def saves(file, name):
    with open(name + ".pkl", "wb") as write_file:
        pickle.dump(file, write_file)


#Naive Bayes
def gaussian_naive_bayes(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    f = open(path + '\\gaussian_naive_bayes_settings.json')

    settings = json.load(f)

    naive = GaussianNB(**settings)

    naive.fit(X, y, **kwargs)
    
    saves(naive, path+"\\gaussian_naive_bayes_fit_property")


#Arvore de decisões 
def decision_tree_classifier(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    f = open(path + '\\decision_tree_classifier_settings.json')

    settings = json.load(f)

    d_treec = DecisionTreeClassifier(**settings)

    d_treec.fit(X, y, **kwargs)
    
    saves(d_treec, path+"\\decision_tree_classifier_fit_property")


#Maquina de Vetores de Suporte
def support_vector_machine(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    f = open(path + '\\support_vector_machine_settings.json')

    settings = json.load(f)

    svm = SVC(**settings)

    svm.fit(X, y, **kwargs)

    saves(svm, path+"\\support_vector_machine_fit_property")


#Regressão Logistica
def logistic_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    f = open(path + '\\logistic_regression_settings.json')

    settings = json.load(f)

    logistic = LogisticRegression(**settings)

    logistic.fit(X, y, **kwargs)

    saves(logistic, path+"\\logistic_regression_fit_property")

#APRENDIZAGEM BASEADA EM INSTÂNCIAS 

def k_nearest_neighbors(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    f = open(path + '\\k_nearest_neighbors_settings.json')

    settings = json.load(f)

    knn = KNeighborsClassifier(**settings)

    knn.fit(X, y, **kwargs)

    saves(knn, path+"\\k_nearest_neighbors_fit_property")

#Arvore Aleatoria

def random_florest(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    f = open(path + '\\random_florest_settings.json')

    settings = json.load(f)

    d_florest = RandomForestClassifier(**settings)

    d_florest.fit(X, y, **kwargs)

    saves(d_florest, path+"\\random_florest_fit_property")

#XGBOOST

def xgboost(X: npt.ArrayLike, y: npt.ArrayLike, path, **kwargs) -> np.ndarray:

    f = open(path + '\\xgboost_settings.json')

    settings = json.load(f)

    xg = XGBClassifier(**settings)

    xg.fit(X, y, **kwargs)

    saves(xg, path+"\\xgboost_fit_property")



_fit_methods = {
    "GaussianNB": gaussian_naive_bayes,
    "DecisionTreeClassifier": decision_tree_classifier,
    "SVM": support_vector_machine,
    "LogisticRegression": logistic_regression,
    "KNeighborsClassifier": k_nearest_neighbors,
    "RandomForestClassifier": random_florest,
    'XGBClassifier': xgboost
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
    
    
        
    fun(X, y, path, **kwargs)
