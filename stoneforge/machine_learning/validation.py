import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from . import fit
from sklearn.preprocessing import StandardScaler
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

def saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)

#Naive Bayes
def gaussian_naive_bayes(X: npt.ArrayLike, y: npt.ArrayLike, path, n_splits, random_state, **kwargs) -> np.ndarray:

    f = open(path + '\\gaussian_naive_bayes_settings.json')

    settings = json.load(f)

    naive = GaussianNB(**settings)

    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = random_state)

    result = cross_val_score(naive, X, y, cv = kfold)
    mean_result = {}
    mean_result['mean_accuracy'] = round(result.mean() * 100,3)
    if path:
        saves(mean_result,path + '\\mean_accuracy')
    if not path:
        return mean_result

#Decision Tree Classifier
def decision_tree_classifier(X: npt.ArrayLike, y: npt.ArrayLike, path, n_splits, random_state, **kwargs) -> np.ndarray:

    f = open(path + '\\decision_tree_classifier_settings.json')

    settings = json.load(f)

    tree = DecisionTreeClassifier(**settings)

    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = random_state)

    result = cross_val_score(tree, X, y, cv = kfold)
    mean_result = {}
    mean_result['mean_accuracy'] = round(result.mean() * 100,3)
    if path:
        saves(mean_result,path + '\\mean_accuracy')
    if not path:
        return mean_result

#svm
def support_vector_machine(X: npt.ArrayLike, y: npt.ArrayLike, path, n_splits, random_state, **kwargs) -> np.ndarray:

    f = open(path + '\\support_vector_machine_settings.json')

    settings = json.load(f)

    svm = SVC(**settings)
    #x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = test_size, random_state = random_state)
    #sklearn.metrics.confusion_matrix( y_treino, y_teste, *, labels=None, sample_weight=None, normalize=None)

    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = random_state)

    result = cross_val_score(svm, X, y, cv = kfold)
    mean_result = {}
    mean_result['mean_accuracy'] = round(result.mean() * 100,3)
    if path:
        saves(mean_result,path + '\\mean_accuracy')
    if not path:
        return mean_result

#LogisticRegression
def logistic_regression(X: npt.ArrayLike, y: npt.ArrayLike, path, n_splits, random_state, **kwargs) -> np.ndarray:

    f = open(path + '\\logistic_regression_settings.json')

    settings = json.load(f)

    logistic = LogisticRegression(**settings)
    #x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = test_size, random_state = random_state)
    #sklearn.metrics.confusion_matrix( y_treino, y_teste, *, labels=None, sample_weight=None, normalize=None)

    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = random_state)

    result = cross_val_score(logistic, X, y, cv = kfold)
    mean_result = {}
    mean_result['mean_accuracy'] = round(result.mean() * 100,3)
    if path:
        saves(mean_result,path + '\\mean_accuracy')
    if not path:
        return mean_result


#KNeighborsClassifier
def  k_nearest_neighbors(X: npt.ArrayLike, y: npt.ArrayLike, path, n_splits, random_state, **kwargs) -> np.ndarray:

    f = open(path + '\\k_nearest_neighbors_settings.json')

    settings = json.load(f)

    knn = KNeighborsClassifier(**settings)
    #x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = test_size, random_state = random_state)
    #sklearn.metrics.confusion_matrix( y_treino, y_teste, *, labels=None, sample_weight=None, normalize=None)

    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = random_state)

    result = cross_val_score(knn, X, y, cv = kfold)
    mean_result = {}
    mean_result['mean_accuracy'] = round(result.mean() * 100,3)
    if path:
        saves(mean_result,path + '\\mean_accuracy')
    if not path:
        return mean_result


#RandomForestClassifier
def  random_florest(X: npt.ArrayLike, y: npt.ArrayLike, path, n_splits, random_state, **kwargs) -> np.ndarray:

    f = open(path + '\\random_florest_settings.json')

    settings = json.load(f)

    random = RandomForestClassifier(**settings)
    #x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = test_size, random_state = random_state)
    #sklearn.metrics.confusion_matrix( y_treino, y_teste, *, labels=None, sample_weight=None, normalize=None)

    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = random_state)

    result = cross_val_score(random, X, y, cv = kfold)
    mean_result = {}
    mean_result['mean_accuracy'] = round(result.mean() * 100,3)
    if path:
        saves(mean_result,path + '\\mean_accuracy')
    if not path:
        return mean_result


#XGBClassifier
def  xgboost(X: npt.ArrayLike, y: npt.ArrayLike, path, n_splits, random_state, **kwargs) -> np.ndarray:

    f = open(path + '\\xgboost_settings.json')

    settings = json.load(f)

    xgboost = XGBClassifier(**settings)
    #x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = test_size, random_state = random_state)
    #sklearn.metrics.confusion_matrix( y_treino, y_teste, *, labels=None, sample_weight=None, normalize=None)

    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = random_state)

    result = cross_val_score(xgboost, X, y, cv = kfold)
    mean_result = {}
    mean_result['mean_accuracy'] = round(result.mean() * 100,3)
    if path:
        saves(mean_result,path + '\\mean_accuracy')
    if not path:
        return mean_result

_fit_methods = {
    "GaussianNB": gaussian_naive_bayes,
    "DecisionTreeClassifier": decision_tree_classifier,
    "SVM": support_vector_machine,
    "LogisticRegression": logistic_regression,
    "KNeighborsClassifier": k_nearest_neighbors,
    "RandomForestClassifier": random_florest,
    'XGBClassifier': xgboost,
    #'CatBoostClassifier': catboost
    #'AutomlClassifier': automl 
    }


def validation(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "GaussianNB", path = ".", n_splits = 30, random_state = 5, **kwargs):

    X_norm = StandardScaler().fit_transform(X)

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
    #if method == "CatBoostClassifier":
    #    fun = _fit_methods[method]
    #if method == "AutoML":
        #fun = _fit_methods[method]

    X_norm = StandardScaler().fit_transform(X)
        
    fun(X_norm, y, path, n_splits, random_state, **kwargs)




