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


#Naive Bayes
def gaussian_naive_bayes(X: npt.ArrayLike, y: npt.ArrayLike, path, n_splits, random_state, **kwargs) -> np.ndarray:

    f = open(path + '\\gaussian_naive_bayes_settings.json')

    settings = json.load(f)

    naive = GaussianNB(**settings)
    #x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = test_size, random_state = random_state)
    #sklearn.metrics.confusion_matrix( y_treino, y_teste, *, labels=None, sample_weight=None, normalize=None)

    kfold = KFold(n_splits = n_splits, shuffle=True, random_state = random_state)

    result = cross_val_score(naive, X, y, cv = kfold)
    print(result)


_fit_methods = {
    "GaussianNB": gaussian_naive_bayes,
    #"DecisionTreeClassifier": decision_tree_classifier,
    #"SVM": support_vector_machine,
    #"LogisticRegression": logistic_regression,
    #"KNeighborsClassifier": k_nearest_neighbors,
    #"RandomForestClassifier": random_florest,
    #'XGBClassifier': xgboost,
    #'CatBoost': catboost
    #'AutomlClassifier': automl 
    }

def validation(X: npt.ArrayLike , y: npt.ArrayLike, method: str = "GaussianNB", path = ".", n_splits = 30, random_state = 5, **kwargs):

    X_norm = StandardScaler().fit_transform(X)

    if method == "GaussianNB":
        fun = _fit_methods[method]
    #if method == "DecisionTreeClassifier":
    #    fun = _fit_methods[method]
    #if method == "SVM":
    #    fun = _fit_methods[method]
    #if method == "LogisticRegression":
    #    fun = _fit_methods[method]
    #if method == "KNeighborsClassifier":
    #    fun = _fit_methods[method]
    #if method == "RandomForestClassifier":
    #    fun = _fit_methods[method]
    #if method == "XGBClassifier":
    #    fun = _fit_methods[method]
    #if method == "CatBoost":
    #    fun = _fit_methods[method]
    #if method == "AutoML":
        #fun = _fit_methods[method]

    X_norm = StandardScaler().fit_transform(X)
        
    fun(X_norm, y, path, n_splits, random_state, **kwargs)




