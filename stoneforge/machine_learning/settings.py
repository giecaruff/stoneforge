import numpy as np
import numpy.typing as npt
import json
import warnings

def saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)

def settings(method: str = "GaussianNB", path = ".", **kwargs):

    if method == "GaussianNB":
        saves(kwargs, path+"\\gaussian_naive_bayes_settings")

    if method == "DecisionTreeClassifier":
        saves(kwargs, path+"\\decision_tree_classifier_settings")

    if method == "SVM":
        saves(kwargs, path+"\\support_vector_machine_settings")

    if method == "LogisticRegression":
        kwargs['solver'] = 'liblinear' ### due to some error in scikit
        saves(kwargs, path+"\\logistic_regression_settings")

    if method == "KNeighborsClassifier":
        saves(kwargs, path+'\\k_nearest_neighbors_settings')

    if method == "RandomForestClassifier":
        saves(kwargs, path+'\\random_florest_settings')

    if method == "XGBClassifier":
        saves(kwargs, path+'\\xgboost_settings')
    
    if method == "AutoML":
        saves(kwargs, path+'\\automl_settings')
