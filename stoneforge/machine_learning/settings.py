import json
import pickle
import os

ML_METHODS = [
    "gaussian_naive_bayes",
    "decision_tree_classifier",
    "support_vector_machine",
    "logistic_regression",
    "k_neighbors_classifier",
    "random_forest_classifier",
#    "x_g_boost_classifier",
]

def methods():
    return ML_METHODS

def saves(file, name):
    with open(name+'_settings.json', 'w') as write_file:
        json.dump(file, write_file)

def settings(method: str = "gaussian_naive_bayes", path = ".", **kwargs):

    if method == "gaussian_naive_bayes":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "decision_tree_classifier":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "support_vector_machine":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "logistic_regression":
        kwargs['solver'] = 'liblinear' ### due to some error in scikit
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "k_neighbors_classifier":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

    if method == "random_forest_classifier":
        if path:
            saves(kwargs, os.path.join(path, method))
        else:
            return pickle.dumps(kwargs)

#    if method == "x_g_boost_classifier":
#        if path:
#            saves(kwargs, os.path.join(path, method))
#        else:
#            return pickle.dumps(kwargs)

    else:
        return methods()
    
    #if method == "AutoML":
        #saves(kwargs, path+'\\automl_settings')
