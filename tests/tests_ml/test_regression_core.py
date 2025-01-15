# %% ============================================================== #
import sys
import os
import numpy as np
import pandas as pd
import time
import pytest
import pickle

if __package__:
    from ..data_replacement import *
    from ..datasets import dataload
else:
    from stoneforge.data_replacement import *
    from stoneforge.datasets import dataload

# %% ============================================================== #

def _path_selection(target_parts):
    current_abs_dir = os.path.dirname(__file__)

    while not os.path.exists(os.path.join(current_abs_dir, target_parts[0])):
        current_abs_dir = os.path.dirname(current_abs_dir)

    target_abs_path = os.path.join(current_abs_dir, *target_parts)

    return target_abs_path

main_path = _path_selection(["stoneforge", "tests", "tests_ml", ".mlprj"])
print(main_path)

class setup_methods:
    def __init__(self, X, Xte, y, yte, path, decimals):
        self.X = X
        self.Xte = Xte
        self.y = y
        self.yte = yte
        self.path = path
        self.decimals = decimals

    def test_method(self, method, degree = 2):

        #if method == "linear_regression_polynomial":
        #    a = settings(method = method, verbose = True, path = False, degree = degree)
        #    print("settings done",len(a))
        #    b = fit(X = self.X, y = self.y, method = method, path = False, settings = a)
        #    print("fit done",type(b))
        #    c = fit(X = self.X, y = self.y, method = "scaler_regression", path = False, settings = a)
        #    d = predict(self.Xte, method = method, path = False, fit_info = b, scalers = c)
        #    print(np.mean(d))
        #else:
        a = settings(method = method, verbose = True, path = False)
        print("settings done",type(a))
        b = fit(X = self.X, y = self.y, method = method, path = False, settings = a)
        print("fit done",type(b))
        c = fit(X = self.X, y = self.y, method = "scaler_regression", path = False, settings = a)
        d = predict(self.Xte, method = method, path = False, fit_info = b, scalers = c)
        print('mean:',d,'\n')
        

# %% ============================================================== #

IK1,unik1 = dataload.ik1()

IK1_c = IK1[~IK1.isin([-999.0]).any(axis=1)]

DP1,unik1 = dataload.dp1()

DP1_c = DP1[~DP1.isin([-999.0]).any(axis=1)]

# %% ============================================================== #

DT = list(IK1_c["DT"]) + list(DP1_c["DT"])
GR = list(IK1_c["GR"]) + list(DP1_c["GR"])

RHOB = list(IK1_c["RHOB"]) + list(DP1_c["RHOB"])
NPHI = list(IK1_c["NPHI"]) + list(DP1_c["NPHI"])

np.random.seed(42)

data_matrix = np.array([RHOB,NPHI,DT,GR], float).T
np.random.shuffle(data_matrix)

RHOB = data_matrix[:,0]
NPHI = data_matrix[:,1]
DT = data_matrix[:,2]
GR = data_matrix[:,3]

X = np.array([RHOB,NPHI]).T

ES1,unik1 = dataload.es1()
ES1_c = ES1[~ES1.isin([-999.0]).any(axis=1)]
X_test = np.array([ES1_c["RHOB"],ES1_c["NPHI"]],float).T
y_test = np.array([ES1_c["DT"],ES1_c["GR"]])

#Y = np.array([DT,GR]).T
Y = np.array(DT)
print(np.shape(Y))

_test = setup_methods(X, X_test, Y, y_test, main_path, 3)


# %% ============================================================== #


_test.test_method("linear_regression_simple")


# %% ============================================================== #


_test.test_method("linear_regression_polynomial")


# %% ============================================================== #


_test.test_method("decision_tree_regression")


# %% ============================================================== #
# works only for one parameter

_test.test_method("support_vector_regression")


# %% ============================================================== #


_test.test_method("random_forest_regression")


# %% ============================================================== #
# works for only one parameter

_test.test_method("lightgbm_regression")


# %% ============================================================== #


_test.test_method("xgboost_regression")


# %% ============================================================== #
# works for only one parameter

_test.test_method("catboost_regression")


# %% ============================================================== #
