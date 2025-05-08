# %% ============================================================== #
import os
import numpy as np
import time

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

        start = time.time()
        if method == "linear_regression_polynomial":
            settings(method = method, verbose = True, degree = degree, path = self.path)
        else:
            settings(method = method, verbose = True, path = self.path)
        fit(X = self.X, y = self.y, method = method, path = self.path)
        y_predicted = predict(self.Xte, method = method, path = self.path)
        print(evaluation(self.yte,y_predicted, decimals = self.decimals, path = False))
        end = time.time()
        print("Time :",(end-start) * 10**3, "ms \n")

# %% ============================================================== #

IK1,unik1 = dataload.ik1()

IK1_c = IK1[~IK1.isin([-999.0]).any(axis=1)]

DP1,unik1 = dataload.dp1()

DP1_c = DP1[~DP1.isin([-999.0]).any(axis=1)]

# %% ============================================================== #

DT = list(IK1_c["DT"]) + list(DP1_c["DT"])

RHOB = list(IK1_c["RHOB"]) + list(DP1_c["RHOB"])
NPHI = list(IK1_c["NPHI"]) + list(DP1_c["NPHI"])

np.random.seed(42)

data_matrix = np.array([RHOB,NPHI,DT], float).T
np.random.shuffle(data_matrix)

RHOB = data_matrix[:,0]
NPHI = data_matrix[:,1]
DT = data_matrix[:,2]

X = np.array([RHOB,NPHI]).T

ES1,unik1 = dataload.es1()
ES1_c = ES1[~ES1.isin([-999.0]).any(axis=1)]
X_test = np.array([ES1_c["RHOB"],ES1_c["NPHI"]],float).T
y_test = np.array(ES1_c["DT"])

_test = setup_methods(X, X_test, DT, y_test, main_path, 3)

# %% ============================================================== #

#def test_linear_regression_simple():
#    _test.test_method("linear_regression_simple")
#    assert True

# %% ============================================================== #

#def test_linear_regression_polynomial():
#    _test.test_method("linear_regression_polynomial")
#    assert True

# %% ============================================================== #

def test_decision_tree():
    _test.test_method("decision_tree_regression")
    assert True

# %% ============================================================== #

def test_support_vector():
    _test.test_method("support_vector_regression")
    assert True

# %% ============================================================== #

def test_random_forest():
    _test.test_method("random_forest_regression")
    assert True

# %% ============================================================== #

def test_lightgbm():
    _test.test_method("lightgbm_regression")
    assert True

# %% ============================================================== #

def test_xgboost():
    _test.test_method("xgboost_regression")
    assert True

# %% ============================================================== #

#def test_catboost():
#    _test.test_method("catboost_regression")
#    assert True

# %% ============================================================== #
