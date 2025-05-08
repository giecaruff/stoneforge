# %% ============================================================== #
import sys
import os
import numpy as np
import pandas as pd
import time
import pytest
import pickle

if __package__:
    from ..machine_learning import *
    from ..datasets import dataload
else:
    from stoneforge.machine_learning import *
    from stoneforge.datasets import dataload

# %% ============================================================== #


class setup_methods:
    def __init__(self, X, Xte, y, yte):
        self.X = X
        self.Xte = Xte
        self.y = y
        self.yte = yte

    def test_method(self, method):

        print("method:",method)
        a = settings(method = method, path = False)
        print("settings done",type(a))
        b = fit(X = self.X, y = self.y, method = method, path = False, settings = a)
        print("fit done",type(b))
        c = fit(X = self.X, y = self.y, method = "scalers", path = False, settings = a)
        d = predict(self.Xte, method = method, path = False, fit_info = b, scalers = c)
        print('value:',d,'DONE! \n')
        

# %% ============================================================== #

IK1,unik1 = dataload.ik1()

IK1_c = IK1[~IK1.isin([-999.0]).any(axis=1)]

DP1,unik1 = dataload.dp1()

DP1_c = DP1[~DP1.isin([-999.0]).any(axis=1)]

# %% ============================================================== #

DT = list(IK1_c["DT"]) + list(DP1_c["DT"])
GR = np.array(list(IK1_c["GR"]) + list(DP1_c["GR"]),float)

RHOB = list(IK1_c["RHOB"]) + list(DP1_c["RHOB"])
NPHI = list(IK1_c["NPHI"]) + list(DP1_c["NPHI"])

gr_min = min(GR)
gr_max = max(GR)
vsh = (GR - gr_min) / (gr_max - gr_min)
condition = (vsh <= 0.4)
LITO = np.where(condition, 49, 57)

np.random.seed(42)

data_matrix = np.array([RHOB,NPHI,LITO], float).T
np.random.shuffle(data_matrix)

RHOB = data_matrix[:,0]
NPHI = data_matrix[:,1]
LITO = data_matrix[:,2]

X = np.array([RHOB,NPHI]).T

ES1,unik1 = dataload.es1()
ES1_c = ES1[~ES1.isin([-999.0]).any(axis=1)]
X_test = np.array([ES1_c["RHOB"],ES1_c["NPHI"]],float).T
test_vsh = np.array((ES1_c["GR"] - gr_min) / (gr_max - gr_min),float)
test_condition = (test_vsh <= 0.4)
y_test = np.where(test_condition, 49, 57)

_test = setup_methods(X, X_test, LITO, y_test )


# %% ============================================================== #


_test.test_method("gaussian_naive_bayes")


# %% ============================================================== #


_test.test_method("decision_tree_classifier")


# %% ============================================================== #


_test.test_method("support_vector_machine")


# %% ============================================================== #


_test.test_method("logistic_regression")


# %% ============================================================== #


_test.test_method("k_neighbors_classifier")


# %% ============================================================== #


_test.test_method("random_forest_classifier")


# %% ============================================================== #


_test.test_method("x_g_boost_classifier")


# %% ============================================================== #


#_test.test_method("cat_boost_classifier")


# %% ============================================================== #