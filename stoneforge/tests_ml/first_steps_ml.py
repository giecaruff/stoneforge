# %%
import numpy as np
import sys
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split

if __package__:
    from ..machine_learning import *
else:
    print('passed')
    sys.path.append(os.path.dirname(__file__) + '/..')
    import machine_learning

# %%

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
n = len(set(y))

print(np.shape(X))
print(np.shape(y))
print(n)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# %%

machine_learning.settings(method = "GaussianNB", path = "ml_sets", priors=[0.33, 0.33, 0.34])

# %%

machine_learning.fit(method = "GaussianNB", path = "ml_sets", X = X_train, y = y_train)

# %%

classification = machine_learning.predict(method = "GaussianNB", path = "ml_sets", x = X_test)

# %%