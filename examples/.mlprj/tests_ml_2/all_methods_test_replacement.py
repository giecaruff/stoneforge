# %%

import numpy as np
import sys
import os
import pandas
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, matthews_corrcoef, mean_absolute_error
import json
import matplotlib.pyplot as plt

if __package__:
    from ..data_replacement import *
else:
    print('passed lr')
    sys.path.append(os.path.dirname(__file__) + '/..')
    import data_replacement

if __package__:
    from ..preprocessing import *
else:
    print('passed mgt')
    sys.path.append(os.path.dirname(__file__) + '/..')
    import preprocessing

# %%

#project = preprocessing.project("D:\\appy_projetos\\wells")
#project = preprocessing.project("Y:\\appy_dados\\wells_es")
project = preprocessing.project("Y:\\appy_dados\\wells_es_restricted")
project.import_folder()
project.import_several_wells()

print(project.well_names_paths)

#mr = open("mnemonics_replacement.json")
#mnemonics_replacement = json.load(mr)

mnemonics_replacement = {
    "DEPTH": ["DEPTH","DEPT","MD"],
    "VP":["VP","vp","Vp"],
    "VS":["VS","vs","Vs"],
    "RHOB":["RHOB","RHLA","RHBA","RHLA3","RHBA4"]
}

ref_mnemonics = list(mnemonics_replacement.keys())
print(ref_mnemonics)

project.data_replacement(mnemonics_replacement)
project.convert_into_matrix(ref_mnemonics)

for p in project.well_names_las:
    print("poco:",p)

# %%

tw_data,vw_data = preprocessing.well_train_test_split(['6-BRSA-639-ESS_Default_final'],project.well_data)

mega_data = preprocessing.data_assemble(tw_data,'data')
print(np.shape(mega_data))

a = preprocessing.predict_processing(vw_data,'data')
mega_data = a._remove_dummies(mega_data)
print(np.shape(mega_data))

# %%

data_replacement.settings(method = "linear_regression_simple", path='_ml_project') # for the case there is polynomial 1D
data_replacement.settings(method = "linear_regression_polynomial", path='_ml_project', degree = 2)
data_replacement.settings(method = "decision_tree_regression", path='_ml_project')
data_replacement.settings(method = "support_vector_regression", path='_ml_project')
data_replacement.settings(method = "random_forest_regression", path='_ml_project')
#data_replacement.settings(method = "lightgbm_regression", path='_ml_project')
#data_replacement.settings(method = "LogisticRegression", path='_ml_project')
#data_replacement.settings(method = "xgboost_regression", path='_ml_project')

# %%

y = mega_data[:,2] # 2 for VS 
X = np.delete(mega_data,(2), axis=1) # 2 for VS
X = np.delete(X,(0), axis=1) # 0 for MD
print(np.shape(X),X) # must be only VP and RHOB
X = np.array(X, dtype='float') 
y = np.array(y, dtype='float')
print(np.shape(y),y)

# %%

data_replacement.fit(X,y,method = "linear_regression_simple", path = "_ml_project")
data_replacement.fit(X,y,method = "linear_regression_polynomial", path = "_ml_project")
data_replacement.fit(X,y,method = "decision_tree_regression", path = "_ml_project")
data_replacement.fit(X,y,method = "support_vector_regression", path = "_ml_project")
#data_replacement.fit(X,y,method = "LogisticRegression", path = "_ml_project")
#data_replacement.fit(X,y,method = "lightgbm_regression", path = "_ml_project")
data_replacement.fit(X,y,method = "random_forest_regression", path = "_ml_project")
#data_replacement.fit(X,y,method = "xgboost_regression", path = '_ml_project')

# %%
import matplotlib.pyplot as plt
pre_pros = preprocessing.predict_processing(vw_data,'data')
xy_raw = pre_pros.matrix_values()

y_db = {}
x_db = {}
for w in xy_raw:
    print(w)
    #print( xy_raw[w])
    y_db[w] = xy_raw[w][:,2]
    _xy = xy_raw[w]
    _xy = np.delete(_xy,(2), axis=1)
    x_db[w] = np.delete(_xy,(0), axis=1)
    #print(x_db)
    #ym_db = data_replacement.predict(x_db[w], method = "linear_regression_polynomial", path = "_ml_project")
    ym_db = data_replacement.predict(x_db[w], method = "linear_regression_simple", path = "_ml_project")
    #print(mean_squared_error(ym_db, y_db[w]))
    a = data_replacement.evaluation(ym_db, y_db[w], decimals = 3, path = False)
    print(a)
    plt.plot(ym_db, y_db[w],'.')
    #plt.plot(ym_db,'.')
    plt.grid()
    plt.show()

# %%

print(a)

# %%

class_db = {}

for well in x_db:
    class_db[well] = data_replacement.predict(x_db[well], method = "linear_regression_simple", path = "_ml_project")
    class_db[well] = data_replacement.predict(x_db[well], method = "linear_regression_polynomial", path = "_ml_project")
    class_db[well] = data_replacement.predict(x_db[well], method = "decision_tree_regression", path = "_ml_project")
    class_db[well] = data_replacement.predict(x_db[well], method = "support_vector_regression", path = "_ml_project")
    #class_db[well] = data_replacement.predict(x_db[well], method = "LogisticRegression", path = "_ml_project")
    #class_db[well] = data_replacement.predict(x_db[well], method = "xgboost_regression", path = "_ml_project")
    class_db[well] = data_replacement.predict(x_db[well], method = "random_forest_regression", path = "_ml_project")

pre_pros.return_curve(class_db)
# %%
