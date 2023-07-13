# %%

import numpy as np
import sys
import os
import pandas

from sklearn import datasets
from sklearn.model_selection import train_test_split

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
project = preprocessing.project("C:\\Users\\joseaugustodias\\Desktop\\wells")
project.import_folder()
project.import_several_wells()

print(project.well_names_paths)


#%%

mnemonics_replacement = {
    #'DEPTH':['DEPTH'],
    'GR':['GR'],
    #'CAL':['CAL','DCAL','HCAL','CALI'],
    'RHOB':["RHOB","RHOZ","RHLA","RHBA","RHLA3","RHBA4"],
    #'RES':["ILD","HDRS","RT","AHT901","AT90","RT90"],
    'NPHI':['NPHI'],
}

ref_mnemonics = list(mnemonics_replacement.keys())

project.data_replacement(mnemonics_replacement)
project.convert_into_matrix(ref_mnemonics)

# %%

b = np.delete(project.well_data['7-MP-33D-BA']['data'], -2, axis=0)
print(b,np.shape(b))

mnem = ['DEPTH', 'GR', 'CAL', 'RES', 'RHOB']
unit = ['M', 'gAPI', 'in', 'ohm.m', 'g/cm3']

project.well_data['test'] = {}

project.well_data['test']['data'] = b
project.well_data['test']['units'] = unit
project.well_data['test']['mnemonics'] = mnem

project.shape_check(mnemonics_replacement)


# %%

tw_data,vw_data = preprocessing.well_train_test_split(['7-MP-22-BA','7-MP-50D-BA'],project.well_data)


mega_data = preprocessing.data_assemble(tw_data,'data')
print(np.shape(mega_data))


# %%
a = preprocessing.predict_processing(vw_data,'data')
mega_data = a._remove_dummies(mega_data)
print(np.shape(mega_data))

# %%

data_replacement.settings(method = "linear_regression_simple", path='_ml_project') # for the case there is polynomial 1D
#data_replacement.settings(method = "linear_regression", path='_ml_project', degree = 2)
#data_replacement.settings(method = "DecisionTreeClassifier", path='_ml_project')
#data_replacement.settings(method = "SVM", path='_ml_project')
#data_replacement.settings(method = "LogisticRegression", path='_ml_project')
#data_replacement.settings(method = "KNeighborsClassifier", path='_ml_project')
#data_replacement.settings(method = "RandomForestClassifier", path='_ml_project')
#data_replacement.settings(method = "xgboost_regression", path='_ml_project')


# %%

mnemonics = []
for i in mnemonics_replacement:
    mnemonics.append(i)

print(mnemonics)


# %%

y = mega_data[:,1] # 1 for RHOB 
X = np.delete(mega_data,(1), axis=1) # 1 for RHOB also
print(X)
X = np.array(X, dtype='float') 
y = np.array(y, dtype='float')
print(y)

# %%

data_replacement.fit(X,y,method = "linear_regression_simple", path = "_ml_project")
#data_replacement.fit(X,y,method = "decision_tree", gs = True, path = "_ml_project")
#data_replacement.fit(X,y,method = "SVM", path = "_ml_project")
#data_replacement.fit(X,y,method = "LogisticRegression", path = "_ml_project")
#data_replacement.fit(X,y,method = "KNeighborsClassifier", path = "_ml_project")
#data_replacement.fit(X,y,method = "RandomForestClassifier", path = "_ml_project")
#data_replacement.fit(X,y,method = "xgboost_regression", path = '_ml_project')

# %%
import matplotlib.pyplot as plt
pre_pros = preprocessing.predict_processing(vw_data,'data')
xy_raw = pre_pros.matrix_values()

y_db = {}
x_db = {}
for w in xy_raw:
    #print( xy_raw[w])
    y_db[w] = xy_raw[w][:,1]
    x_db[w] = np.delete(xy_raw[w],(1), axis=1)
    print(x_db)
    ym_db = data_replacement.predict(x_db[w], method = "linear_regression_simple", path = "_ml_project")
    plt.plot(ym_db, y_db[w],'.')
    #plt.plot(ym_db,'.')
    plt.grid()
    plt.show()

# %%

class_db = {}

for well in x_db:
    class_db[well] = data_replacement.predict(x_db[well], method = "xgboost_regression", path = "_ml_project")
    #class_db[well] = machine_learning.predict(x_db[well], method = "GaussianNB", path = "_ml_project")
    #class_db[well] = machine_learning.predict(x_db[well], method = "DecisionTreeClassifier", path = "_ml_project")
    #class_db[well] = machine_learning.predict(x_db[well], method = "SVM", path = "_ml_project")
    #class_db[well] = machine_learning.predict(x_db[well], method = "LogisticRegression", path = "_ml_project")
    #class_db[well] = machine_learning.predict(x_db[well], method = "KNeighborsClassifier", path = "_ml_project")
    #class_db[well] = machine_learning.predict(x_db[well], method = "RandomForestClassifier", path = "_ml_project")
    #class_db[well] = machine_learning.predict(x_db[well], method = "XGBClassifier", path = "_ml_project")
    #class_db[well] = machine_learning.predict(x_db[well], method = "CatBoostClassifier", path = "_ml_project")

pre_pros.return_curve(class_db)
# %%

#a = preprocessing.predict_processing(vw_data,'data')

for w in vw_data:
    x = np.delete(vw_data[w]['data'].T,(1), axis=1)
    y = vw_data[w]['data'].T[:,1]
    print(len(y))
    is_nan_y = []
    is_not_nan_x = []
    for i in range(len(y)):
        if np.isnan(y[i]):
            is_nan_y.append(i)
            #print(i)
        if not np.isnan(x[i]).any():
            is_not_nan_x.append(i)
    is_nan_y = set(is_nan_y)
    is_not_nan_x = set(is_not_nan_x)

    l_data = vw_data[w]['data']
    c_data = a._remove_dummies(l_data)

    intersection = is_not_nan_x.intersection(is_nan_y)
    print(list(intersection))

    n_x_data = []
    y_original = []
    for i in intersection:
        y_original.append(y[i])
        n_x_data.append(x[i])
    predictment = data_replacement.predict(n_x_data, method = "xgboost_regression", path = "_ml_project")
    print("RHOB",len(predictment))
    print(len(c_data[:,1]))
    break

# %%

print(predictment)

# %%
x_r = a.matrix_values()

y_vdb = {}
x_db = {}
for well in x_r:

    print(np.shape(x_r[well]))
    y_vdb[well] = x_r[well][:,0]
    x_db[well] = np.delete(x_r[well],(0), axis=1)
    print(x_db)
    break


# %%

class_db = {}

for x in x_db:
    class_db[x] = data_replacement.predict(x_db[x], method = "GaussianNB", path = "_ml_project")
    class_db[x] = data_replacement.predict(x_db[x], method = "DecisionTreeClassifier", path = "_ml_project")
    class_db[x] = data_replacement.predict(x_db[x], method = "SVM", path = "_ml_project")
    class_db[x] = data_replacement.predict(x_db[x], method = "LogisticRegression", path = "_ml_project")
    class_db[x] = data_replacement.predict(x_db[x], method = "KNeighborsClassifier", path = "_ml_project")
    class_db[x] = data_replacement.predict(x_db[x], method = "RandomForestClassifier", path = "_ml_project")
    class_db[x] = data_replacement.predict(x_db[x], method = "XGBClassifier", path = "_ml_project")

a.return_curve(class_db)


# %%

data_replacement.settings(method = "GaussianNB")
data_replacement.fit(X,y,method = "GaussianNB")

for x in x_db:
    class_db[x] = data_replacement.predict(x_db[x], method = "GaussianNB")
    

# %%
