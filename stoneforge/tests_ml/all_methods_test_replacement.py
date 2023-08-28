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
project = preprocessing.project("Y:\\appy_dados\\wells_es")
project.import_folder()
project.import_several_wells()

print(project.well_names_paths)


#%%

mnemonics_replacement = {
    'DEPTH':['DEPTH','MD'],
    #'GR':['GR'],
    #'CAL':['CAL','DCAL','HCAL','CALI'],
    'RHOB':["RHOB","RHOZ","RHLA","RHBA","RHLA3","RHBA4"],
    #'RES':["ILD","LLD","HDRS","RT","AHT901","AT90","RT90"],
    'NPHI':['NPHI','NPOR'],
    'DT':['DT','DTCO']
}

ref_mnemonics = list(mnemonics_replacement.keys())

project.data_replacement(mnemonics_replacement)
project.convert_into_matrix(ref_mnemonics)

# %%

tw_data,vw_data = preprocessing.well_train_test_split(['3-BRSA-277-ESS_Default_final','3-SHEL-24-ESS_Default_final'],project.well_data)


mega_data = preprocessing.data_assemble(tw_data,'data')
print(np.shape(mega_data))


# %%
a = preprocessing.predict_processing(vw_data,'data')
mega_data = a._remove_dummies(mega_data)
print(np.shape(mega_data))

# %%

#data_replacement.settings(method = "linear_regression_simple", path='_ml_project') # for the case there is polynomial 1D
#data_replacement.settings(method = "linear_regression", path='_ml_project', degree = 2)
#data_replacement.settings(method = "DecisionTreeClassifier", path='_ml_project')
#data_replacement.settings(method = "SVM", path='_ml_project')
#data_replacement.settings(method = "LogisticRegression", path='_ml_project')
#data_replacement.settings(method = "KNeighborsClassifier", path='_ml_project')
#data_replacement.settings(method = "RandomForestClassifier", path='_ml_project')
data_replacement.settings(method = "xgboost_regression", path='_ml_project')


# %%

y = mega_data[:,-1] # 1 for RHOB
print('aaa',mega_data[:,-1])
X = np.delete(mega_data,(-1), axis=1) # 1 for RHOB also
X = np.delete(X,(0), axis=1) # 1 for RHOB also
print(X)
X = np.array(X, dtype='float') 
y = np.array(y, dtype='float')
print(y)

# %%

#data_replacement.fit(X,y,method = "linear_regression_simple", path = "_ml_project")
#data_replacement.fit(X,y,method = "decision_tree", gs = True, path = "_ml_project")
#data_replacement.fit(X,y,method = "SVM", path = "_ml_project")
#data_replacement.fit(X,y,method = "LogisticRegression", path = "_ml_project")
#data_replacement.fit(X,y,method = "KNeighborsClassifier", path = "_ml_project")
#data_replacement.fit(X,y,method = "RandomForestClassifier", path = "_ml_project")
data_replacement.fit(X,y,method = "xgboost_regression", path = '_ml_project')

# %%
import matplotlib.pyplot as plt
pre_pros = preprocessing.predict_processing(vw_data,'data')
xy_raw = pre_pros.matrix_values()

y_db = {}
x_db = {}
for w in xy_raw:
    y_db[w] = xy_raw[w][:,-1]
    x_db[w] = np.delete(xy_raw[w],(-1), axis=1)
    x_db[w] = np.delete(x_db[w],(0), axis=1)
    print(x_db)
    print(y_db)
    ym_db = data_replacement.predict(x_db[w], method = "xgboost_regression", path = "_ml_project")
    plt.plot(ym_db, y_db[w],'.')
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

class_db_full = pre_pros.return_curve(class_db)
print(class_db_full)
# %%

for w in class_db_full:
    print(list(class_db_full[w]))
    break



# %%
