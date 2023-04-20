# %%

import numpy as np
import sys
import os
import pandas 

from sklearn import datasets
from sklearn.model_selection import train_test_split

if __package__:
    from ..machine_learning import *
else:
    print('passed ml')
    sys.path.append(os.path.dirname(__file__) + '/..')
    import machine_learning

if __package__:
    from ..preprocessing import *
else:
    print('passed mgt')
    sys.path.append(os.path.dirname(__file__) + '/..')
    import preprocessing

# %%
project = preprocessing.project("D:\\appy_projetos\\wells")
#project = preprocessing.project("C:\\Users\\joseaugustodias\\Desktop\\pocos")
project.import_folder()
project.import_several_wells()

print(project.well_names_paths)

#%%

mnemonics_replacement = {
    'DEPTH':['DEPTH'],
    'GR':['GR'],
    'CAL':['CAL','DCAL','HCAL','CALI'],
    'RHOB':["RHOB","RHLA","RHBA","RHLA3","RHBA4"],
    'RES':["ILD","HDRS","RT","AHT901","AT90","RT90"],
    'NPHI':['NPHI'],
    'Lithology':['Lith_new']
}

ref_mnemonics = list(mnemonics_replacement.keys())

project.data_replacement(mnemonics_replacement)
project.convert_into_matrix(ref_mnemonics)

# %%

well_test = np.delete(project.well_data['7-MP-33D-BA']['data'], -2, axis=0)
print(well_test,np.shape(well_test))

mnem = ['DEPTH', 'GR', 'CAL', 'RES', 'RHOB']
unit = ['M', 'gAPI', 'in', 'ohm.m', 'g/cm3']

project.well_data['test'] = {}

project.well_data['test']['data'] = well_test
project.well_data['test']['units'] = unit
project.well_data['test']['mnemonics'] = mnem

project.shape_check(mnemonics_replacement)

# %%
# split dataset by well

tw_data,vw_data = preprocessing.well_train_test_split(['7-MP-22-BA','7-MP-50D-BA'],project.well_data)

mega_data = preprocessing.data_assemble(tw_data,'data')
print(np.shape(mega_data))

# %%

print(ref_mnemonics)

def _remove_dummies(data):
    data_1 = np.array(data).T
    data_2 = data_1[~np.isnan(data_1).any(axis=1)]

    return data_2

mega_data = _remove_dummies(mega_data)

print(np.shape(mega_data))

y = mega_data[:,-1]
X = np.delete(mega_data,(-1), axis=1)

# %%

lito_values = [{
    "name": "ARENITO",
    "short_name": "ARN",
    "code": 49,
    "patch_property": {
        "color": "#ffff3f",
        "hatch": ".",
        "alpha": 1.0,
        "hatchcolor": "#000000"
    }
    },
    {
    "name": "ARENITO ARGILOSO",
    "short_name": "ARL",
    "code": 25,
    "patch_property": {
        "color": "#7eff00",
        "hatch": ".",
        "alpha": 1.0,
        "hatchcolor": "#000000"
    }
    }]

for i in lito_values:
    print(i["code"])

print(project.class_counts(y,class_dict = lito_values))

# %%

machine_learning.settings(method = "GaussianNB", path='_ml_project')
machine_learning.settings(method = "DecisionTreeClassifier", path='_ml_project')
machine_learning.settings(method = "SVM", path='_ml_project')
machine_learning.settings(method = "LogisticRegression", path='_ml_project')
machine_learning.settings(method = "KNeighborsClassifier", path='_ml_project')
machine_learning.settings(method = "RandomForestClassifier", path='_ml_project')
machine_learning.settings(method = "XGBClassifier", path='_ml_project')
machine_learning.settings(method = "CatBoostClassifier", path='_ml_project')

# %%

pre_pros = preprocessing.predict_processing(vw_data,'data')
xy_raw = pre_pros.matrix_values()

y_db = {}
x_db = {}
for well in xy_raw:
    y_db[well] = xy_raw[well][:,-1]
    x_db[well] = np.delete(xy_raw[well],(-1), axis=1)

# %%
for w in tw_data:
    print(w)

machine_learning.fit(X,y,method = "GaussianNB", path = "_ml_project")
#machine_learning.fit(X,y,method = "DecisionTreeClassifier", gs = True, path = "_ml_project")
#machine_learning.fit(X,y,method = "DecisionTreeClassifier", path = "_ml_project")
#machine_learning.fit(X,y,method = "SVM", path = "_ml_project")
#machine_learning.fit(X,y,method = "LogisticRegression", path = "_ml_project")
#machine_learning.fit(X,y,method = "KNeighborsClassifier", gs = True, path = "_ml_project")
#machine_learning.fit(X,y,method = "RandomForestClassifier", gs = True, path = "_ml_project")
#machine_learning.fit(X,y,method = "XGBClassifier",gs = True, path = "_ml_project")

#machine_learning.fit(X,y,method = "CatBoostClassifier",gs = True, path = "_ml_project")
#machine_learning.fit(X,y,method = "CatBoostClassifier", path = "_ml_project")

# %%

class_db = {}

for well in x_db:
    class_db[well] = machine_learning.predict(x_db[well], method = "GaussianNB", path = "_ml_project")
    class_db[well] = machine_learning.predict(x_db[well], method = "DecisionTreeClassifier", path = "_ml_project")
    class_db[well] = machine_learning.predict(x_db[well], method = "SVM", path = "_ml_project")
    class_db[well] = machine_learning.predict(x_db[well], method = "LogisticRegression", path = "_ml_project")
    class_db[well] = machine_learning.predict(x_db[well], method = "KNeighborsClassifier", path = "_ml_project")
    class_db[well] = machine_learning.predict(x_db[well], method = "RandomForestClassifier", path = "_ml_project")
    class_db[well] = machine_learning.predict(x_db[well], method = "XGBClassifier", path = "_ml_project")
    class_db[well] = machine_learning.predict(x_db[well], method = "CatBoostClassifier", path = "_ml_project")

pre_pros.return_curve(class_db)

# %%

class_db = {}

machine_learning.settings(method = "DecisionTreeClassifier")
machine_learning.fit(X,y,method = "DecisionTreeClassifier")

for well in x_db:
    class_db[well] = machine_learning.predict(x_db[well], method = "DecisionTreeClassifier")

# %%

machine_learning.evaluation(class_db['7-MP-50D-BA'],y_db['7-MP-50D-BA'],decimals=5)

# %%

print(class_db['7-MP-50D-BA'])
print(y_db['7-MP-50D-BA'])

# %%
