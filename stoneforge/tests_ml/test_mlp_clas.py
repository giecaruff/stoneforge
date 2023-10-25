#  %%

import numpy as np
import sys
import os
import pandas 

from sklearn import datasets
from sklearn.model_selection import train_test_split

if __package__:
    from ..mlp_classification import *
else:
    print('passed ml')
    sys.path.append(os.path.dirname(__file__) + '/..')
    import mlp_classification

if __package__:
    from ..preprocessing import *
else:
    print('passed mgt')
    sys.path.append(os.path.dirname(__file__) + '/..')
    import preprocessing

# %%
#project = preprocessing.project("C:\\Users\\josea\OneDrive\\Área de Trabalho\pocoss")
project = preprocessing.project('C:\\Users\\joseaugustodias\\Desktop\\pocos')
project.import_folder()
project.import_several_wells()

print(project.well_names_paths)

#%%

mnemonics_replacement = {
    #'DEPTH':['DEPTH',"MD"],
    'GR':['GR'],
    'CAL':['CAL','DCAL','HCAL','CALI'],
    'RHOB':["RHOB","RHLA","RHBA","RHLA3","RHBA4"],
    #'RES':["ILD","LLD","HDRS","RT","AHT901","AT90","RT90"],
    'DT':["DTCO","DT"],
    'NPHI':['NPHI'],
    'LITO':['LITO', 'Lith_new']
}

ref_mnemonics = list(mnemonics_replacement.keys())

project.data_replacement(mnemonics_replacement)
project.convert_into_matrix(ref_mnemonics)



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
mlp_classification.settings(method = "MLPClassifier", path='_ml_project')

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

mlp_classification.fit(X,y,method = "MLPClassifier", path = "_ml_project")

# %%

class_db = {}

for well in x_db:
    class_db[well] = mlp_classification.predict(x_db[well], method = "MLPClassifier", path = "_ml_project")
pre_pros.return_curve(class_db)

# %%
class_db = {}

mlp_classification.settings(method = "MLPClassifier")
mlp_classification.fit(X,y,method = "MLPClassifier")

for well in x_db:
    class_db[well] = mlp_classification.predict(x_db[well], method = "MLPClassifier")


# %%

mlp_classification.evaluation(class_db['7-MP-50D-BA'],y_db['7-MP-50D-BA'],decimals=5)

# %%

print(class_db['7-MP-50D-BA'])
print(y_db['7-MP-50D-BA'])

# %%
print(list(set(y_db['7-MP-50D-BA'])))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
confusion_matrix(class_db['7-MP-50D-BA'], y_db['7-MP-50D-BA'], labels = np.array(list(set(y_db['7-MP-50D-BA']))))

# %%
print(classification_report(class_db['7-MP-50D-BA'], y_db['7-MP-50D-BA'] ,labels = np.array(list(set(y_db['7-MP-50D-BA'])))))
# %%
