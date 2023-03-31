# %%
### Validation tests for ml methods

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
### Importing well data

#project = preprocessing.project("D:\\appy_projetos\\wells")
project = preprocessing.project("C:\\Users\\joseaugustodias\\Desktop\\pocos")
project.import_folder()
project.import_several_wells()

print("project data paths:",project.well_names_paths)


#%%
### Setting mnemonics for machine learning (choosing mnemonics and replace similar ones)

mnemonics_replacement = {
    'DEPTH':['DEPTH'],
    'GR':['GR'],
    'CAL':['CAL','DCAL','HCAL','CALI'],
    'RHOB':["RHOB","RHOZ","RHLA","RHBA","RHLA3","RHBA4"],
    'RES':["ILD","HDRS","RT","AHT901","AT90","RT90"],
    'NPHI':['NPHI'],
    'Lithology':['Lith_new']
}

ref_mnemonics = list(mnemonics_replacement.keys())

print("mnemonics of reference",ref_mnemonics)

project.data_replacement(mnemonics_replacement)
project.convert_into_matrix(ref_mnemonics)

# %%
### Spliting data (by wells) into training (tw_data) and validation (vw_data)
### also incorporating all the training data into a single matrix

tw_data,vw_data = preprocessing.well_train_test_split(['7-MP-22-BA','7-MP-50D-BA'],project.well_data)

mega_data = preprocessing.data_assemble(tw_data,'data')
print("training matrix shape:",np.shape(mega_data))

# %%
### Removing dummies from data

def _remove_dummies(data):
    data_1 = np.array(data).T
    data_2 = data_1[~np.isnan(data_1).any(axis=1)]

    return data_2

mega_data = _remove_dummies(mega_data)

print("training matrix shape (without dummies):",np.shape(mega_data))

# %%
### Selecting lithology (target) and well log data (forecasters) for training

y = mega_data[:,-1]
X = np.delete(mega_data,(-1), axis=1)

# %%
### preprocessing data for training

data_processing = preprocessing.predict_processing(vw_data,'data')
xy_raw = data_processing.matrix_values()

y_v = {} # target data for validation
x_v = {} # forecasters data for validation
for well in xy_raw:
    y_v[well] = xy_raw[well][:,-1]
    x_v[well] = np.delete(xy_raw[well],(-1), axis=1)

# %%
###  Validation, training of data and classification

machine_learning.validation(X, y, random_state = 2,n_splits = 10, path = "_ml_project")
machine_learning.settings(method = "GaussianNB", path='_ml_project')
machine_learning.fit(X,y,method = "GaussianNB", path = "_ml_project")

class_db = {}
for x in x_v:
    class_db[x] = machine_learning.predict(x_v[x], method = "GaussianNB", path = "_ml_project")

# %%
### Method evaluation after classification

print("lithology for classified 7-MP-22-BA well",class_db['7-MP-22-BA'],len(class_db['7-MP-22-BA']))
print("lithology for original 7-MP-22-BA well",y_v['7-MP-22-BA'],len(y_v['7-MP-22-BA']))

# %%

print(class_db['7-MP-22-BA'],len(class_db['7-MP-22-BA']))
print(y_v['7-MP-22-BA'],len(y_v['7-MP-22-BA']))
# %%

a = 0
for i in range(len(class_db['7-MP-22-BA'])):
    if class_db['7-MP-22-BA'][i] == y_v['7-MP-22-BA'][i]:
        a += 1

print(a)


machine_learning.evaluation(y,y,"_ml_project")

#print(json_dict)
# %%
len(a.return_curve(class_db)['7-MP-50D-BA'])
# %%