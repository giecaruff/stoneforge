# %%

import numpy as np
import sys
import os

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
project.import_folder()
project.import_several_wells()

#%%

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

print(ref_mnemonics)

def _remove_dummies(data):
    data_1 = np.array(data).T
    data_2 = data_1[~np.isnan(data_1).any(axis=1)]

    return data_2

mega_data = _remove_dummies(mega_data)

print(np.shape(mega_data))
#print(vw_data['7-MP-22-BA']['data'])

# %%

y = mega_data[:,-1]
X = np.delete(mega_data,(-1), axis=1)

# %%

machine_learning.settings(method = "GaussianNB")
machine_learning.settings(method = "DecisionTreeClassifier")
machine_learning.settings(method = "SVM")
machine_learning.settings(method = "LogisticRegression")
machine_learning.settings(method = "KNeighborsClassifier")
machine_learning.settings(method = "RandomForestClassifier")
machine_learning.settings(method = "XGBClassifier")

# %%

a = preprocessing.predict_processing(vw_data,'data')
x_r= a.matrix_values()

y_vdb = {}
x_db = {}
for i in x_r:
    y_vdb[i] = x_r[i][:,-1]
    x_db[i] = np.delete(x_r[i],(-1), axis=1)


# %%

machine_learning.fit(X,y,method = "GaussianNB", path = ".")
#machine_learning.fit(X,y,method = "DecisionTreeClassifier", path = ".")
#machine_learning.fit(X,y,method = "SVM", path = ".")
#machine_learning.fit(X,y,method = "LogisticRegression", path = ".")
#machine_learning.fit(X,y,method = "KNeighborsClassifier", path = ".")
#machine_learning.fit(X,y,method = "RandomForestClassifier", path = ".")
#machine_learning.fit(X,y,method = "XGBClassifier", path = ".")

# %%

class_db = {}

for x in x_db:
    class_db[x] = machine_learning.predict(x_db[x], method = "GaussianNB", path = "",)

a.return_curve(class_db)

# %%
