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

#project = preprocessing.project("D:\\appy_projetos\\wells")
project = preprocessing.project("C:\\Users\\joseaugustodias\\Desktop\\pocos")
project.import_folder()
project.import_several_wells()

print(project.well_names_paths)

#Na proxima janela tem que aparecer o histograma
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

#Histograma
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
#Tamanho teste x treino 

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
#Opção se vai querer ou não fazer a redimensionamento do dado

from sklearn.preprocessing import LabelEncoder

y = mega_data[:,-1]
X = np.delete(mega_data,(-1), axis=1)
#X = np.array(X, dtype='float')
#y = np.array(y, dtype='int')

y1 = [49., 25., 54., 57., 49.]
y2 = [25., 54., 57., 49.]
y3 = [25., 57., 49.]

tr3 = [0, 3, 1]
#tr3 = [0, 2, 1]

le = LabelEncoder()
print(le.fit_transform(y1))
print(le.fit_transform(y2))
print(le.fit_transform(y3))
print(le.fit_transform(tr3))


# %%
#Aqui o usuario precisa escolher 1. Escolher o métodos, 2. se ele vai querer utilizar o pseudo automatização; 3.Se não quiser, ajustar os parametros que ele quer utlizar
machine_learning.settings(method = "GaussianNB", path='_ml_project')
machine_learning.settings(method = "DecisionTreeClassifier", path='_ml_project',max_depth = 3, criterion = 'gini')
#machine_learning.settings(method = "SVM", path='_ml_project')
#machine_learning.settings(method = "LogisticRegression", path='_ml_project')
machine_learning.settings(method = "KNeighborsClassifier", path='_ml_project', n_neighbors = np.arange(3,61,2), weights = uniform , p = np.arange(1,6))
#achine_learning.settings(method = "RandomForestClassifier", path='_ml_project', n_estimators =  100, learning_rate = 0.5, max_depth =  5 )
#machine_learning.settings(method = "XGBClassifier", path='_ml_project', n_estimators =  100, learning_rate = 0.5, max_depth =  5 ))
#machine_learning.settings(method = "CatBoostClassifier", path='_ml_project')


# %%

a = preprocessing.predict_processing(vw_data,'data')
x_r = a.matrix_values()

y_vdb = {}
x_db = {}
for i in x_r:
    y_vdb[i] = x_r[i][:,-1]
    x_db[i] = np.delete(x_r[i],(-1), axis=1)

# %%
#Rodar o metodo

#machine_learning.fit(X,y,method = "GaussianNB", path = "_ml_project")
machine_learning.fit(X,y,method = "DecisionTreeClassifier", path = "_ml_project")
#machine_learning.fit(X,y,method = "SVM", path = "_ml_project")
#machine_learning.fit(X,y,method = "LogisticRegression", path = "_ml_project")
#machine_learning.fit(X,y,method = "KNeighborsClassifier", gs = True, path = "_ml_project")
#machine_learning.fit(X,y,method = "RandomForestClassifier", gs = True, path = "_ml_project")
#machine_learning.fit(X,y,method = "XGBClassifier",gs = True, path = "_ml_project")
#machine_learning.fit(X,y,method = "CatBoostClassifier",gs = True, path = "_ml_project")


# %%

class_db = {}

for x in x_db:
    #class_db[x] = machine_learning.predict(x_db[x], method = "GaussianNB", path = "_ml_project")
    class_db[x] = machine_learning.predict(x_db[x], method = "DecisionTreeClassifier", path = "_ml_project")
    #class_db[x] = machine_learning.predict(x_db[x], method = "SVM", path = "_ml_project")
    #class_db[x] = machine_learning.predict(x_db[x], method = "LogisticRegression", path = "_ml_project")
    #class_db[x] = machine_learning.predict(x_db[x], method = "KNeighborsClassifier", path = "_ml_project")
    #class_db[x] = machine_learning.predict(x_db[x], method = "RandomForestClassifier", path = "_ml_project")
    #class_db[x] = machine_learning.predict(x_db[x], method = "XGBClassifier", path = "_ml_project")
    #class_db[x] = machine_learning.predict(x_db[x], method = "CatBoostClassifier", path = "_ml_project")


a.return_curve(class_db)



# %%

machine_learning.settings(method = "GaussianNB")
machine_learning.fit(X,y,method = "GaussianNB")

for x in x_db:
    class_db[x] = machine_learning.predict(x_db[x], method = "GaussianNB")
    

# %%

print(y,len(y))

# %%

class_db['7-MP-50D-BA']

# %%

aa = a.return_curve(class_db)

print(class_db['7-MP-22-BA'],len(class_db['7-MP-22-BA']))
print(y_vdb['7-MP-22-BA'],len(y_vdb['7-MP-22-BA']))
# %%

a = 0
for i in range(len(class_db['7-MP-22-BA'])):
    if class_db['7-MP-22-BA'][i] == y_vdb['7-MP-22-BA'][i]:
        a += 1

print(a)

# %%
