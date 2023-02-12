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

project = preprocessing.project("D:\\appy_projetos\\pocos")
project.import_folder()
project.import_several_wells()

#%%

mnemonics_replacement = {
    'DEPTH':['DEPTH'],
    'GR':['GR'],
    'CAL':['CAL','DCAL','HCAL','CALI'],
    'RHOB':["RHOB","RHOZ","RHLA","RHBA","RHLA3","RHBA4"],
    'RES':["ILD","HDRS","RT","AHT901","AT90","RT90"],
    'NPHI':['NPHI']
}
project.data_replacement(mnemonics_replacement)
project.convert_into_matrix()

# %%

for i in project.well_data:
    #print(np.shape(project.well_data[i]['data']))
    print(i)
    break

print(project.well_data['4-BRSA-879D-BA'])
print(project.well_data['4-BRSA-879D-BA']['data'][0])
#print(np.shape(project.well_data['4-BRSA-879D-BA']['data']))

b = np.delete(project.well_data['4-BRSA-879D-BA']['data'], -2, axis=0)
print(b,np.shape(b))

mnem = ['DEPTH', 'GR', 'CAL', 'RES', 'RHOB']
unit = ['M', 'gAPI', 'in', 'ohm.m', 'g/cm3']

project.well_data['test'] = {}

project.well_data['test']['data'] = b
project.well_data['test']['units'] = unit
project.well_data['test']['mnemonics'] = mnem

print(project.well_data)

project.shape_check(mnemonics_replacement)

# %%

#print(project.well_names_las)

def well_training_split(well_names,well_database):

    all_wells = set(well_database.keys())
    v_wells = set(well_names)
    t_wells = all_wells - v_wells

    t_database = {}
    for w in list(t_wells):
        t_database[w] = well_database[w]

    v_database = {}
    for w in list(v_wells):
        v_database[w] = well_database[w]

    return (t_database,v_database)

tw_data,vw_data = well_training_split(['7-MP-22-BA','7-MP-50D-BA'],project.well_data)

print(vw_data)

# %%
