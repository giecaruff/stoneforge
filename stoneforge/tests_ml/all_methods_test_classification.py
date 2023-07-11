# %%

import numpy as np
import sys
import os
import pandas
import matplotlib.pyplot as plt

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
    #'CAL':['CAL','DCAL','HCAL','CALI'],
    #'RHOB':["RHOB","RHLA","RHBA","RHLA3","RHBA4"],
    'RES':["ILD","HDRS","RT","AHT901","AT90","RT90"],
    'DT':["DT"],
    #'NPHI':['NPHI'],
    'Lithology':['Lith_new']
}

ref_mnemonics = list(mnemonics_replacement.keys())

project.data_replacement(mnemonics_replacement)
project.convert_into_matrix(ref_mnemonics)
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

GR = X[:,1]
print(GR, len(GR))

# %%
# Ordem crescente
oGR = np.array(sorted(GR))
print(oGR)

# %%

moGR = []
for i in range(1,len(oGR)):
    moGR.append( (oGR[i - 1] + oGR[i])/2. )
print(len(moGR),moGR)

# %%

mxGR = []
mnGR = []
n = 26000 # 26000
print(moGR[n])
for i in range(len(oGR)):
    v = oGR[i]
    if v < moGR[n]:
        mnGR.append(i)
    if v > moGR[n]:
        mxGR.append(i)

print(len(mnGR),mnGR)
print(len(mxGR),mxGR)

# %%

mxl = y[mxGR]
mnl = y[mnGR]
print(mnl, len(mnl))
print(mxl, len(mxl))

def dict_count(values, perc = 1):

    val = list(set(values))
    n = len(values)
    f_dict = {}
    for v in val:
        data = []
        for i in values:
            if i == v:
                data.append(1)
        f_dict[v] = round(perc*sum(data)/n,4)

    return f_dict

vmnl = dict_count(mnl)
vmxl = dict_count(mxl)
print(vmnl)
print(vmxl)

# %%

def gini (dict):

    sm = []
    for k in dict:
        dv = dict[k]
        sm.append(dv*(1 - dv))

    return sum(sm)

gmnl = gini(vmnl)
gmxl = gini(vmxl)

print(gmnl)
print(gmxl)

# %%

def h_calc(ne,nd,nm,ge,gd):

    return ( (ne/nm)*ge + (nd/nm)*gd )

h_val = h_calc(len(mnl),len(mxl),len(oGR),gmnl,gmxl)

print(h_val)

# %%

def dict_count(values, perc = 1):

    val = list(set(values))
    n = len(values)
    f_dict = {}
    for v in val:
        data = []
        for i in values:
            if i == v:
                data.append(1)
        f_dict[v] = round(perc*sum(data)/n,4)

    return f_dict

# ================================================== #

def gini (dict):

    sm = []
    for k in dict:
        dv = dict[k]
        sm.append(dv*(1 - dv))

    return sum(sm)

# ================================================== #

def sch_calculate(X_data):

    n = len(X_data[:,0])
    
    shape = np.shape(X_data)
    for j in range(shape[1]):
        oGR = X_data[:,j]
        
        for i in range(1,len(oGR)):
            v = (oGR[i - 1] + oGR[i])/2.
            mxGR = []
            mnGR = []
            for ii in range(len(oGR)):
                if v < oGR[ii]:
                    mnGR.append(ii)
                if v > oGR[ii]:
                    mxGR.append(ii)

            mxl = y[mxGR]
            mnl = y[mnGR]
            
            vmnl = dict_count(mnl)
            vmxl = dict_count(mxl)
                
            gmnl = gini(vmnl)
            gmxl = gini(vmxl)

            h_val = h_calc(len(mnl),len(mxl),len(oGR),gmnl,gmxl)
            #moGR.append( (oGR[i - 1] + oGR[i])/2. )

        print(X_data[:,i])

#sch_calculate(X)

# %%
