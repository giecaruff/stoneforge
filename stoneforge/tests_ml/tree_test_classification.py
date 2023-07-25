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
# middle elements for subdivision

def middle_calc(property):
    oGR = np.array(sorted(property))
    moGR = []
    for i in range(1,len(oGR)):
        moGR.append( (oGR[i - 1] + oGR[i])/2. )
    #print(len(moGR),moGR)
    return moGR

moGR = middle_calc(GR)

middle_calc(GR)

# %%
# separation of original dataset into two based onto the GR value
# 26000 is related to the 68.4075 API

def data_sep(property,middle_values,pos):
    mxGR = []
    mnGR = []

    for i in range(len(property)):
        v = property[i]
        if v < middle_values[pos]:
            mnGR.append(i)
        if v > middle_values[pos]:
            mxGR.append(i)

    return (mnGR,mxGR)

mnGR,mxGR = data_sep(GR,moGR,26000)

# %%
# counting the proportion of facies for each subdivision

mxl = y[mxGR]
mnl = y[mnGR]
print(mnl, len(mnl))
print(mxl, len(mxl))

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

vmnl = dict_count(mnl)
vmxl = dict_count(mxl)
print(vmnl)
print(vmxl)

# %%
# Calculating the gini index for each subdivision

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
# final calculation for those specific subdivision

def h_calc(ne,nd,nm,ge,gd):

    return ( (ne/nm)*ge + (nd/nm)*gd )

h_val = h_calc(len(mnl),len(mxl),len(oGR),gmnl,gmxl)

print(h_val)

# %%

class Tree:

    def __init__(self, X, y):

        self.X = X
        self.y = y

        self.x = 0.
        self.mx = 0.

    def single_curve_cal(self, pos, div_pos, visualization = False):
        x = X[:,pos]
        ox = np.array(sorted(x))
        mox = self._middle_calc(x)
        mnx,mxx = self._data_sep(x,mox,div_pos)
        

        mxl = y[mxx]
        mnl = y[mnx]
        if visualization:
            print(mnl, len(mnl))
            print(mxl, len(mxl))

        vmnl = self._dict_count(mnl)
        vmxl = self._dict_count(mxl)

        if visualization:
            print(vmnl)
            print(vmxl)
        gmnl = self._gini(vmnl)
        gmxl = self._gini(vmxl)

        h_val = self._h_calc(len(mnl),len(mxl),len(ox),gmnl,gmxl)
        if visualization:
            print(h_val)
        
    def _middle_calc(self,property):
        ox = np.array(sorted(property))
        mox = []
        for i in range(1,len(ox)):
            mox.append( (ox[i - 1] + ox[i])/2. )
        return mox
    
    def _data_sep(self,property,middle_values,pos):
        mxGR = []
        mnGR = []

        for i in range(len(property)):
            v = property[i]
            if v < middle_values[pos]:
                mnGR.append(i)
            if v > middle_values[pos]:
                mxGR.append(i)

        return (mnGR,mxGR)
    
    def _dict_count(self, values, perc = 1):

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
    
    def _gini(self,dict):

        sm = []
        for k in dict:
            dv = dict[k]
            sm.append(dv*(1 - dv))

        return sum(sm)
    
    def _h_calc(self,ne,nd,nm,ge,gd):

        return ( (ne/nm)*ge + (nd/nm)*gd )
    
# %%

a = Tree(X,y)
a.single_curve_cal(1, 26000)

# %%

A = []
for i in range(len(X[:,0])):
    A.append(a.single_curve_cal(1, i))
# %%

print(A)

# %%
