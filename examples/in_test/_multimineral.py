import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stoneforge.data_management.preprocessing import DataLoader, DataManager

# Manual Access:
las2 = DataLoader(r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/las2/npra/IK1.las", filetype='las2')
print('header itens:',las2.data_obj.header.keys())

# Example (Manual Access): Accessing data as DataFrame
data_las2, units_las2 = las2.dataframe(las2.data_obj.data)
data_las2 = data_las2.replace(-999.0, np.nan)

# Adding facies
IK1 = DataManager(las2, depth="DEPT")

IK1.add_facie(name="LEDGE_SANDSTONE", top=10619, bottom=10842)

# View Facies LEDGE_SANDSTONE interval
IK1.LEDGE_SANDSTONE

# Solução de sistema (teste simplificado)

DEPT = np.array(IK1.LEDGE_SANDSTONE['DEPT'])
GR = np.array(IK1.LEDGE_SANDSTONE['GR'])
DT = np.array(IK1.LEDGE_SANDSTONE['DT'])
NPHI = np.array(IK1.LEDGE_SANDSTONE['NPHI'])
RHOB = np.array(IK1.LEDGE_SANDSTONE['RHOB'])

# A0 = Matriz de valores tabelados

A0 = np.array(
    [
        [20,11,111,160,0.0001], # GR
        [2.650,2.710,2.657,2.56,1.100], # RHOB
        [0.000,0.000,48.1,40.0,100.0], # NPHI
        [55.5,47.8,100,130,185], # DT
        [1.0,1.0,1.0,1.0,1.0] # 1
    ]
)

# Demais operações:

AI0 = np.linalg.inv(A0)

X0 = []
for i in range (len(GR)):
    B0 = np.array([GR[i],RHOB[i],NPHI[i],DT[i],1.0],float)
    x = np.dot(AI0,B0)
    if i == 0:
        print("Valores encontrados:",B0)
        print("Proporção da composição:",x)
    X0.append(x)
X0 = np.array(X0)

# A1 = Matriz de valores tabelados

A1 = np.array(
    [
    [20,11,111,160], # GR
    [2.650,2.710,2.657,2.56], # RHOB
    [0.000,0.000,48.1,40.0], # NPHI
    [55.5,47.8,100,130], # DT
    [1.0,1.0,1.0,1.0] # 1
    ]
    )

# Demais operações:

AA1 = np.dot(A1.T,A1)
AI1 = np.linalg.inv(AA1)
AT1 = np.dot(A1,AI1)

X1 = []
for i in range (len(GR)):
    B1 = np.array([GR[i],RHOB[i],NPHI[i],DT[i],1.0],float)
    x = np.dot(B1,AT1)
    if i == 0:
        print("Valores encontrados:",B1)
        print("Proporção da composição:",x)
    X1.append(x)
X1 = np.array(X1)

# A2 = Matriz de valores tabelados

A2 = np.array([
    #[20,11,111,160,0.0001], # GR
    [2.650,2.710,2.657,2.56,1.10], # RHOB
    [0.000,0.000,48.1,40.0,100.00], # NPHI
    [55.5,47.8,100,130,185], # DT
    [1.0,1.0,1.0,1.0,1.0] # 1
])

# Demais operações:

AA2 = np.dot(A2,A2.T)
AI2 = np.linalg.inv(AA2)
AT2 = np.dot(AI2,A2)

X2 = []
for i in range (len(GR)):
    B2 = np.array([RHOB[i],NPHI[i],DT[i],1.0],float)
    x = np.dot(B2,AT2)
    if i == 0:
        print("Valores encontrados:",B2)
        print("Proporção da composição:",x)
    X2.append(x)
X2 = np.array(X2)

class Elan:

    def __init__(self,md,gr,rhob,nphi,dt,lito = False):
        self.md = md
        self.lito = lito #if lito.any:
        self.m = len(md)

        self.litho_code = {
            57:"green",
            49:"yellow",
            54:"maroon",
            25:"grey"
            }

        ones_log = np.ones(self.m).T

        # valores na ordem: (quartzo, calcita, lama, arcóseo, fluido)
        gr_values = np.array([20,11,111,160,0.0001])
        dt_values = np.array([55.5,47.8,100,130,185])
        rhob_values = np.array([2.650,2.710,2.657,2.56,1.100])
        nphi_values = np.array([0.000,0.000,48.1,40.0,100.0])
        ones_values = np.array([1.0,1.0,1.0,1.0,1.0])

        self.matrix = np.array([gr_values,dt_values,rhob_values,nphi_values,ones_values])
        self.elems = {'qtz':0,'cal':1,'shl':2,'ark':3,'fld':4}
        self.mnems = {'gr':0,'dt':1,'rhob':2,'nphi':3,'ones':4}
        self.datst = np.array([gr,dt,rhob,nphi,ones_log],float)
        self.names = ['quartzo','calcita','lama','arcóseo','fluido']
        self.elem_colors = ['#eaec61','#6fb5db','#438d8e','orange','navy']
        self.mnem_colors = ['green','black','red','blue']
        self.mnems_names = ['GR','DT','RHOB','NPHI']
        self.mnesm_units = ['API','us/ft','g/cm3','v/v']

    def matrix_crop(self,mnems = [],elems = []):

        self.mnems_val = [self.mnems[x] for x in mnems]
        self.elems_val = [self.elems[x] for x in elems]

        set5 = set(range(5))

        mnems_list = set5 - set(set5 - set(self.mnems_val))
        elems_list = set5 - set(set5 - set(self.elems_val))

        matrix = self.matrix
        submatrix = matrix[np.ix_(self.mnems_val,self.elems_val)]

        datst = self.datst[np.ix_(self.mnems_val)]

        return (submatrix,mnems_list,elems_list,datst)

    # ======================================================================#


    def system_sol(self, info, reg = 0.0, min_r = False):

        A0 = info[0]
        datst = info[3]
        self.ssol_mnems = info[1]
        self.ssol_elems = info[2]

        X0 = []
        AI0 = np.linalg.inv(A0 + (np.eye(A0.shape[0])*reg))

        for i in range (self.m):
            B0 = datst[:,i]
            x = np.dot(AI0,B0)
            X0.append(x)
        X0 = np.array(X0)

        if min_r:
            X0 = self._min_r(X0)

        self.ssol_x0 = X0

    def min_qd(self, info, reg = 0.0, min_r = False):

        A1 = info[0]
        datst = info[3]
        self.mnqd_mnems = info[1]
        self.mnqd_elems = info[2]

        AA1 = np.dot(A1.T,A1)
        AI1 = np.linalg.inv(AA1 + (np.eye(AA1.shape[0])*reg))
        AT1 = np.dot(A1,AI1)

        X1 = []
        for i in range (self.m):
            B1 = datst[:,i]
            x = np.dot(B1,AT1)
            X1.append(x)
        X1 = np.array(X1)

        if min_r:
            X1 = self._min_r(X1)

        self.mnqd_x1 = X1

    def moore_pen(self, info, reg = 0.0, min_r = False):

        A2 = info[0]
        datst = info[3]
        self.mrpen_mnems = info[1]
        self.mrpen_elems = info[2]

        AA2 = np.dot(A2,A2.T)
        AI2 = np.linalg.inv(AA2 + (np.eye(AA2.shape[0])*reg))
        AT2 = np.dot(AI2,A2)

        X2 = []
        for i in range (len(GR)):
            B2 = datst[:,i]
            x = np.dot(B2,AT2)
            X2.append(x)
        X2 = np.array(X2)

        if min_r:
            X2 = self._min_r(X2)

        self.mrpen_x2 = X2

    # ======================================================================#

    def _min_r(self,X0):

        m,n = np.shape(X0)
        aux = np.copy(X0)
        for i in range(n):
            X0[:,i] = aux[:,i] - np.min(aux[:,i])

        return X0
    
# Elan Code:
EE = Elan(DEPT,GR,RHOB,NPHI,DT)

#A0 = EE.matrix_crop(['gr','rhob','ones'],['qtz','shl','fld']) # fld, qtz, shl cal, ark
A0 = EE.matrix_crop(['rhob','gr','ones'],['fld', 'qtz', 'shl'])

EE.system_sol(A0, reg = 0.30, min_r = True)