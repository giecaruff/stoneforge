# %%

"""
=======================================
Load and crop Dataset Exercise
=======================================

A tutorial exercise about loading well log datasets
"""

from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt


from stoneforge.preprocessing import las2
well = las2.read('../datasets/DP1.las')
#well = las2.read('7-MP-22-BA.las')
DP1_DATA = {} # data information from DP1 welllog


# %%

well_DP1 = {}

for i in range(len(well['curve'])):
    name = well['curve'][i]['mnemonic']
    unit = well['curve'][i]['unit']

    well_DP1[name] = {}
    well_DP1[name]["unit"] = unit
    well_DP1[name]["name"] = name
    well_DP1[name]["data"] = well['data'][i]
    
print(well_DP1)

# %%

# Visualizando as curvas presentes no poço DP1
fig, ax = plt.subplots(1, 4)
fig.set_size_inches(12, 12)

ax[0].plot(well_DP1["CALI"]["data"], well_DP1["DEPT"]["data"], color='black')
ax[0].set_title(well_DP1["CALI"]["name"])
ax[0].set_xlabel(well_DP1["CALI"]["unit"])
ax[0].set_ylabel(well_DP1["DEPT"]["name"] + ' (' + well_DP1["DEPT"]["unit"] + ')')
ax[0].invert_yaxis()
ax[0].grid()

ax[1].plot(well_DP1["DT"]["data"], well_DP1["DEPT"]["data"], color='blue')
ax[1].set_title(well_DP1["DT"]["name"])
ax[1].set_xlabel(well_DP1["DT"]["unit"])
ax[1].invert_yaxis()
ax[1].set_yticklabels([])
ax[1].grid()

ax[2].plot(well_DP1["RHOB"]["data"], well_DP1["DEPT"]["data"], color='red')
ax[2].set_title(well_DP1["RHOB"]["name"])
ax[2].set_xlabel(well_DP1["RHOB"]["unit"])
ax[2].invert_yaxis()
ax[2].set_yticklabels([])
ax[2].grid()

ax[3].plot(well_DP1["NPHI"]["data"], well_DP1["DEPT"]["data"], color='black')
ax[3].set_title(well_DP1["NPHI"]["name"])
ax[3].set_xlabel(well_DP1["NPHI"]["unit"])
ax[3].invert_yaxis()
ax[3].set_yticklabels([])
ax[3].grid()

plt.show()

# %%

well_DP1_c = {}

for i in range(len(well['curve'])):
    name = well['curve'][i]['mnemonic']
    unit = well['curve'][i]['unit']

    well_DP1_c[name] = {}
    well_DP1_c[name]["unit"] = unit
    well_DP1_c[name]["name"] = name
    well_DP1_c[name]["data"] = well['data'][i]

top = 6450
bot = 6690

for n in well_DP1:
    l_data = []
    for i in range(len(well_DP1["DEPT"]["data"])):
        if well_DP1["DEPT"]["data"][i] >= top and well_DP1["DEPT"]["data"][i] < bot:
            l_data.append(well_DP1[n]["data"][i])
        
    well_DP1_c[n]["data"] = np.array(l_data)

fig, ax = plt.subplots(1, 4)
fig.set_size_inches(12, 12)

ax[0].plot(well_DP1_c["CALI"]["data"], well_DP1_c["DEPT"]["data"], color='black')
ax[0].set_title(well_DP1_c["CALI"]["name"])
ax[0].set_xlabel(well_DP1_c["CALI"]["unit"])
ax[0].set_ylabel(well_DP1_c["DEPT"]["name"] + ' (' + well_DP1_c["DEPT"]["unit"] + ')')
ax[0].invert_yaxis()
ax[0].grid()

ax[1].plot(well_DP1_c["DT"]["data"], well_DP1_c["DEPT"]["data"], color='blue')
ax[1].set_title(well_DP1_c["DT"]["name"])
ax[1].set_xlabel(well_DP1_c["DT"]["unit"])
ax[1].invert_yaxis()
ax[1].set_yticklabels([])
ax[1].grid()

ax[2].plot(well_DP1_c["RHOB"]["data"], well_DP1_c["DEPT"]["data"], color='red')
ax[2].set_title(well_DP1_c["RHOB"]["name"])
ax[2].set_xlabel(well_DP1_c["RHOB"]["unit"])
ax[2].invert_yaxis()
ax[2].set_yticklabels([])
ax[2].grid()

ax[3].plot(well_DP1_c["NPHI"]["data"], well_DP1_c["DEPT"]["data"], color='black')
ax[3].set_title(well_DP1_c["NPHI"]["name"])
ax[3].set_xlabel(well_DP1_c["NPHI"]["unit"])
ax[3].invert_yaxis()
ax[3].set_yticklabels([])
ax[3].grid()

plt.show()

# %%

m, n = np.polyfit(well_DP1_c["DEPT"]["data"], well_DP1_c["RHOB"]["data"], 1)

yn = np.polyval([m, n], well_DP1_c["DEPT"]["data"])

plt.plot(well_DP1_c["DEPT"]["data"],well_DP1_c["RHOB"]["data"])
plt.plot(well_DP1_c["DEPT"]["data"],yn)
plt.show()

RHOB_C = well_DP1_c["RHOB"]["data"] - yn + np.mean(well_DP1_c["RHOB"]["data"])

# %%

from stoneforge.petrophysics import porosity
from stoneforge.petrophysics import shale_volume

PHID = porosity.porosity("density", rhob = RHOB_C, rhom = 2.65, rhof = 1.10)
VSH = shale_volume.vshale(gr = well_DP1_c["GR"]["data"], grmin = 0.0, grmax = 150.0, method = "linear")

# %%

PHIN = porosity.porosity("neutron", nphi = well_DP1_c["NPHI"]["data"]/100. + 0.15, vsh = VSH, nphi_sh = 0.481)
#print(well_DP1_c["NPHI"]["data"]/100.)
print(PHIN)

# %%

PHIND = porosity.porosity("neutron-density", phid = PHID, phin = PHIN)
print(PHIND)

# %%

PHIT = porosity.porosity(method = "sonic", dt = well_DP1["DT"]["data"], dtma = 55.5, dtf = 185)
print(PHIT)

# %%

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(12, 12)

ax[0].plot(well_DP1_c["RHOB"]["data"], well_DP1_c["DEPT"]["data"], color='red')
ax[0].plot(yn,well_DP1_c["DEPT"]["data"],color='orange',label="trend")
ax[0].set_title(well_DP1_c["RHOB"]["name"])
ax[0].set_xlabel(well_DP1_c["RHOB"]["unit"])
ax[0].set_ylabel(well_DP1_c["DEPT"]["name"] + ' (' + well_DP1_c["DEPT"]["unit"] + ')')
ax[0].invert_yaxis()
ax[0].grid()
ax[0].legend()

ax[1].plot(RHOB_C, well_DP1_c["DEPT"]["data"], color='red')
ax[1].set_title("RHOB \n simple correction")
ax[1].set_xlabel(well_DP1_c["RHOB"]["unit"])
ax[1].invert_yaxis()
ax[1].set_yticklabels([])
ax[1].grid()

ax[2].plot(PHID, well_DP1_c["DEPT"]["data"], color='red')
ax[2].set_title("PHID")
ax[2].set_xlabel("-")
ax[2].invert_yaxis()
ax[2].set_yticklabels([])
ax[2].grid()

# %%
from stoneforge.petrophysics import water_saturation

# %%

sw = water_saturation.archie()