# %%

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import las2
from stoneforge import preprocessing
from stoneforge.petrophysics import shale_volume
from stoneforge.petrophysics import water_saturation
from io import StringIO
# %%

prj = preprocessing.project("C:\\Users\\mmram\\Downloads")
prj.import_folder()

prj.import_well('3-BRSA-1053-RJS_NMR_SLB_merge')

data = prj.well_data

for mnm in data['3-BRSA-1053-RJS_NMR_SLB_merge']:
    print(mnm)


# %%

print(data['3-BRSA-1053-RJS_NMR_SLB_merge']['MD']['data'])
print(data['3-BRSA-1053-RJS_NMR_SLB_merge']['MD']['unit'])

PHIT = data['3-BRSA-1053-RJS_NMR_SLB_merge']['TCMR']['data']
PHIE = data['3-BRSA-1053-RJS_NMR_SLB_merge']['CMRP_3MS']['data']
# %%

MD = data['3-BRSA-1053-RJS_NMR_SLB_merge']['MD']['data']
VSHALE = shale_volume.vshale('ehigie',phit = PHIT,phie = PHIE)
print(VSHALE)

FF = data['3-BRSA-1053-RJS_NMR_SLB_merge']['CMFF']['data']
SW = water_saturation.water_saturation( method = 'crain', phi = PHIT, ff = FF )
print(SW)

# %%

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 12)

ax[0].plot(VSHALE, MD, color='green')
ax[0].set_title("VSHALE")
ax[0].set_xlabel("-")
ax[0].set_ylabel("DEPTH (m)")
ax[0].invert_yaxis()
ax[0].grid()
ax[1].plot(SW, MD, color='blue')
ax[1].set_title("SW")
ax[1].set_xlabel("-")
ax[1].invert_yaxis()
ax[1].grid()

# %%

data['3-BRSA-1053-RJS_NMR_SLB_merge']['VSHALE'] = {}
data['3-BRSA-1053-RJS_NMR_SLB_merge']['VSHALE']['data'] = VSHALE
data['3-BRSA-1053-RJS_NMR_SLB_merge']['VSHALE']['unit'] = '-'

data['3-BRSA-1053-RJS_NMR_SLB_merge']['SW'] = {}
data['3-BRSA-1053-RJS_NMR_SLB_merge']['SW']['data'] = SW
data['3-BRSA-1053-RJS_NMR_SLB_merge']['SW']['unit'] = '-'

# %%

SDATA = {}
SDATA['well'] = [{'mnemonic': 'NULL', 'unit': '', 'value': '-999.0', 'description': ''}]
SDATA['curve'] = []
ALLDATA = []
for d in data['3-BRSA-1053-RJS_NMR_SLB_merge']:
    mnemonic = d
    unit = data['3-BRSA-1053-RJS_NMR_SLB_merge'][d]['unit']
    l_data = data['3-BRSA-1053-RJS_NMR_SLB_merge'][d]['data']
    SDATA['curve'].append({'mnemonic': mnemonic, 'unit': unit, 'value': '', 'description': ''})
    ALLDATA.append(l_data)
SDATA['data'] = np.array(ALLDATA)
#{'mnemonic': 'DEPT', 'unit': 'M', 'value': '', 'description': ''}

# %%

SDATA

lasfile = StringIO()
las2.write(lasfile, SDATA)
print(lasfile.getvalue())
# %%
