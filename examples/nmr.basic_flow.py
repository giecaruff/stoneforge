# %%

import numpy as np
import numpy.typing as npt
import pandas as pd
import lasio as las
import matplotlib.pyplot as plt
from stoneforge import preprocessing
from stoneforge.petrophysics import shale_volume
from stoneforge.petrophysics import water_saturation
# %%

prj = preprocessing.project("C:\\Users\\graduando\\Documents\\nmr_data")
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
