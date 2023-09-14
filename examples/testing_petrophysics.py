# %%

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from stoneforge import preprocessing
from stoneforge.petrophysics import shale_volume
from stoneforge.petrophysics import water_saturation
from stoneforge.petrophysics import porosity
from io import StringIO
# %%

prj = preprocessing.project("Y:\\appy_dados\\wells_es")
prj.import_folder()

prj.import_well('6-BRSA-639-ESS_Default_final')

data = prj.well_data

for mnm in data['6-BRSA-639-ESS_Default_final']:
    print(mnm)

# %%
RHOB = data['6-BRSA-639-ESS_Default_final']['RHOB']['data']
LLD = data['6-BRSA-639-ESS_Default_final']['LLD']['data']
PHID = porosity.porosity(method='density',rhob = RHOB, rhof = 1.10, rhom = 2.65)
SW = water_saturation.water_saturation(method='archie',phi = PHID, rt = LLD, rw = 0.0171, a = 1, m = 2, n = 3)

plt.plot(SW)
# %%
