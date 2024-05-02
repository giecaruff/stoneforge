# %% ============================================================== #
# Maybe this function should be put in another place

from . import las2
import pandas as pd

def import_well(path):

    read_data = las2.read(path)

    mnemonic = [a['mnemonic'] for a in read_data['curve']]
    unit = [a['unit'] for a in read_data['curve']]
    well_data = {}
    units = {}

    for i in range(len(mnemonic)):
        well_data[mnemonic[i]] = read_data['data'][i]
        units[mnemonic[i]] = unit[i]

    df = pd.DataFrame(well_data)

    return (df,units)

# %%