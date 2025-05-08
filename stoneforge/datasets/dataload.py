# %%

#from stoneforge.preprocessing import project
# this maybe be redundant because of preprocessing
from . import las2
import os
import pandas as pd

def _import_well(fpath):
    
    #well_data = {}
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, fpath)

    read_data = las2.read(path)

    mnemonic = [a['mnemonic'] for a in read_data['curve']]
    unit = [a['unit'] for a in read_data['curve']]
    #well_data[name] = {}
    well_data = {}
    units = {}

    for i in range(len(mnemonic)):
        well_data[mnemonic[i]] = read_data['data'][i]
        units[mnemonic[i]] = unit[i]
        #well_data[name][mnemonic[i]] = {}
        #well_data[name][mnemonic[i]]['data'] = read_data['data'][i]
        #well_data[name][mnemonic[i]]['unit'] = unit[i]

    df = pd.DataFrame(well_data)

    #return well_data
    return (df,units)

# ===================================================================== #

def dp1():

    return _import_well('DP1.las')

def es1():

    return _import_well('ES1.las')

def ik1():

    return _import_well('IK1.las')

def wd1():

    return _import_well('WD1.las')

# %%