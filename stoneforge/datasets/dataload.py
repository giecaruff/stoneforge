# %%

#from stoneforge.preprocessing import project
from . import las2
import os

def _import_well(fpath,name):
    
    well_data = {}
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, fpath)

    read_data = las2.read(path)

    mnemonic = [a['mnemonic'] for a in read_data['curve']]
    unit = [a['unit'] for a in read_data['curve']]
    well_data[name] = {}

    for i in range(len(mnemonic)):
        well_data[name][mnemonic[i]] = {}
        well_data[name][mnemonic[i]]['data'] = read_data['data'][i]
        well_data[name][mnemonic[i]]['unit'] = unit[i]

    return well_data

# ===================================================================== #

def dp1():

    return _import_well('DP1.las','DP1')

def es1():

    return _import_well('ES1.las','ES1')

def ik1():

    return _import_well('IK1.las','IK1')

def wd1():

    return _import_well('WD1.las','WD1')

# %%