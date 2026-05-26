from stoneforge.data_management.preprocessing import DataLoader, _download_to_tempfile
from stoneforge.pseudo_wells import anadrill_siliciclastic, lithology_generator
from stoneforge.pseudo_wells.pseudo_tools import merge_lithology
from stoneforge.data_management.preprocessing import resampling
import numpy as np
import pandas as pd

# Tabular example usage
DATA = DataLoader(r"https://github.com/giecaruff/datasets/blob/main/wells/tab/evaluation/teste_tsv.tsv", filetype='tabr', sep="\t", std="US")
del(DATA)

# Las2 example usage
DATA = DataLoader(r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/las2/npra/DP1.las", filetype='las2')
data_las2, units_las2 = DATA.dataframe(DATA.data_obj.data)
del(data_las2)
del(units_las2)
DATA.__del__()
del(DATA)

# Las3 example usage
DATA = DataLoader(r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/las3/evalutaion/example_las3.las")
del(DATA)

# dlis example usage
DATA = DataLoader(r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/dlis/IODP_DSP_leg_96/DSDP_leg_96_hole_616_96_processed_data.dlis")
del(DATA)

# Resampling

markov_chain = np.array(
    [[0.93, 0.07, 0.00, 0.00], # Shale
    [0.02, 0.97, 0.01, 0.00], # Sandstone
    [0.05, 0.10, 0.85, 0.00], # Arcose
    [0.00, 0.00, 0.00, 0.00]] # Sandstone mixed with clay
    )

markov_lithology = lithology_generator.simple(markov_chain,
    lithology_code  = [0,3,4,8],
    sampling = 3000,
    initial_state = 0
)

data_entry = merge_lithology(markov_lithology)

well_1,units_1 = anadrill_siliciclastic(data_entry, top = 800.0, step=0.10, random_state=42)

data_1 = pd.DataFrame.from_dict(well_1)
sub_data_nearest_1 = resampling(data_1, "DEPTH", step = 0.30, top=1000.0, bottom=1100.0, mode="nearest")
