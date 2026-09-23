from stoneforge.io.dlist import DLISAccess
from stoneforge.data_management.preprocessing import _download_to_tempfile

url = r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/dlis/IODP_DSP_leg_96/DSDP_leg_96_hole_616_96_processed_data.dlis"
path = _download_to_tempfile(url)

data_test = DLISAccess(path)
data_test.show_header()
mns = data_test.mnemonics()
data = data_test.extract(mnemonics=mns)
#data_test.export_csv(data)