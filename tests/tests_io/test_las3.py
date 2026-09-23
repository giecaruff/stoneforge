from stoneforge.io.las3 import LAS3Parser
from stoneforge.data_management.preprocessing import _download_to_tempfile

url = r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/las3/evalutaion/example_las3.las"
path = _download_to_tempfile(url)

data_test = LAS3Parser(path)
data_test.force_association()
