from stoneforge.io.las2 import LAS2Parser, read, write
from stoneforge.data_management.preprocessing import _download_to_tempfile

url = r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/las2/npra/DP1.las"
path = _download_to_tempfile(url)

data_test = LAS2Parser(path)
data = read(path)
saves = write("test.las",data)