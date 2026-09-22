from stoneforge.io.tabr import TABParser
from stoneforge.data_management.preprocessing import _download_to_tempfile

url = r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/tab/evaluation/synth_toc_1300_3000.csv"
path = _download_to_tempfile(url)

data = TABParser(path)
