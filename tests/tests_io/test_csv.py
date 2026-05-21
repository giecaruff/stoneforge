from stoneforge.io.tabr import TABParser
from stoneforge.data_management.preprocessing import _download_to_tempfile

url = r"https://github.com/giecaruff/datasets/blob/main/wells/tab/evaluation/teste_tsv.tsv"
path = _download_to_tempfile(url)

data = TABParser(path)