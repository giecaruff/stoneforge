from stoneforge.data_management.preprocessing import DataLoader

# Tabular example usage
DATA = DataLoader(r"https://github.com/giecaruff/datasets/blob/main/wells/tab/evaluation/teste_tsv.tsv", filetype='tabr', sep="\t", std="US")
del(DATA)

# Las2 example usage
DATA = DataLoader(r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/las2/npra/DP1.las", filetype='las2')
del(DATA)

# Las3 example usage
DATA = DataLoader(r"https://raw.githubusercontent.com/giecaruff/datasets/refs/heads/main/wells/las3/evalutaion/example_las3.las")
del(DATA)