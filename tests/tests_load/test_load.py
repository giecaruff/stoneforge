from stoneforge import datasets

# Tabular example usage
DATA0 = datasets.test_data()
DATA0.tcsv

# Las2 example usage
DATA1 = datasets.NPRAlaska()
DATA1.ik1.data_obj.data

# Las3 example usage
DATA2 = datasets.test_data()
DATA2.tlas3

# Dlis example usage
DATA3 = datasets.DSDP_leg_96()
DATA3.original