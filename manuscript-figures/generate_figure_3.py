import os
import pickle

import src.wavefinder as wf

data_path = os.path.join(os.getcwd(), "manuscript-figures/data/")
output_path = os.path.join(os.getcwd(), "manuscript-figures/output")

# Import Data for figure 1 -------------------------------------------------------------------

file = os.path.join(data_path, "figure3.pkl")
with open(file, 'rb') as handle:
    (cases, deaths) = pickle.load(handle)

wf.plot_peaks([cases, deaths], 'Ghana', True, output_path)
