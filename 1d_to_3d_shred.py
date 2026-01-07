import torch
import random
import numpy as np

print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
# If available:
device = torch.device("mps")
x = torch.ones(1, device=device)
print(x)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


import numpy as np


import math
np.math = math
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os


RDE_dataset = np.load(os.path.join("./high_fidelity_sim", "interpolated_combined.npy"))
grid_points = np.load(os.path.join("./high_fidelity_sim", "cylindrical_grid.npy"), allow_pickle=True).item()


# Make this into the folder
%cd sindy-shred

import sindy
import sindy_shred
from processdata import load_data
from processdata import TimeSeriesDataset



num_sensors = 1000
lags = 10
# load_X = load_data('SST')
load_X = RDE_dataset
n = load_X.shape[0]
m = load_X.shape[1]
print(" n = ", n, " m = ", m)
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)