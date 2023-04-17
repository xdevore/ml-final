import os, sys
from matplotlib import pyplot as plt
%matplotlib inline
import time
import numpy as np
import pickle
import pandas
import gzip

sys.path.append("..")
import cca_core

def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


with gzip.open("./model_activations/SVHN/model_0_lay03.p", "rb") as f:
    acts1 = pickle.load(f)
    
with gzip.open("./model_activations/SVHN/model_1_lay03.p", "rb") as f:
    acts2 = pickle.load(f)

print(acts1.shape, acts2.shape)

num_datapoints, h, w, channels = acts1.shape
f_acts1 = acts1.reshape((num_datapoints*h*w, channels))

num_datapoints, h, w, channels = acts2.shape
f_acts2 = acts2.reshape((num_datapoints*h*w, channels))

print(f_acts1.shape, f_acts2.shape)

start = time.time()
f_results = cca_core.get_cca_similarity(f_acts1.T, f_acts2.T, epsilon=1e-10, verbose=False)
print('Time: {:.2f} seconds'.format(time.time() - start))
_plot_helper(f_results["cca_coef1"], "CCA Coef idx", "CCA coef value")
print("Mean CCA similarity", np.mean(f_results["cca_coef1"]))
