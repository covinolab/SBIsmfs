import os
import sys
sys.path.insert(0, os.path.abspath("../scr"))

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt 

from gls_random_gen import gaussian_rn


N = 1000000
samples = gaussian_rn(N)

_ = plt.hist(samples, bins=100)

mean = np.mean(samples)
std = np.std(samples)
skew = stats.skew(samples)
kurtosis = stats.kurtosis(samples)

results = {'Mean : ': mean,
           'Standard deviation : ': std,
           'Skewness : ': skew,
           'Kurtosis -3.0 : ': kurtosis
           }

for name, value in results.items():
    print(name, value, end='\n')