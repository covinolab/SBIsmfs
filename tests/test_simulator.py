import os, sys
sys.path.insert(0, os.path.abspath("../scr"))

import numpy as np
import matplotlib.pyplot as plt
import torch

from simulator import smfe_simulator, smfe_simulator_mm

parameters = torch.tensor(
    [0, 3, 6.94227994, -0.67676768, -4.23232323, -3.72438672, 0.45021645,
     2.48196248, 0.45021645, 3.72438672, -4.23232323, -0.67676768,  6.94227994]
)

q_stats = smfe_simulator(parameters)

print('Summary statistics : ', q_stats, sep='\n')
print('Shape of statistics : ', q_stats.size(), sep='\n')

q_stats = smfe_simulator_mm(parameters)

print('Summary statistics : ', q_stats, sep='\n')
print('Shape of statistics : ', q_stats.size(), sep='\n')
