from simulator.simulator import get_simulator_from_config
import torch
import matplotlib.pyplot as plt
q = get_simulator_from_config('test.config')(torch.ones(13))
plt.plot(q)
plt.show()