'''Spline parameters'''
N_knots = 15
N_const_knots = 4
N_knots_prior = N_knots - N_const_knots

'''Prior parameters (lower limit, upper limit)'''
logD_lims = (-2, 1) 
k_lims = (1, 5) # Normllay not log scale (1, 5)
spline_lims = (0, 10) # Normally (0, 10)

'''Simulation parameters'''
T = 50000
dt = 5e-4
N = int(T / dt)
saving_freq = 100
Dx = 1
N_knots = 15
min_x = -6
max_x = 6
max_G_0 = 70
max_G_1 = 30
init_xq_range = (-2, 2)

'''Summary statistics'''
min_bin = -5
max_bin = 5
num_bins = 20
lag_times = [1, 10, 100, 1000, 10000, 100000]