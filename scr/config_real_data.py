'''Spline parameters'''
N_knots = 15
N_const_knots = 4
N_knots_prior = N_knots - N_const_knots

'''Prior parameters (lower limit, upper limit)'''
logD_lims = (-2, 1) 
k_lims = (-2, 0) # Normllay not log scale (1, 5)
spline_lims = (0, 35) # Normally (0, 10)

'''Simulation parameters'''
T = 2000000
dt = 0.1
N = round(T / dt)
saving_freq = 10
Dx = 0.38
min_x = 505
max_x = 545
max_G_0 = 30
max_G_1 = 10
init_xq_range = (510, 540)

'''Summary statistics'''
min_bin = 500
max_bin = 550
num_bins = 20
lag_times = [1, 10, 100, 1000, 10000, 100000]