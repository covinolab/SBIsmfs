import os
import sys
sys.path.insert(0, os.path.abspath("../scr"))

from gls_spline import gls_spline_der, gls_spline
import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt 


x_knots = np.linspace(0, 100, 10)
y_knots = np.sin(x_knots)
x_eval = np.linspace(0, 100, 1000)

y_gsl_d = gls_spline_der(x_knots, y_knots, x_eval)
y_gsl = gls_spline(x_knots, y_knots, x_eval)

cs_scipy = ip.CubicSpline(x_knots, y_knots)
y_scp = cs_scipy(x_eval)
y_scp_d = cs_scipy(x_eval, nu=1)

plt.plot(x_eval, y_gsl)
plt.plot(x_eval, y_gsl_d)
plt.plot(x_eval, y_scp)
plt.plot(x_eval, y_scp_d)
plt.plot(x_knots, y_knots, 'ob')
plt.show()
