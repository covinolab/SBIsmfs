import pytest
import numpy as np
from sbi_smfs.utils.gsl_spline import c_spline, c_spline_der


def test_cspline():
    x_knots = np.linspace(-6, 6, 1000)
    y_knots = np.sin(x_knots)
    x_axis = np.linspace(-3, 3, 10)
    y_axis = c_spline(x_knots, y_knots, x_axis)
    assert np.allclose(y_axis, np.sin(x_axis))


def test_cspline_der():
    x_knots = np.linspace(-6, 6, 1000)
    y_knots = np.sin(x_knots)
    x_axis = np.linspace(-3, 3, 10)
    y_axis = c_spline_der(x_knots, y_knots, x_axis)
    assert np.allclose(y_axis, np.cos(x_axis))


def test_cspline_error():
    x_knots = np.linspace(-6, 6, 1000)
    y_knots = np.sin(x_knots)
    x_axis = np.linspace(-3, 10, 10)
    y_axis = c_spline(x_knots, y_knots, x_axis)
    assert y_axis is None


def test_cspline_der_error():
    x_knots = np.linspace(-6, 6, 1000)
    y_knots = np.sin(x_knots)
    x_axis = np.linspace(-3, 10, 10)
    y_axis = c_spline_der(x_knots, y_knots, x_axis)
    assert y_axis is None
