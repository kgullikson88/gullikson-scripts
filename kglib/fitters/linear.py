"""
Defines a few functions for linear fits
"""

from __future__ import print_function, division, absolute_import

import numpy as np
from astropy.modeling import fitting
from astropy.modeling.polynomial import Chebyshev2D

import statsmodels.api as sm
from statsmodels.robust.norms import TukeyBiweight


def RobustFit(x, y, fitorder=3, weight_fcn=TukeyBiweight(), badregions=None):
    """
    Performs a robust fit (less sensitive to outliers) to x and y

    Parameters:
    ===========
    - x:            numpy.ndarray
                    The x-coordinates of the function to fit

    - y:            numpy.ndarray
                    The y-coordinates of the function to fit

    - fitorder:     integer
                    The order of the fit

    - badregions:   A list of lists
                    Contains wavelength regions to ignore in the fit

    Returns:
    ========
    The fitted y values at each x-point.
    """
    # Re-scale x for stability
    if badregions is None:
        x_train = x
        y_train = y
    else:
        cond = np.any([(x >= reg[0]) & (x <= reg[1]) for reg in badregions], axis=0)
        x_train = x[~cond]
        y_train = y[~cond]
    m, s = x.mean(), x.std()

    x = (x - m) / s
    x_train = (x_train - m) / s
    X = np.ones(x.size)
    X_train = np.ones(x_train.size)
    for i in range(1, fitorder + 1):
        X = np.column_stack((X, x ** i))
        X_train = np.column_stack((X_train, x_train ** i))
    fitter = sm.RLM(y_train, X_train, M=weight_fcn)
    results = fitter.fit()
    return results.predict(X)


def ChebFit2D(x, y, z, x_degree=2, y_degree=2):
    """
    Perform a fit to a 2D polynomial with point z=f(x,y)

    Parameters:
    ===========
    - x:            numpy.ndarray
                    The x-coordinates of the function to fit

    - y:            numpy.ndarray
                    The y-coordinates of the function to fit

    - z:            numpy.ndarray
                    The z-coordinates of the function to fit

    - x_degree:     integer
                    The degree of the chebyshev polynomial in the x direction

    - y_degree:     integer
                    The degree of the chebyshev polynomial in the y direction

    Returns:
    ========
    A callable function that takes new values of (x,y) and returns the fitted z

    """
    p_init = Chebyshev2D(x_degree=x_degree, y_degree=y_degree)
    f = fitting.LinearLSQFitter()

    p = f(p_init, x, y, z)

    return p