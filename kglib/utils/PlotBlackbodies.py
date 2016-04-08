from astropy import constants

import numpy as np


def Planck(x, T):
    """
    Return the Planck function at wavelength x (in cm) and temperature T (in K)
    """
    h = constants.h.cgs.value
    c = constants.c.cgs.value
    k = constants.k_B.cgs.value
    pi = np.pi
    return 2 * pi * h * c ** 2 / x ** 5 * 1.0 / (np.exp(h * c / (x * k * T)) - 1.0)


def Planck_nu(nu, T):
    """
    Return the Planck function at frequency nu (in 1/s) and temperature T (in K)
    """
    h = constants.h.cgs.value
    c = constants.c.cgs.value
    k = constants.k_B.cgs.value
    pi = np.pi
    x = h * nu / (k * T)
    return 2 * pi * h * nu ** 3 / c ** 2 * 1.0 / (np.exp(x) - 1.0)


