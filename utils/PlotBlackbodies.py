import numpy as np
from astropy import constants


def Planck(x, T):
    h = constants.h.cgs.value
    c = constants.c.cgs.value
    k = constants.k_B.cgs.value
    pi = np.pi
    return 2 * pi * h * c ** 2 / x ** 5 * 1.0 / (np.exp(h * c / (x * k * T)) - 1.0)


def Planck_nu(nu, T):
    h = constants.h.cgs.value
    c = constants.c.cgs.value
    k = constants.k_B.cgs.value
    pi = np.pi
    x = h * nu / (k * T)
    return 2 * pi * h * nu ** 3 / c ** 2 * 1.0 / (np.exp(x) - 1.0)


