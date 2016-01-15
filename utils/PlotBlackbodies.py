import sys
from scipy.interpolate import UnivariateSpline

import pylab
import numpy as np
from astropy import constants, units


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


if __name__ == "__main__":
    T1 = float(sys.argv[1])
    T2 = float(sys.argv[2])
    if T2 > T1:
        T = T1
        T2 = T1
        T1 = T

    if len(sys.argv) > 3:
        # Read in a filter file
        #Filter format is two columns: One for wavelength in nanometer
        # and the other for transmission (as a fraction, not a percent)
        filterfile = sys.argv[3]
        x, y = np.loadtxt(filterfile, unpack=True)
        FILTER = UnivariateSpline(x, y, s=0)
        filt = True
        first = x[0]
        last = x[-1]
    else:
        filt = False
        first = 100
        last = 10000

    x = np.arange(first, last, 1)
    xcm = x * units.nm.to(units.cm)
    y1 = Planck(xcm, T1)
    y2 = Planck(xcm, T2)

    # Filter?
    if filt:
        y1 = y1 * FILTER(x)
        y2 = y2 * FILTER(x)

    pylab.figure(1)
    pylab.loglog(x, y1 / y1.max(), label=str(T1) + " K Blackbody")
    pylab.loglog(x, y2 / y1.max(), label=str(T2) + " K Blackbody")
    #pylab.title("Blackbody Spectra")
    pylab.xlabel("Wavelength (nm)")
    pylab.ylabel("Normalized Flux")
    pylab.legend(loc='best')

    pylab.figure(2)
    pylab.loglog(x, y1 / y2, 'k-')
    pylab.title("Blackbody Flux Ratio")
    pylab.xlabel("Wavelength (nm)")
    pylab.ylabel("Flux Ratio")

    pylab.show()
