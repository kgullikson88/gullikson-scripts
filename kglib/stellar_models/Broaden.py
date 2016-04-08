"""
  This is a general function for all broadening. Importing this
  will let the user do rotational broadening, macroturbulent 
  broadening, and Gaussian broadening (reducing the resolution)
"""
from __future__ import print_function, division, absolute_import

from scipy.special import erf  # Error function
from scipy.signal import fftconvolve

import numpy as np
from astropy import constants, units

from .RotBroad_Fast import Broaden as RotBroad


def MacroBroad(data, vmacro, extend=True):
    """
      This broadens the data by a given macroturbulent velocity.
    It works for small wavelength ranges. I need to make a better
    version that is accurate for large wavelength ranges! Sorry
    for the terrible variable names, it was copied from
    convol.pro in AnalyseBstar (Karolien Lefever)

    Parameters:
    ===========
    -data:     kglib.utils.DataStructures.xypoint instance
               Stores the data to be broadened. The data MUST
               be equally-spaced before calling this!

    -vmacro:   float
               The macroturbulent velocity, in km/s

    -extend:   boolean
               If true, the y-axis will be extended to avoid edge-effects

    Returns:
    ========
    A broadened version of data.
    """
    # Make the kernel
    c = constants.c.cgs.value * units.cm.to(units.km)
    sq_pi = np.sqrt(np.pi)
    lambda0 = np.median(data.x)
    xspacing = data.x[1] - data.x[0]
    mr = vmacro * lambda0 / c
    ccr = 2 / (sq_pi * mr)

    px = np.arange(-data.size() / 2, data.size() / 2 + 1) * xspacing
    pxmr = abs(px) / mr
    profile = ccr * (np.exp(-pxmr ** 2) + sq_pi * pxmr * (erf(pxmr) - 1.0))

    # Extend the xy axes to avoid edge-effects, if desired
    if extend:

        before = data.y[-profile.size / 2 + 1:]
        after = data.y[:profile.size / 2]
        extended = np.r_[before, data.y, after]

        first = data.x[0] - float(int(profile.size / 2.0 + 0.5)) * xspacing
        last = data.x[-1] + float(int(profile.size / 2.0 + 0.5)) * xspacing
        x2 = np.linspace(first, last, extended.size)

        conv_mode = "valid"

    else:
        extended = data.y.copy()
        x2 = data.x.copy()
        conv_mode = "same"

    # Do the convolution
    newdata = data.copy()
    newdata.y = fftconvolve(extended, profile / profile.sum(), mode=conv_mode)

    return newdata
  

  
