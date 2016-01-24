import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from astropy import constants, units
from utils import FittingUtilities
from scipy.signal import fftconvolve
import warnings

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


"""
  Here is the main, optimized function. It is stolen shamelessly from
  the avsini.c implementation given in the SPECTRUM code distribution.

  A wrapper called 'Broaden' with my standard calls is below
"""
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1] convolve(np.ndarray[DTYPE_t, ndim=1] y,
                                             np.ndarray[DTYPE_t, ndim=1] ys,
                                             long num,
                                             long nd,
                                             double st,
                                             double dw,
                                             double vsini,
                                             double u):
  cdef double beta, gam, w, dlc, c1, c2, dv, r1, r2, v
  cdef long i, n1, n2, n
  cdef double clight = 299791.0
  cdef DTYPE_t f, t, s

  beta = (1.0-u)/(1.0-u/3.0)
  gam = u/(1.0-u/3.0)

  #End effect
  n1 = nd + 1
  ys[1:n1] = y[1:n1]
  n2 = num - nd - 1
  ys[n2:num+1] = y[n2:num+1]
  if vsini < 0.5:
    return y

  #Convolve with rotation profile
  w = st + (n1 - 1)*dw
  for n in range(n1, n2+1):
    w = w+dw
    s = 0.0
    t = 0.0
    dlc = w*vsini/clight
    c1 = 0.63661977*beta/dlc;
    c2 = 0.5*gam/dlc;
    dv = dw/dlc;

    for i in range(-nd, nd+1):
      v = i*dv
      r2 = 1.0 - v**2
      if r2 > 0.0:
        f = c1*sqrt(r2) + c2*r2
        t += f
        s += f*y[n+i]
    ys[n] = s/t
  
  return ys



"""
  This is the wrapper function. The user should call this!
"""

def Broaden_old(model, vsini, epsilon=0.5, linear=False, findcont=False):
  """
    model:           xypoint instance with the data (or model) to be broadened
    vsini:           the velocity (times sin(i) ) of the star, in cm/s
    epsilon:          Linear limb darkening. I(u) = 1-epsilon + epsilon*u
    linear:          flag for if the x-spacing is already linear. If true, we don't need to linearize
    findcont:        flag to decide if the continuum needs to be found
  """

  if not linear:
      x = np.linspace(model.x[0], model.x[-1], model.size())
      model = FittingUtilities.RebinData(model, x)
  else:
      x = model.x
  if findcont:
        model.cont = FittingUtilities.Continuum(model.x, model.y, lowreject=1.5, highreject=10)

  #Need to prepare a few more variables before calling 'convolve'
  broadened = model.copy()
  num = model.size()
  ys = np.ones(num)
  start = model.x[0]
  dwave = model.x[1] - model.x[0]

  s2 = (start + num*dwave)*vsini/(dwave*constants.c.cgs.value)
  vsini *= units.cm.to(units.km)
  nd = s2 + 5.5

  broadened.y = convolve(model.y, ys, num, nd, start, dwave, vsini, epsilon)
  return broadened



def Broaden(model, vsini, epsilon=0.5, linear=False, findcont=False):
    """
      model:           xypoint instance with the data (or model) to be broadened
      vsini:           the velocity (times sin(i) ) of the star, in cm/s
      epsilon:          Linear limb darkening. I(u) = 1-epsilon + epsilon*u
      linear:          flag for if the x-spacing is already linear in log-spacing. If true, we don't need to linearize
      findcont:        flag to decide if the continuum needs to be found
    """
    c = constants.c.cgs.value

    if not linear:
        x0 = model.x[0]
        x1 = model.x[-1]
        x = np.logspace(np.log10(x0), np.log10(x1), model.size())
        model = FittingUtilities.RebinData(model, x)
    else:
        x = model.x
    if findcont:
        model.cont = FittingUtilities.Continuum(model.x, model.y, lowreject=1.5, highreject=10)

    # Make the broadening kernel.
    dx = np.log(x[1]/x[0])
    lim = vsini/c
    if lim < dx:
        #vsini is too small. Don't broaden
        warnings.warn("vsini too small ({}). Not broadening!".format(vsini))
        return model.copy()
    #d_logx = np.arange(-lim, lim, dx*np.log10(200.0))
    d_logx = np.arange(0.0, lim, dx)
    d_logx = np.r_[-d_logx[::-1][:-1], d_logx]
    alpha = 1.0 - (d_logx/lim)**2
    B = (1.0-epsilon)*np.sqrt(alpha) + epsilon*np.pi*alpha/4.0 #Broadening kernel
    B /= B.sum()  #Normalize

    # Do the convolution
    broadened = model.copy()
    broadened.y = fftconvolve(model.y, B, mode='same')

    return broadened
