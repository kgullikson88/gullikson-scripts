"""
    Just a set of helper functions that I use often
    VERY miscellaneous!
"""
from __future__ import print_function, absolute_import

import os
import logging

from astropy import units as u, constants
from astropy import convolution
from scipy.optimize import bisect
from scipy.stats import scoreatpercentile
from scipy.signal import kaiserord, firwin, lfilter
from scipy.interpolate import InterpolatedUnivariateSpline as spline, UnivariateSpline
from astropy.io import fits as pyfits
import numpy as np
from astropy.time import Time
from statsmodels.stats.proportion import proportion_confint
import pandas as pd

from kglib.utils import DataStructures
from kglib.spectral_type import SpectralTypeRelations
from kglib.utils import readmultispec as multispec


try:
    import emcee
    emcee_import = True
except ImportError:
    print("Warning! emcee module not loaded! BayesFit Module will not be available!")
    emcee_import = False
from kglib.utils import FittingUtilities
try:
    import mlpy
    mlpy_import = True
except ImportError:
    print( "Warning! mlpy not loaded! Denoise will not be available")
    mlpy_import = False
import warnings


def ensure_dir(f):
    """
      Ensure that a directory exists. Create if it doesn't
    """
    d = os.path.dirname(f)
    if d == "":
        d = f
    if not os.path.exists(d):
        os.makedirs(d)


def SmoothData(order, windowsize=91, smoothorder=5, lowreject=3, highreject=3,
               numiters=10, expand=0, normalize=True):
    """
    Denoise the data, and smooth the result with an iterative savitzy-golay filter.
    Returns the smoothed result, which can be subtracted from the original for
    unsharp-masking.

    Parameters:
    ===========
    - order:         kglib.utils.DataStructures.xypoint instance
                     The data to smooth. The variable is called 'order'
                     because I deal with echelle orders.

    - windowsize:    odd integer
                     The size of the savitzy-golay window, in pixels

    - smoothorder:   integer
                     The order of the savitzy-golay filter. Higher
                     gives a closer representation of the data, but can
                     go crazy if you go too high.

    - lowreject:     integer
                     In each iteration, reject pixels less than lowreject
                     standard deviations below the fitted line.

    - highreject:    integer
                     In each iteration, reject pixels more than highreject
                     standard deviations above the fitted line.

    - numiters:      integer
                     The number of rejection iterations to do

    - expand:        integer
                     The number of pixels to the left and right of a bad
                     pixel (which is defined by lowreject and highreject)
                     that we will reject.

    - normalize:     boolean
                     If true, the output has a maximum value of 1.0

    Returns:
    ========
    kglib.utils.DataStructures.xypoint instance with the smoothed data.
    """
    denoised = Denoise(order.copy())
    denoised.y = FittingUtilities.Iterative_SV(denoised.y,
                                               windowsize,
                                               smoothorder,
                                               lowreject=lowreject,
                                               highreject=highreject,
                                               numiters=numiters,
                                               expand=expand)
    if normalize:
        denoised.y /= denoised.y.max()
    return denoised


def astropy_smooth(data, vel, linearize=False, kernel=convolution.Gaussian1DKernel, **kern_args):
    """
    Smooth using a gaussian filter, using astropy.

    Parameters:
    ===========
    - data:          kglib.utils.DataStructures.xypoint instance
                     The data to smooth.

    - vel:           float
                     The velocity scale to smooth out.
                     Can either by an astropy quantity or a float in km/s

    - linearize:     boolean
                     If True, we will put the data in a constant
                     log-wavelength spacing grid before smoothing.
                     The output has the same spacing as the input
                     regardless of this variable.

    - kernel:        astropy.convolution kernel
                     The astropy kernel to use. The default is the
                     Gaussian1DKernel.

    - kern_args:     Additional kernel arguments beyond width

    Returns:
    ========
    A smoothed version of the data, on the same wavelength grid as the data
    """

    if linearize:
        original_data = data.copy()
        datafcn = spline(data.x, data.y, k=3)
        linear = DataStructures.xypoint(data.x.size)
        linear.x = np.logspace(np.log10(data.x[0]), np.log10(data.x[-1]), linear.size())
        linear.y = datafcn(linear.x)
        data = linear

    # Figure out feature size in pixels
    if not isinstance(vel, u.quantity.Quantity):
        vel *= u.km / u.second

    featuresize = (vel / constants.c).decompose().value
    dlam = np.log(data.x[1] / data.x[0])
    Npix = featuresize / dlam

    # Make kernel and smooth
    kern = kernel(Npix, **kern_args)
    smoothed = convolution.convolve(data.y, kern, boundary='extend')

    if linearize:
        fcn = spline(data.x, smoothed)
        return fcn(original_data.x)
    return smoothed


def roundodd(num):
    """
    Round the given number to the nearest odd number.
    """
    rounded = round(num)
    if rounded % 2 != 0:
        return rounded
    else:
        if rounded > num:
            return rounded - 1
        else:
            return rounded + 1



def BinomialErrors_old(nobs, Nsamp, alpha=0.16):
    """
    DO NOT USE THIS! Use BinomialErrors instead!

    One sided confidence interval for a binomial test.

    If after Nsamp trials we obtain nobs
    trials that resulted in success, find c such that

    P(nobs/Nsamp < mle; theta = c) = alpha

    where theta is the success probability for each trial.

    Code stolen shamelessly from stackoverflow:
    http://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
    """
    warnings.warn('Use BinomialErrors instead!', DeprecationWarning)
    from scipy.stats import binom

    p0 = float(nobs) / float(Nsamp)
    upper_errfcn = lambda c: binom.cdf(nobs, Nsamp, c) - alpha
    lower_errfcn = lambda c: binom.cdf(nobs, Nsamp, c) - (1.0 - alpha)
    return p0, bisect(lower_errfcn, 0, 1), bisect(upper_errfcn, 0, 1)


def BinomialErrors(nobs, Nsamp, alpha=0.05, method='jeffrey'):
    """
    This is basically just statsmodels.stats.proportion.proportion_confint
    with a different default method. It also returns the proportion nobs/Nsamp

    Parameters:
    ===========
    - nobs:     integer
                The number of "successes"

    - Nsamp:    integer
                The total number of trials. Should be >= nobs.

    - alpha:    float in (0, 1)
                Probability that the true value lies outside the
                resulting error (or something like that).
                alpha=0.05 is about 2-sigma.

    - method:   string
                The calculation method. This is just passed to
                `statsmodels.stats.proportion.proportion_confint`

    Returns:
    ========
    - prob:     float
                The estimate for the probability. prob = nobs / Nsamp

    - low:      float
                The lower bound on the probability

    - high:     float
                The upper bound on the probability
    """

    low, high = proportion_confint(nobs, Nsamp, method=method, alpha=alpha)

    if nobs == 0:
        low = 0.0
        p = 0.0
    elif nobs == Nsamp:
        high = 1.0
        p = 1.0
    else:
        p = float(nobs) / float(Nsamp)

    return p, low, high


def GetSurrounding(full_list, value, return_index=False):
    """
    Takes a list and a value, and returns the two list elements
      closest to the value
    If return_index is True, it will return the index of the surrounding
      elements rather than the elements themselves
    """
    sorter = np.argsort(full_list)
    full_list = sorted(full_list)
    closest = np.argmin([abs(v - value) for v in full_list])
    next_best = closest - 1 if full_list[closest] > value or closest == len(full_list) - 1 else closest + 1
    if return_index:
        return sorter[closest], sorter[next_best]
    else:
        return full_list[closest], full_list[next_best]


def ReadExtensionFits(datafile):
    """
      A convenience function for reading in fits extensions without needing to
      give the name of the standard field names that I use. The standard field
      names are :
          - 'wavelength'
          - 'flux'
          - 'continuum'
          - 'error'
    """
    return ReadFits(datafile,
                    extensions=True,
                    x="wavelength",
                    y="flux",
                    cont="continuum",
                    errors="error")


def ReadFits(datafile, errors=False, extensions=False, x=None, y=None, cont=None, return_aps=False, debug=False):
    """
    Read a fits file. If extensions=False, it assumes IRAF's multispec format.
    Otherwise, it assumes the file consists of several fits extensions with
    binary tables, with the table names given by the x,y,cont, and errors keywords.

    See ReadExtensionFits for a convenience function that assumes my standard names

    Parameters:
    ===========
    - datafile:       string
                      The name of the file to read

    - errors:         boolean, integer, or string
                      If False, indicates there are no errors in the datafile
                      If an integer AND extensions = False, indicates the index
                      that the errors are found in.
                      If a string AND extenstions = True, indicates the name
                      of the error field in the binary table.

    - extensions:     boolean
                      Is the data stored in several fits extensions? If not,
                      we assume it is in multispec format

    - x:              string
                      The name of the field with the x-coordinate (wavelength, etc).
                      Ignored if extensions = False

    - y:              string
                      The name of the field with the y-coordinate (flux, etc).
                      Ignored if extensions = False

    - cont:           string
                      The name of the field with the continuum estimate.
                      Ignored if extensions = False

    - return_aps:     boolean
                      Return the aperture wavelength fields as well as the
                      extracted orders. The wavelength fields define the
                      wavelengths in multispec format. Ignored if extensions = True.

    - debug:          boolean
                      Print some extra information to screen

    Returns:
    ========
    A list of kglib.utils.DataStructures.xypoint instances with the data for each
    echelle order.
    """
    if debug:
        print( "Reading in file %s: " % datafile)

    if extensions:
        # This means the data is in fits extensions, with one order per extension
        # At least x and y should be given (and should be strings to identify the field in the table record array)
        if type(x) != str:
            x = raw_input("Give name of the field which contains the x array: ")
        if type(y) != str:
            y = raw_input("Give name of the field which contains the y array: ")
        orders = []
        hdulist = pyfits.open(datafile)
        if cont == None:
            if not errors:
                for i in range(1, len(hdulist)):
                    data = hdulist[i].data
                    xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y))
                    orders.append(xypt)
            else:
                if type(errors) != str:
                    errors = raw_input("Give name of the field which contains the errors/sigma array: ")
                for i in range(1, len(hdulist)):
                    data = hdulist[i].data
                    xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y), err=data.field(errors))
                    orders.append(xypt)
        elif type(cont) == str:
            if not errors:
                for i in range(1, len(hdulist)):
                    data = hdulist[i].data
                    xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y), cont=data.field(cont))
                    orders.append(xypt)
            else:
                if type(errors) != str:
                    errors = raw_input("Give name of the field which contains the errors/sigma array: ")
                for i in range(1, len(hdulist)):
                    data = hdulist[i].data
                    xypt = DataStructures.xypoint(x=data.field(x), y=data.field(y), cont=data.field(cont),
                                                  err=data.field(errors))
                    orders.append(xypt)

    else:
        # Data is in multispec format rather than in fits extensions
        # Call Rick White's script
        try:
            retdict = multispec.readmultispec(datafile, quiet=not debug)
        except ValueError:
            warnings.warn("Wavelength not found in file %s. Using a pixel grid instead!" % datafile)
            hdulist = pyfits.open(datafile)
            data = hdulist[0].data
            hdulist.close()
            numpixels = data.shape[-1]
            numorders = data.shape[-2]
            wave = np.array([np.arange(numpixels) for i in range(numorders)])
            retdict = {'flux': data,
                       'wavelen': wave,
                       'wavefields': np.zeros(data.shape)}

        # Check if wavelength units are in angstroms (common, but I like nm)
        hdulist = pyfits.open(datafile)
        header = hdulist[0].header
        hdulist.close()
        wave_factor = 1.0  #factor to multiply wavelengths by to get them in nanometers
        for key in sorted(header.keys()):
            if "WAT1" in key:
                if "label=Wavelength" in header[key] and "units" in header[key]:
                    waveunits = header[key].split("units=")[-1]
                    if waveunits == "angstroms" or waveunits == "Angstroms":
                        # wave_factor = u.nm/u.angstrom
                        wave_factor = u.angstrom.to(u.nm)
                        if debug:
                            print( "Wavelength units are Angstroms. Scaling wavelength by ", wave_factor)

        if errors == False:
            numorders = retdict['flux'].shape[0]
        else:
            numorders = retdict['flux'].shape[1]
        orders = []
        for i in range(numorders):
            wave = retdict['wavelen'][i] * wave_factor
            if errors == False:
                flux = retdict['flux'][i]
                err = np.ones(flux.size) * 1e9
                err[flux > 0] = np.sqrt(flux[flux > 0])
            else:
                if type(errors) != int:
                    errors = int(raw_input("Enter the band number (in C-numbering) of the error/sigma band: "))
                flux = retdict['flux'][0][i]
                err = retdict['flux'][errors][i]
            cont = FittingUtilities.Continuum(wave, flux, lowreject=2, highreject=4)
            orders.append(DataStructures.xypoint(x=wave, y=flux, err=err, cont=cont))
        if return_aps:
            # Return the aperture wavefields too
            orders = [orders, retdict['wavefields']]
    return orders


def OutputFitsFileExtensions(column_dicts, template, outfilename, mode="append", headers_info=[], primary_header={}):
    """
    Function to output a fits file in my standard format.

    Parameters:
    ===========
    - column_dicts:        list of dictionaries
                           Maps the name of the column to a numpy.ndarray with
                           the data. Example of a column would be the wavelength
                           or flux at each pixel. Each list item should describe
                           a different echelle order.

    - template             string
                           The filename of the template fits file. The header will
                           be taken from this file and used as the main header of
                           the new file.

    - mode                 string
                           Determines how the outputted file is made. mode = "append"
                           will just add a fits extension to the existing file
                           (and then save it as outfilename). mode = "new" will
                           create an entirely new fits file.

    - header_info          list of lists
                           Each sub-list should have size 2 where the first element
                           is the name of the new keyword, and the second element
                           is the corresponding value. A 3rd element may be added as
                           a comment. The length of the top-level list should be the
                           same as the length of column_dicts. This header information
                           gets put into the extension headers, so can contain stuff
                           relevant for that echelle order only.

    - primary_header       dictionary
                           Keywords to insert into the primary fits header. The key
                           will be the header key, and the value will be the header
                           value.
    """

    # Do some error checking.
    if not isinstance(column_dicts, list):
        column_dicts = [column_dicts, ]
    if len(headers_info) < len(column_dicts):
        for i in range(len(column_dicts) - len(headers_info)):
            headers_info.append([])

    if mode == "append":
        hdulist = pyfits.open(template)
    elif mode == "new":
        header = pyfits.getheader(template)
        pri_hdu = pyfits.PrimaryHDU(header=header)
        hdulist = pyfits.HDUList([pri_hdu, ])
    else:
        raise ValueError('Unknown file creation mode ({})! Use either "append" or "new"'.format(mode))

    if len(primary_header.keys()) > 0:
        for key in primary_header:
            hdulist[0].header[key] = primary_header[key]

    for i in range(len(column_dicts)):
        column_dict = column_dicts[i]
        header_info = headers_info[i]
        columns = []
        for key in column_dict.keys():
            columns.append(pyfits.Column(name=key, format="D", array=column_dict[key]))
        cols = pyfits.ColDefs(columns)
        tablehdu = pyfits.BinTableHDU.from_columns(cols)

        # Add keywords to extension header
        num_keywords = len(header_info)
        header = tablehdu.header
        for i in range(num_keywords):
            info = header_info[i]
            if len(info) > 2:
                header.set(info[0], info[1], info[2])
            elif len(info) == 2:
                header.set(info[0], info[1])

        hdulist.append(tablehdu)

    hdulist.writeto(outfilename, clobber=True, output_verify='ignore')
    hdulist.close()


def LowPassFilter(data, vel, width=5, linearize=False):
    """
    Function to apply a low-pass filter to data.

    Parameters:
    ===========
    - data:        kglib.utils.DataStructures.xypoint instance
                   The data to filter.

    - vel:         float
                   The width of the features to smooth out, in velocity space (in cm/s).

    - width:       integer
                   How long it takes the filter to cut off, in units of wavenumber.

    - linearize:   boolean
                   Do we need to resample the data to a constant wavelength spacing
                   before filtering? Set to True if the data is unevenly sampled.

    Returns:
    ========
    If linearize = True:  Returns a numpy.ndarray with the new x-axis, and the filtered data
    Otherwise:  Returns just a numpy.ndarray with the filtered data.
    """

    if linearize:
        data = data.copy()
        datafcn = spline(data.x, data.y, k=1)
        errorfcn = spline(data.x, data.err, k=1)
        contfcn = spline(data.x, data.cont, k=1)
        linear = DataStructures.xypoint(data.x.size)
        linear.x = np.linspace(data.x[0], data.x[-1], linear.size())
        linear.y = datafcn(linear.x)
        linear.err = errorfcn(linear.x)
        linear.cont = contfcn(linear.x)
        data = linear

    # Figure out cutoff frequency from the velocity.
    featuresize = data.x.mean() * vel / constants.c.cgs.value  # vel MUST be given in units of cm
    dlam = data.x[1] - data.x[0]  # data.x MUST have constant x-spacing
    cutoff_hz = 1.0 / featuresize

    sample_rate = 1.0 / dlam
    nyq_rate = sample_rate / 2.0  # The Nyquist rate of the signal.
    width /= nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

    # Extend data to prevent edge effects
    y = np.r_[data.y[::-1], data.y, data.y[::-1]]

    # Use lfilter to filter data with the FIR filter.
    smoothed_y = lfilter(taps, 1.0, y)

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate
    delay_idx = np.searchsorted(data.x, data.x[0] + delay) - 1
    smoothed_y = smoothed_y[data.size() + delay_idx:-data.size() + delay_idx]
    if linearize:
        return linear.x, smoothed_y
    else:
        return smoothed_y


def IterativeLowPass(data, vel, numiter=100, lowreject=3, highreject=3, width=5, linearize=False):
    """
    An iterative version of LowPassFilter.
    It will ignore outliers in the low pass filter. New parameters that are
    not described in the docstring fro LowPassFilter are:

    Parameters:
    ===========
    - numiter:       integer
                     The maximum number of iterations to take

    - lowreject:     integer
                     How many sigma below the current filtered curve do we
                     count as bad and ignore in the next iteration?

    - highreject:    integer
                     How many sigma above the current filtered curve do we
                     count as bad and ignore in the next iteration?
    """

    datacopy = data.copy()
    if linearize:
        datafcn = spline(datacopy.x, datacopy.y, k=3)
        errorfcn = spline(datacopy.x, datacopy.err, k=1)
        contfcn = spline(datacopy.x, datacopy.cont, k=1)
        linear = DataStructures.xypoint(datacopy.x.size)
        linear.x = np.linspace(datacopy.x[0], datacopy.x[-1], linear.size())
        linear.y = datafcn(linear.x)
        linear.err = errorfcn(linear.x)
        linear.cont = contfcn(linear.x)
        datacopy = linear.copy()

    done = False
    iter = 0
    datacopy.cont = FittingUtilities.Continuum(datacopy.x, datacopy.y, fitorder=9, lowreject=2.5, highreject=5)
    while not done and iter < numiter:
        done = True
        iter += 1
        smoothed = LowPassFilter(datacopy, vel, width=width)
        residuals = datacopy.y / smoothed
        mean = np.mean(residuals)
        std = np.std(residuals)
        badpoints = np.where(np.logical_or((residuals - mean) < -lowreject * std, residuals - mean > highreject * std))[
            0]
        if badpoints.size > 0:
            done = False
            datacopy.y[badpoints] = smoothed[badpoints]
    if linearize:
        return linear.x, smoothed
    else:
        return smoothed


def HighPassFilter(data, vel, width=5, linearize=False):
    """
    Function to apply a high-pass filter to data.

    Parameters:
    ===========
    - data:        kglib.utils.DataStructures.xypoint instance
                   The data to filter.

    - vel:         float
                   The width of the features to smooth out, in velocity space (in cm/s).

    - width:       integer
                   How long it takes the filter to cut off, in units of wavenumber.

    - linearize:   boolean
                   Do we need to resample the data to a constant wavelength spacing
                   before filtering? Set to True if the data is unevenly sampled.

    Returns:
    ========
    If linearize = True:  Returns a numpy.ndarray with the new x-axis, and the filtered data
    Otherwise:  Returns just a numpy.ndarray with the filtered data.
    """

    if linearize:
        original_data = data.copy()
        datafcn = spline(data.x, data.y, k=3)
        errorfcn = spline(data.x, data.err, k=3)
        contfcn = spline(data.x, data.cont, k=3)
        linear = DataStructures.xypoint(data.x.size)
        linear.x = np.linspace(data.x[0], data.x[-1], linear.size())
        linear.y = datafcn(linear.x)
        linear.err = errorfcn(linear.x)
        linear.cont = contfcn(linear.x)
        data = linear

    # Figure out cutoff frequency from the velocity.
    featuresize = 2 * data.x.mean() * vel / constants.c.cgs.value  # vel MUST be given in units of cm
    dlam = data.x[1] - data.x[0]  # data.x MUST have constant x-spacing
    Npix = featuresize / dlam

    nsamples = data.size()
    sample_rate = 1.0 / dlam
    nyq_rate = sample_rate / 2.0  # The Nyquist rate of the signal.
    width /= nyq_rate
    cutoff_hz = min(1.0 / featuresize, nyq_rate - width * nyq_rate / 2.0)  # Cutoff frequency of the filter

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    if N % 2 == 0:
        N += 1

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero=False)

    # Extend data to prevent edge effects
    y = np.r_[data.y[::-1], data.y, data.y[::-1]]

    # Use lfilter to filter data with the FIR filter.
    smoothed_y = lfilter(taps, 1.0, y)

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate
    delay_idx = np.searchsorted(data.x, data.x[0] + delay)
    smoothed_y = smoothed_y[data.size() + delay_idx:-data.size() + delay_idx]
    if linearize:
        fcn = spline(data.x, smoothed_y)
        return fcn(original_data.x)
    else:
        return smoothed_y








if mlpy_import:
    def Denoise(data):
        """
        This function implements the denoising given in the url below:
        http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4607982&tag=1

        with title "Astronomical Spectra Denoising Based on Simplifed SURE-LET Wavelet Thresholding"

        The data should be a kglib.utils.DataStructures.xypoint instance.
        """
        y, boolarr = mlpy.wavelet.pad(data.y)
        WC = mlpy.wavelet.dwt(y, 'd', 10, 0)
        # Figure out the unknown parameter 'a'
        sum1 = 0.0
        sum2 = 0.0
        numlevels = int(np.log2(WC.size))
        start = 2 ** (numlevels - 1)
        median = np.median(WC[start:])
        sigma = np.median(np.abs(WC[start:] - median)) / 0.6745
        for w in WC:
            phi = w * np.exp(-w ** 2 / (12.0 * sigma ** 2))
            dphi = np.exp(-w ** 2 / (12.0 * sigma ** 2)) * (1 - 2 * w ** 2 / (12 * sigma ** 2) )
            sum1 += sigma ** 2 * dphi
            sum2 += phi ** 2
        a = -sum1 / sum2

        # Adjust all wavelet coefficients
        WC = WC + a * WC * np.exp(-WC ** 2 / (12 * sigma ** 2))

        # Now, do a soft threshold
        threshold = scoreatpercentile(WC, 80.0)
        WC[np.abs(WC) <= threshold] = 0.0
        WC[np.abs(WC) > threshold] -= threshold * np.sign(WC[np.abs(WC) > threshold])

        #Transform back
        y2 = mlpy.wavelet.idwt(WC, 'd', 10)
        data.y = y2[boolarr]
        return data


    # Kept for legacy support, since I was using Denoise3 in several codes in the past.
    def Denoise3(data):
        warnings.warn('Use Denoise function instead!', DeprecationWarning)
        return Denoise(data)


def Gauss(x, mu, sigma, amp=1):
    """ Return a gaussian at location x, with parameters mu, sigma, and amp """
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def FindOutliers(data, numsiglow=6, numsighigh=3, numiters=10, expand=0):
    """
    Find outliers in the data. Outliers are defined as
    points that are more than numsiglow standard deviations
    below the mean, or numsighigh standard deviations above
    the mean. Returns the index of the outliers in the data.

    Data should be an kglib.utils.DataStructures.xypoint instance
    The expand keyword will expand the rejected points some number
      from every rejected point.
    """

    done = False
    i = 0
    good = np.arange(data.size()).astype(int)

    while not done and i < numiters:
        sig = np.std(data.y[good] / data.cont[good])
        outliers = np.where(np.logical_or(data.y / data.cont - 1.0 > numsighigh * sig,
                                          data.y / data.cont - 1.0 < -numsiglow * sig))[0]
        good = np.where(np.logical_and(data.y / data.cont - 1.0 <= numsighigh * sig,
                                       data.y / data.cont - 1.0 >= -numsiglow * sig))[0]
        i += 1
        if outliers.size < 1:
            break

    # Now, expand the outliers by 'expand' pixels on either
    exclude = []
    for outlier in outliers:
        for i in range(max(0, outlier - expand), min(outlier + expand + 1, data.size())):
            exclude.append(i)

    # Remove duplicates from 'exclude'
    temp = []
    [temp.append(i) for i in exclude if not i in temp]
    return np.array(temp)


def IsListlike(arg):
    """
    This function just tests to check if the object acts like a list
    """
    from six import string_types

    if isinstance(arg, string_types):
        return False
    try:
        _ = [x for x in arg]
        return True
    except TypeError:  # catch when for loop fails
        return False



def mad(arr):
    """
    Median average deviation.
    The parameter `arr` is any list-like object (list, tuple, numpy.ndarray)
    """
    if not IsListlike(arr):
        raise ValueError("The input to mad must be a list-like object!")

    median = np.nanmedian(arr)
    arr = np.array(arr)
    return np.nanmedian(np.abs(arr - median))


def split_radec(radec, to_float=False):
    """
    Splits an RA/DEC string into separate RA and DEC strings.

    Parameters:
    ===========
    - radec:    string
                Has the form return by simbad queries: "00 10 02.20293 +11 08 44.9280"

    - to_float: boolean
                If True, it will convert the RA and DEC values to floats.
                Otherwise, it leaves them in hexadecimal
    """
    delim = '+' if '+' in radec else '-'
    segments = radec.split(delim)
    ra = segments[0].strip()
    dec = delim + segments[1].strip()

    if to_float:
        ra = 15 * convert_hex_string(ra, delimiter=' ')
        dec = convert_hex_string(dec, delimiter=' ')

    return ra, dec


def radec2altaz(ra, dec, obstime, lat=None, lon=None, debug=False):
    """
    calculates the altitude and azimuth, given an ra, dec, time, and observatory location

    Parameters:
    ===========
    - ra:        float
                 The right ascension of the target (in degrees)

    - dec:       float
                 The declination of the target (in degrees)

    - obstime:   astropy.time.Time object
                 Contains the time of the observation.
                 Can also contain the observatory location if
                 lat and lon are not given.

    - lat:       float
                 The latitude of the observatory, in degrees.
                 Not needed if given in the obstime object

    - lon:      float
                 The longitude of the observatory, in degrees.
                 Not needed if given in the obstime object

    Returns:
    ========
    The altitude and azimuth of the object, both in degrees.
    """

    if lat is None:
        lat = obstime.lat.degree
    if lon is None:
        lon = obstime.lon.degree
    obstime = Time(obstime.isot, format='isot', scale='utc', location=(lon, lat))

    # Find the number of days since J2000
    j2000 = Time("2000-01-01T12:00:00.0", format='isot', scale='utc')
    dt = (obstime - j2000).value  # number of days since J2000 epoch

    # get the UT time
    tstring = obstime.isot.split("T")[-1]
    segments = tstring.split(":")
    ut = float(segments[0]) + float(segments[1]) / 60.0 + float(segments[2]) / 3600

    # Calculate Local Sidereal Time
    lst = obstime.sidereal_time('mean').deg

    # Calculate the hour angle
    HA = lst - ra
    while HA < 0.0 or HA > 360.0:
        s = -np.sign(HA)
        HA += s * 360.0

    # convert everything to radians
    dec *= np.pi / 180.0
    lat *= np.pi / 180.0
    HA *= np.pi / 180.0

    # Calculate the altitude
    alt = np.arcsin(np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(HA))

    # calculate the azimuth
    az = np.arccos((np.sin(dec) - np.sin(alt) * np.sin(lat)) / (np.cos(alt) * np.cos(lat)))
    if np.sin(HA) > 0:
        az = 2.0 * np.pi - az

    if debug:
        print( "UT: ", ut)
        print( "LST: ", lst)
        print( "HA: ", HA * 180.0 / np.pi)

    return alt * 180.0 / np.pi, az * 180.0 / np.pi


def safe_convert(s, default=0):
    """
    Converts something to a float. If an error occurs, returns the default value.
    """
    try:
        v = float(s)
    except ValueError:
        v = default
    return v


def convert_hex_string(string, delimiter=":", debug=False):
    """
    Converts a hexadecimal coordinate string to a decimal

    Parameters:
    ===========
    - string:      string
                   The hex string to convert

    - delimiter:   string
                   The delimiter between hours, minutes, and seconds in
                   the hex string

    Returns:
    ========
    The decimal number equivalent to the input hex string.
    """
    if debug:
        print('Parsing hex string {}'.format(string))
    segments = string.split(delimiter)
    s = -1.0 if '-' in string else 1.0
    return s * (abs(safe_convert(segments[0])) + safe_convert(segments[1]) / 60.0 + safe_convert(segments[2]) / 3600.0)


def convert_to_hex(val, delimiter=':', force_sign=False):
    """
    Converts a numerical value into a hexidecimal string

    Parameters:
    ===========
    - val:           float
                     The decimal number to convert to hex.

    - delimiter:     string
                     The delimiter between hours, minutes, and seconds
                     in the output hex string.

    - force_sign:    boolean
                     Include the sign of the string on the output,
                     even if positive? Usually, you will set this to
                     False for RA values and True for DEC

    Returns:
    ========
    A hexadecimal representation of the input value.
    """
    s = np.sign(val)
    s_factor = 1 if s > 0 else -1
    val = np.abs(val)
    degree = int(val)
    minute = int((val  - degree)*60)
    second = (val - degree - minute/60.0)*3600.
    if degree == 0 and s_factor < 0:
        return '-00{2:s}{0:02d}{2:s}{1:.2f}'.format(minute, second, delimiter)
    elif force_sign or s_factor < 0:
        deg_str = '{:+03d}'.format(degree * s_factor)
    else:
        deg_str = '{:02d}'.format(degree * s_factor)
    return '{0:s}{3:s}{1:02d}{3:s}{2:.2f}'.format(deg_str, minute, second, delimiter)


def GetZenithDistance(header=None, date=None, ut=None, ra=None, dec=None, lat=None, lon=None, debug=False):
    """
    Function to get the zenith distance to an object.
    the zenith distance is the angular distance from the zenith
    to that object.

    Parameters:
    ===========
    - header:   astropy.io.fits header object, or a dictionary
                Must have at least the following keys:
                'date-obs', 'ra', and 'dec'

    - date:     string
                The UT date of the observation, in isot format.
                (only used if header not given)

    - ut:       string
                The UTC time of the observation
                (only used if header not given)

    - ra:       float
                The right ascension of the observation, in degrees
                (only used if header not given)

    - dec:      float
                The declination of the observation, in degrees
                (only used if header not given)

    - lat:      float
                The latitude of the observatory, in degrees

    - lon:      float
                The longitude of the observatory, in degrees

    Returns:
    ========
    The zenith distance of the object, in degrees
    """

    if header is None:
        obstime = Time("{}T{}".format(date, ut), format='isot', scale='utc', location=(lon, lat))
    else:
        obstime = Time(header['date-obs'], format='isot', scale='utc', location=(lon, lat))
        delimiter = ":" if ":" in header['ra'] else " "
        ra = 15.0 * convert_hex_string(header['ra'], delimiter=delimiter)
        dec = convert_hex_string(header['dec'], delimiter=delimiter)

    if debug:
        print( ra, dec)
    alt, az = radec2altaz(ra, dec, obstime, debug=debug)
    return 90.0 - alt




def FindOrderNums(orders, wavelengths):
    """
      Given a list of kglib.utils.DataStructures.xypoint orders and
      another list of wavelengths, this
      finds the order numbers with the
      requested wavelengths. Note that it finds the first
      order that contains each wavelength, so you should
      take care to give wavelengths that are unique to
      one order.
    """
    nums = []
    for wave in wavelengths:
        for i, order in enumerate(orders):
            if order.x[0] < wave and order.x[-1] > wave:
                nums.append(i)
                break
    return nums



def add_magnitudes(*mag_list):
    """
    Combine magnitudes in the right way.
    Any number of magnitudes can be passed, so long as they are
    all in the same filter/band!.
    """
    flux_list = np.array([10 ** (-m / 2.5) for m in mag_list])
    total_flux = np.sum(flux_list, axis=0)
    total_mag = -2.5 * np.log10(total_flux)
    return total_mag


def fwhm(x, y, k=10, ret_roots=False):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the dataset.  The function
    uses a spline interpolation with smoothing parameter k ('s' in scipy.interpolate.UnivariateSpline).

    If ret_roots=True, return the x-locations at half maximum instead of just
    the distance between them.
    """


    class NoPeaksFound(Exception):
        pass

    half_max = np.max(y) / 2.0
    s = UnivariateSpline(x, y - half_max, s=k)
    roots = s.roots()

    if len(roots) > 2:
        # Multiple peaks. Use the two that straddle the maximum value
        maxvel = x[np.argmax(y)]
        left_idx = np.argmin(maxvel - roots)
        right_idx = np.argmin(roots - maxvel)
        roots = np.array((roots[left_idx], roots[right_idx]))
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                           "the dataset is flat (e.g. all zeros).")
    if ret_roots:
        return roots[0], roots[1]

    return abs(roots[1] - roots[0])


def integral(x, y, I, k=10):
    """
    Integrate y = f(x) for x = 0 to a such that the integral = I
    I can be an array.

    Returns the values a that are found.
    """
    I = np.atleast_1d(I)

    f = UnivariateSpline(x, y, s=k)

    # Integrate as a function of x
    F = f.antiderivative()
    Y = F(x)

    a = []
    for intval in I:
        F2 = UnivariateSpline(x, Y/Y[-1] - intval, s=0)
        a.append(F2.roots())

    return np.hstack(a)


def is_close(num1, num2, inf_true=True, both_inf=False):
    """
    Wrapper around np.isclose that handles Nans/infinities how I want them

    Parameters:
    ===========
    - num1, num2:   floats
                    The numeric values to compare

    - inf_true:     boolean
                    Return True if one or both of the numbers are infinite or NaN?

    - both_inf:     boolean
                    Only return True if both numbers are infinite.
    :return: boolean
    """
    # Treat strings a bit differently.
    from six import string_types

    if isinstance(num1, string_types) or isinstance(num2, string_types):
        return num1 == num2

    if np.isfinite(num1) and np.isfinite(num2):
        return np.isclose(num1, num2)

    elif not inf_true or (inf_true and both_inf and (np.isfinite(num1) or np.isfinite(num2))):
        return False

    return True


class ExtrapolatingUnivariateSpline(spline):
    """
    Does the same thing as InterpolatedUnivariateSpline, but keeps track of if it is
    extrapolating. When extrapolating, this will just return the fill value which
    defaults to NaN.
    """
    def __init__(self, x, y, w=None, bbox=[None]*2, k=3, ext=0, fill_value=np.nan):
        """
        See docstring for InterpolatedUnivariateSpline.
        """
        self.bounds = (min(x), max(x))
        self.fill_value = fill_value
        super(ExtrapolatingUnivariateSpline, self).__init__(x, y, w=w, bbox=bbox, k=k, ext=ext)

    def __call__(self, x, nu=0, ext=None):
        """
        See docstring for InterpolatedUnivariateSpline.__call__
        """
        x = np.atleast_1d(x)
        retval = super(ExtrapolatingUnivariateSpline, self).__call__(x, nu=nu, ext=ext)
        idx = ~((x > self.bounds[0]) & (x < self.bounds[1]))
        retval[idx] = self.fill_value
        return retval


class ErrorPropagationSpline(object):
    """
    Does a spline fit, but returns both the spline value and associated uncertainty.
    This accomplishes the task by generating lots of splines, and return the mean
    and standard deviation of the spline values at the requested coordinates.
    It is therefore roughly N times slower than a normal spline!
    """

    def __init__(self, x, y, yerr, N=1000, *args, **kwargs):
        """
        See docstring for InterpolatedUnivariateSpline
        The parameter `N` gives the number of splines to generate for error propagation.
        """
        xx = np.vstack([x for i in range(N)]).T
        yy = np.vstack([y + np.random.normal(loc=0, scale=yerr) for i in range(N)]).T
        self._splines = [spline(x, yy[:, i], *args, **kwargs) for i in range(N)]

    def __call__(self, x, *args, **kwargs):
        """
        Get the spline value and uncertainty at point(s) x. args and kwargs are passed to spline.__call__
        """
        x = np.atleast_1d(x)
        s = np.vstack([curve(x, *args, **kwargs) for curve in self._splines])
        return (np.mean(s, axis=0), np.std(s, axis=0))


def CombineXYpoints(xypts, snr=None, xspacing=None, numpoints=None, interp_order=3):
    warnings.warn('This function has moved to kglib.utils.DataStructures!', DeprecationWarning)
    from kglib.utils.DataStructures import CombineXYpoints

    return CombineXYpoints(xypts=xypts, snr=snr, xspacing=xspacing,
                           numpoints=numpoints, interp_order=interp_order)


def weighted_mean_and_stddev(arr, weights=None, bad_value=np.nan):
    """
    Returns the weighted mean and standard deviation of an array.

    Parameters:
    ===========
    - arr:          Any list-like object (list, tuple, numpy array...)
                    The array of values.

    - weights:      A list-like object with the same shape as arr
                    The weights to apply to the values.

    - bad_value:    float
                    Return this value if the input array has zero length.

    Returns:
    ========
    The weighted mean and standard deviation of the array.
    """

    arr = np.atleast_1d(arr).astype(np.float)
    weights = np.atleast_1d(weights).astype(np.float)

    if len(arr) == 0:
        logging.warn('Zero-length array given! Mean and standard deviation are undefined')
        return bad_value, bad_value
    if weights is None:
        weights = np.ones_like(arr)

    avg = np.average(arr, weights=weights)
    if len(arr) > 1:
        var = np.average((arr - avg) ** 2, weights=weights)
        V1 = np.sum(weights)
        V2 = np.sum(weights ** 2)

        return avg, np.sqrt(var / (1 - V2 / V1 ** 2)) + np.nansum(1.0 / np.sqrt(weights))
    return avg, 1.0 / np.sqrt(weights[0])


# ############################################################
#  The following functions are really only useful for
#  my project.
#############################################################

def get_max_velocity(p_spt, s_temp):
    MS = SpectralTypeRelations.MainSequence()
    s_spt = MS.GetSpectralType('temperature', s_temp, prec=1e-3)
    R1 = MS.Interpolate('radius', p_spt)
    T1 = MS.Interpolate('temperature', p_spt)
    M1 = MS.Interpolate('mass', p_spt)
    M2 = MS.Interpolate('mass', s_spt)
    G = constants.G.cgs.value
    Msun = constants.M_sun.cgs.value
    Rsun = constants.R_sun.cgs.value
    v2 = 2.0 * G * Msun * (M1 + M2) / (Rsun * R1 * (T1 / s_temp) ** 2)
    return np.sqrt(v2) * u.cm.to(u.km)


@u.quantity_input(v=u.km / u.s, d=u.parsec)
def get_max_separation(p_spt, s_temp, v, d=1.0 * u.parsec):
    """
    Get the maximum separation for a binary candidate
    :param p_spt:   The spectral type of the primary star
    :param s_temp:  The temperature of the companion
    :param v:       The velocity, in km/s, of the companion
    :param d:       The distance, in parsec, to the system
    :return:        The maximum primary-->secondary separation, in arcseconds
    """
    # Convert the companion temperature and primary spectral type to mass
    from kglib.spectral_type import Mamajek_Table

    MS = SpectralTypeRelations.MainSequence()
    MT = Mamajek_Table.MamajekTable()
    teff2mass = MT.get_interpolator('Teff', 'Msun')
    M1 = MS.Interpolate('mass', p_spt)
    M2 = teff2mass(s_temp)
    Mt = (M1 + M2) * u.M_sun

    # Compute the maximum separation
    G = constants.G
    a_max = (G * Mt / v ** 2)
    alpha_max = (a_max / d).to(u.arcsecond, equivalencies=u.dimensionless_angles())
    return alpha_max


OBS_TARGET_FNAME = '{}/Dropbox/School/Research/AstarStuff/TargetLists/Observed_Targets3.xls'.format(os.environ['HOME'])


def read_observed_targets(target_filename=OBS_TARGET_FNAME):
    """
    Reads the observed targets excel file into a pandas dataframe
    :param target_filename: The filename to read. Has a very specific format!
    :return:
    """
    sample_names = ['identifier', 'RA/DEC (J2000)', 'plx', 'Vmag', 'Kmag', 'vsini', 'SpT', 'configuration',
                    'Instrument',
                    'Date',
                    'Temperature', 'Velocity', 'vsini_sec', '[Fe/H]', 'Significance', 'Sens_min', 'Sens_any',
                    'Comments',
                    'Rank', 'Keck', 'VLT', 'Gemini', 'Imaging_Detecton']

    def plx_convert(s):
        try:
            return float(s)
        except ValueError:
            return np.nan

    sample = pd.read_excel(target_filename, sheetname=0, na_values=['     ~'], names=sample_names,
                           converters=dict(plx=plx_convert))
    sample = sample.reset_index(drop=True)[1:]

    # Convert everything to floats
    for col in sample.columns:
        sample[col] = pd.to_numeric(sample[col], errors='ignore')

    return sample


