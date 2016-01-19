import pylab
import numpy as np
from astropy import units, constants
from scipy.interpolate import UnivariateSpline

from utils import DataStructures
from utils.FittingUtilities import Continuum as FindContinuum


pi = np.pi


def CombineIntervals(intervals, overlap=0):
    iteration = 0
    print "\n\n"
    for interval in intervals:
        lastindex = interval.x.size - overlap

        if iteration == 0:
            firstindex = 0
            master_x = interval.x[firstindex:lastindex]
            master_y = interval.y[firstindex:lastindex]
            master_cont = interval.cont[firstindex:lastindex]
        else:
            firstindex = np.searchsorted(interval.x, master_x[-1]) + 1
            master_x = np.append(master_x, interval.x[firstindex:lastindex])
            master_y = np.append(master_y, interval.y[firstindex:lastindex])
            master_cont = np.append(master_cont, interval.cont[firstindex:lastindex])
        iteration += 1

    output = DataStructures.xypoint(master_x.size)
    output.x = master_x.copy()
    output.y = master_y.copy()
    output.cont = master_cont.copy()

    #Scale continuum so the highest normalized flux = 1.0
    maxindex = np.argmax(output.y / output.cont)
    factor = output.y[maxindex] / output.cont[maxindex]
    output.cont *= factor

    return output


def ReadFile(filename):
    #Read in file
    x, y = np.loadtxt(filename, unpack=True)
    model = DataStructures.xypoint(x.size)
    model.x = x.copy() * units.angstrom.to(units.nm)
    model.y = y.copy()
    #Read in continuum
    x, y = np.loadtxt(filename[:-1] + "17", unpack=True)
    cont_fcn = UnivariateSpline(x * units.angstrom.to(units.nm), y, s=0)
    model.cont = cont_fcn(model.x)

    return model


def Broaden(model, vsini, intervalsize=50.0, beta=1.0, linear=False, findcont=False):
    """
      model:           input filename of the spectrum. The continuum data is assumed to be in filename[:-1]+".17"
                       model can also be a DataStructures.xypoint containing the already-read model (must include continuum!)
      vsini:           the velocity (times sin(i) ) of the star
      intervalsize:    The size (in nm) of the interval to use for broadening. Since it depends on wavelength, you don't want to do all at once
      alpha:           Linear limb darkening. beta = b/a where I(u) = a + bu
      linear:          flag for if the x-spacing is already linear. If true, we don't need to make UnivariateSplines and linearize
      findcont:        flag to decide if the continuum needs to be found
    """

    if type(model) == str:
        model = ReadFile(model)

    if not linear:
        model_fcn = UnivariateSpline(model.x, model.y, s=0)
        if not findcont:
            cont_fcn = UnivariateSpline(model.x, model.cont, s=0)

    #Will convolve with broadening profile in steps, to keep accuracy
    #interval size is set as a keyword argument
    profilesize = -1
    firstindex = 0
    lastindex = 0
    intervals = []

    while lastindex < model.x.size - 1:
        lastindex = min(np.searchsorted(model.x, model.x[firstindex] + intervalsize), model.size() - 1)
        interval = DataStructures.xypoint(lastindex - firstindex + 1)
        if linear:
            interval.x = model.x[firstindex:lastindex]
            interval.y = model.y[firstindex:lastindex]
            if not findcont:
                interval.cont = model.cont[firstindex:lastindex]
        else:
            interval.x = np.linspace(model.x[firstindex], model.x[lastindex], lastindex - firstindex + 1)
            interval.y = model_fcn(interval.x)
            if not findcont:
                interval.cont = cont_fcn(interval.x)

        if findcont:
            interval.cont = FindContinuum.Continuum(interval.x, interval.y)


        #Make broadening profile
        wave0 = interval.x[interval.x.size / 2]
        zeta = wave0 * vsini / constants.c.cgs.value
        xspacing = interval.x[1] - interval.x[0]
        wave = np.arange(wave0 - zeta, wave0 + zeta + xspacing, xspacing)
        x = (wave - wave0) / zeta
        x[x < -1] = -1.0
        x[x > 1] = 1.0
        profile = 1.0 / (zeta * (1 + 2 * beta / 3.)) * (
        2 / np.pi * np.sqrt(1 - x ** 2) + 0.5 * beta * ( 1 - x ** 2  ) )
        if profile.size < 10:
            warning.warn("Warning! Profile size too small: %i\nNot broadening!" % (profile.size))
            intervals.append(interval)
            firstindex = lastindex - 2 * profile.size
            continue
        #plt.plot(wave, profile)
        #plt.show()


        """
        wave0 = interval.x[interval.x.size/2]
        zeta = wave0*vsini/constants.c.cgs.value
        xspacing = interval.x[1] - interval.x[0]
        wave = np.arange(wave0 - zeta, wave0 + zeta, xspacing)
        x = np.linspace(-1.0, 1.0, wave.size)
        flux = pi/2.0*(1.0 - 1.0/(1. + 2*beta/3.)*(2/pi*np.sqrt(1.-x**2) + beta/2*(1.-x**2)))
        profile = flux.max() - flux
        #plt.plot(profile)
        """

        #Extend interval to reduce edge effects (basically turn convolve into circular convolution)
        before = interval.y[-profile.size / 2:]
        after = interval.y[:profile.size / 2]
        before = interval.y[-int(profile.size):]
        after = interval.y[:int(profile.size)]
        extended = np.append(np.append(before, interval.y), after)

        if profile.size % 2 == 0:
            left, right = int(profile.size * 1.5), int(profile.size * 1.5) - 1
        else:
            left, right = int(profile.size * 1.5), int(profile.size * 1.5)

        #interval.y = scipy.signal.fftconvolve(extended, profile/profile.sum(), mode="valid")
        #interval.y = scipy.signal.fftconvolve(extended, profile/profile.sum(), mode="full")[left:-right]
        interval.y = np.convolve(extended, profile / profile.sum(), mode="full")[left:-right]
        intervals.append(interval)

        if profile.size > profilesize:
            profilesize = profile.size
        firstindex = lastindex - 2 * profile.size

    #plt.show()
    if len(intervals) > 1:
        return CombineIntervals(intervals, overlap=profilesize)
    else:
        return intervals[0]


"""
Same as above, but performs the convolution in velocity space
Note that this uses epsilon, rather than Beta = epsilon/(1-epsilon) !!
"""


def Broaden2(model, vsini, intervalsize=50.0, epsilon=0.5, linear=False, findcont=False):
    """
      model:           input filename of the spectrum. The continuum data is assumed to be in filename[:-1]+".17"
                       model can also be a DataStructures.xypoint containing the already-read model (must include continuum!)
      vsini:           the velocity (times sin(i) ) of the star
      intervalsize:    The size (in nm) of the interval to use for broadening. Since it depends on wavelength, you don't want to do all at once
      epsilon:          Linear limb darkening. I(u) = 1-epsilon + epsilon*u
      linear:          flag for if the x-spacing is already linear. If true, we don't need to make UnivariateSplines and linearize
      findcont:        flag to decide if the continuum needs to be found
    """

    if type(model) == str:
        model = ReadFile(model)

    if not findcont:
        cont_fcn = UnivariateSpline(model.x, model.cont, s=0)

    if not linear:
        model_fcn = UnivariateSpline(model.x, model.y, s=0)
        x = np.linspace(model.x[0], model.x[-1], model.size())
        model = DataStructures.xypoint(x=x, y=model_fcn(x))
        if not findcont:
            model.cont = cont_fcn(model.x)
        else:
            model.cont = FittingUtilities.Continuum(model.x, model.y, lowreject=1.5, highreject=10)
    elif findcont:
        model.cont = FittingUtilities.Continuum(model.x, model.y, lowreject=1.5, highreject=10)


    #Convert to velocity space
    wave0 = model.x[model.size() / 2]
    model.x = constants.c.cgs.value * (model.x - wave0) / wave0

    #Make broadening profile
    left = np.searchsorted(model.x, -2 * vsini)
    right = np.searchsorted(model.x, 2 * vsini)
    profile = model[left:right]
    profile.y = np.zeros(profile.size())
    dv = profile.x / vsini
    indices = np.where(np.abs(dv) < 1.0)[0]
    profile.y[indices] = 1.0 / (vsini * (1 - epsilon / 3.0)) * (
    2 * (1 - epsilon) / np.pi * np.sqrt(1 - dv[indices] ** 2) + epsilon / 2.0 * (1 - dv[indices] ** 2) )

    #Extend interval to reduce edge effects (basically turn convolve into circular convolution)
    before = model.y[-int(profile.size()):]
    after = model.y[:int(profile.size())]
    extended = np.append(np.append(before, model.y), after)

    if profile.size() % 2 == 0:
        left, right = int(profile.size() * 1.5), int(profile.size() * 1.5) - 1
    else:
        left, right = int(profile.size() * 1.5), int(profile.size() * 1.5)

    model.y = np.convolve(extended, profile.y / profile.y.sum(), mode="full")[left:-right]

    #Return back to wavelength space
    model.x = wave0 * (1 + model.x / constants.c.cgs.value)
    return model


def Test_fcn(model):
    model_fcn = UnivariateSpline(model.x, model.y, s=0)
    cont_fcn = UnivariateSpline(model.x, model.cont, s=0)

    print model_fcn(np.median(model.x))


if __name__ == "__main__":
    #filename = sys.argv[1]
    #SpT = sys.argv[2]
    #vsini = float(sys.argv[3]) #MUST BE GIVEN IN KM S^1

    filename = "BG19000g400v2.vis.7"
    fulldata = ReadFile(filename)
    left = np.searchsorted(fulldata.x, 850)
    right = np.searchsorted(fulldata.x, 950)
    data = DataStructures.xypoint(right - left + 1)
    data.x = fulldata.x[left:right]
    data.y = fulldata.y[left:right]
    data.cont = fulldata.cont[left:right]
    spectrum = Broaden(data, 150 * units.km.to(units.cm))
    pylab.plot(spectrum.x, spectrum.y / spectrum.cont)
    pylab.show()
    #outfilename = "Broadened_" + SpT + "_v%.0f.dat" %(vsini)
    #print "Outputting to ", outfilename
    #np.savetxt(outfilename, np.transpose((spectrum.x, spectrum.y/spectrum.cont)), fmt='%.8f\t%.8g')
