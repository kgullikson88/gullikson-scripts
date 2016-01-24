import sys
import os
import warnings
import logging

from astropy import units, constants
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import minimize_scalar
import numpy as np

from kglib.utils import FittingUtilities, DataStructures, HelperFunctions
from kglib.stellar_models.Broaden import RotBroad
from kglib.utils.PlotBlackbodies import Planck
import Normalized_Xcorr


currentdir = os.getcwd() + "/"
homedir = os.environ["HOME"]
outfiledir = currentdir + "Cross_correlations/"
modeldir = homedir + "/School/Research/Models/Sorted/Stellar/Vband/"
minvel = -1000  # Minimum velocity to output, in km/s
maxvel = 1000

HelperFunctions.ensure_dir(outfiledir)

model_list = [modeldir + "lte30-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte31-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte32-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte33-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte34-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte35-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte36-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte37-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte38-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte39-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte40-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte42-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte43-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte44-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte45-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte46-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte47-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte48-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte49-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte50-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte51-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte52-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte53-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte54-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte55-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte56-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte57-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte58-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte59-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte61-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte63-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte64-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte65-4.00-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte67-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte68-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte69-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte70-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted",
              modeldir + "lte72-4.50-0.0.AGS.Cond.PHOENIX-ACES-2009.HighRes.7.sorted"]

star_list = []
temp_list = []
gravity_list = []
metallicity_list = []
for fname in model_list:
    temp = int(fname.split("lte")[-1][:2]) * 100
    gravity = float(fname.split("lte")[-1][3:7])
    metallicity = float(fname.split("lte")[-1][7:11])
    star_list.append(str(temp))
    temp_list.append(temp)
    gravity_list.append(gravity)
    metallicity_list.append(metallicity)

"""
  This function just processes the model to prepare for cross-correlation
"""


def Process(model, data, vsini, resolution, debug=False, oversample=1, get_weights=False, prim_teff=10000.0):
    # Read in the model if necessary
    if isinstance(model, str):
        logging.debug("Reading in the input model from {0:s}".format(model))
        x, y = np.loadtxt(model, usecols=(0, 1), unpack=True)
        x = x * units.angstrom.to(units.nm)
        y = 10 ** y
        left = np.searchsorted(x, data[0].x[0] - 10)
        right = np.searchsorted(x, data[-1].x[-1] + 10)
        model = DataStructures.xypoint(x=x[left:right], y=y[left:right])
    elif not isinstance(model, DataStructures.xypoint):
        raise TypeError(
            "Input model is of an unknown type! Must be a DataStructures.xypoint or a string with the filename.")


    # Linearize the x-axis of the model
    logging.debug('Linearizing model')
    xgrid = np.linspace(model.x[0], model.x[-1], model.size())
    model = FittingUtilities.RebinData(model, xgrid)


    # Broaden
    logging.debug("Rotationally broadening model to vsini = {0:g} km/s".format(vsini * units.cm.to(units.km)))
    if vsini > 1.0 * units.km.to(units.cm):
        model = RotBroad.Broaden(model, vsini, linear=True)


    # Reduce resolution
    logging.debug(u"Convolving to the detector resolution of {}".format(resolution))
    if resolution is not None and  5000 < resolution < 500000:
        model = FittingUtilities.ReduceResolution(model, resolution)


    # Rebin subsets of the model to the same spacing as the data
    model_orders = []
    weights = []
    flux_ratio = []
    if debug:
        model.output("Test_model.dat")

    for i, order in enumerate(data):
        if debug:
            sys.stdout.write("\rGenerating model subset for order %i in the input data" % (i + 1))
            sys.stdout.flush()
        # Find how much to extend the model so that we can get maxvel range.
        dlambda = order.x[order.size() / 2] * maxvel * 1.5 / 3e5
        left = np.searchsorted(model.x, order.x[0] - dlambda)
        right = np.searchsorted(model.x, order.x[-1] + dlambda)
        right = min(right, model.size() - 2)

        # Figure out the log-spacing of the data
        logspacing = np.log(order.x[1] / order.x[0])

        # Finally, space the model segment with the same log-spacing
        start = np.log(model.x[left])
        end = np.log(model.x[right])
        xgrid = np.exp(np.arange(start, end + logspacing, logspacing))

        segment = FittingUtilities.RebinData(model[left:right + 1].copy(), xgrid)
        segment.cont = FittingUtilities.Continuum(segment.x, segment.y, lowreject=1.5, highreject=5, fitorder=2)
        model_orders.append(segment)

        # Measure the information content in the model, if get_weights is true
        # if get_weights:
        # slopes = [(segment.y[i + 1] / segment.cont[i + 1] - segment.y[i - 1] / segment.cont[i - 1]) /
        #              (segment.x[i + 1] - segment.x[i - 1]) for i in range(1, segment.size() - 1)]
        #    prim_flux = Planck(segment.x*units.nm.to(units.cm), prim_teff)
        #    lines = FittingUtilities.FindLines(segment)
        #    sec_flux = np.median(segment.cont[lines] - segment.y[lines])
        #    flux_ratio.append(np.median(sec_flux) / np.median(prim_flux))
        #    weights.append(np.sum(np.array(slopes) ** 2))

    # if get_weights:
    # weights = np.array(weights) * np.array(flux_ratio)
    #    print "Weights: ", np.array(weights) / np.sum(weights)
    #    return model_orders, np.array(weights) / np.sum(weights)
    return model_orders


def GetCCF(data, model, vsini=10.0, resolution=60000, process_model=True, rebin_data=True, debug=False, outputdir="./",
           addmode="ML", oversample=1, orderweights=None, get_weights=False, prim_teff=10000.0):
    """
    This is the main function. CALL THIS ONE!
    data: a list of xypoint instances with the data
    model: Either a string with the model filename, an xypoint instance, or
           a list of xypoint instances
    vsini: rotational velocity in km/s
    resolution: detector resolution in lam/dlam
    process_model: if true, it will generate a list of model orders suitable
                   for cross-correlation. Otherwise, it assumes the input
                   IS such a list
    rebin_data: If true, it will rebin the data to a constant log-spacing.
                Otherwise, it assumes the data input already is correctly spaced.
    debug: Prints debugging info to the screen, and saves various files.
    addmode: The CCF addition mode. The default is Maximum Likelihood
               (from Zucker 2003, MNRAS, 342, 1291). The other valid option
               is "simple", which will just do a straight addition. Maximum
               Likelihood is better for finding weak signals, but simple is
               better for determining parameters from the CCF (such as vsini)
    oversample: If rebin_data = True, this is the factor by which to over-sample
                the data while rebinning it. Ignored if rebin_data = False
    orderweights: Weights to apply to each order. Only used if addmode="weighted"
    get_weights: If true, and process_model=True, then the Process functions will
                 return both the model orders and weights for each function.In
                 addition, the weights will be returned in the output dictionary.
                 The weights are only used if addmode="weighted"
    prim_teff:   The effective temperature of the primary star. Used to determine the
                 flux ratio, which in turn is used to make the weights. Ignored if
                 addmode is not "weighted" or get_weights is False.
    """
    # Some error checking
    if addmode.lower() not in ["ml", "simple", "dc", "weighted"]:
        sys.exit("Invalid add mode given to Correlate.GetCCF: %s" % addmode)
    if addmode.lower() == "weighted" and orderweights is None and not get_weights:
        raise ValueError("Must give orderweights if addmode == weighted")
    if addmode.lower() == "weighted" and not get_weights and len(orderweights) != len(data):
        raise ValueError("orderweights must be a list-like object with the same size as data!")

    # Re-sample all orders of the data to logspacing, if necessary
    if rebin_data:
        logging.debug("Resampling data to log-spacing")
        for i, order in enumerate(data):
            start = np.log(order.x[0])
            end = np.log(order.x[-1])
            xgrid = np.logspace(start, end, order.size() * oversample, base=np.e)
            neworder = FittingUtilities.RebinData(order, xgrid)
            data[i] = neworder

    # Process the model if necessary
    if process_model:
        model_orders = Process(model, data, vsini * units.km.to(units.cm), resolution, debug=debug,
                               oversample=oversample, get_weights=get_weights, prim_teff=prim_teff)
        # if get_weights:
        # model_orders, orderweights = model_orders
    elif isinstance(model, list) and isinstance(model[0], DataStructures.xypoint):
        model_orders = model
    else:
        raise TypeError("model must be a list of DataStructures.xypoints if process=False!")

    # Now, cross-correlate the new data against the model
    if debug:
        corr, ccf_orders = Correlate(data, model_orders, debug=debug, outputdir=outputdir, addmode=addmode,
                                     orderweights=orderweights, get_weights=get_weights, prim_teff=prim_teff)
    else:
        corr = Correlate(data, model_orders, debug=debug, outputdir=outputdir, addmode=addmode,
                         orderweights=orderweights, get_weights=get_weights, prim_teff=prim_teff)

    retdict = {"CCF": corr,
               "model": model_orders,
               "data": data,
               "weights": orderweights}
    if debug:
        retdict['CCF_orders'] = ccf_orders
    return retdict


"""
  This function does the actual correlation.
"""


def Correlate(data, model_orders, debug=False, outputdir="./", addmode="ML",
              orderweights=None, get_weights=False, prim_teff=10000.0):
    # Error checking
    if "weighted" in addmode.lower() and orderweights is None and not get_weights:
        raise ValueError("Must give orderweights if addmode == weighted")

    corrlist = []
    normalization = 0.0
    normalization = 0.0
    info_content = []
    flux_ratio = []
    snr = []
    for ordernum, order in enumerate(data):
        model = model_orders[ordernum]
        if get_weights:
            slopes = [(model.y[i + 1] / model.cont[i + 1] - model.y[i - 1] / model.cont[i - 1]) /
                      (model.x[i + 1] - model.x[i - 1]) for i in range(1, model.size() - 1)]
            prim_flux = Planck(model.x * units.nm.to(units.cm), prim_teff)
            lines = FittingUtilities.FindLines(model)
            sec_flux = np.median(model.y.max() - model.y[lines])
            flux_ratio.append(np.median(sec_flux) / np.median(prim_flux))
            info_content.append(np.sum(np.array(slopes) ** 2))
            snr.append(1.0 / np.std(order.y))

        reduceddata = order.y / order.cont
        reducedmodel = model.y / model.cont

        # Get the CCF for this order
        l = np.searchsorted(model.x, order.x[0])
        if l > 0:
            if order.x[0] >= model.x[l]:
                dl = (order.x[0] - model.x[l]) / (model.x[l + 1] - model.x[l])
                l += dl
            else:
                logging.debug('Less!')
                dl = (model.x[l] - order.x[0]) / (model.x[l] - model.x[l - 1])
                l -= dl
            logging.debug('dl = {}'.format(dl))
        ycorr = Normalized_Xcorr.norm_xcorr(reduceddata, reducedmodel, trim=False)
        N = ycorr.size
        distancePerLag = np.log(model.x[1] / model.x[0])
        v1 = -(order.size() + l - 0.5) * distancePerLag
        vf = v1 + N * distancePerLag
        offsets = np.linspace(v1, vf, N)
        velocity = -offsets * constants.c.cgs.value * units.cm.to(units.km)
        corr = DataStructures.xypoint(velocity.size)
        corr.x = velocity[::-1]
        corr.y = ycorr[::-1]

        # Only save part of the correlation
        left = np.searchsorted(corr.x, minvel)
        right = np.searchsorted(corr.x, maxvel)
        corr = corr[left:right]

        # Make sure that no elements of corr.y are > 1!
        if max(corr.y) > 1.0:
            corr.y /= max(corr.y)


        # Save correlation
        if np.any(np.isnan(corr.y)):
            warnings.warn("NaNs found in correlation from order %i\n" % (ordernum + 1))
            continue
        normalization += float(order.size())
        corrlist.append(corr.copy())

    if get_weights:
        if debug:
            print("Weight components: ")
            print("lam_0  info  flux ratio,  S/N")
            for i, f, o, s in zip(info_content, flux_ratio, data, snr):
                print(np.median(o.x), i, f, s)
        info_content = (np.array(info_content) - min(info_content)) / (max(info_content) - min(info_content))
        flux_ratio = (np.array(flux_ratio) - min(flux_ratio)) / (max(flux_ratio) - min(flux_ratio))
        snr = (np.array(snr) - min(snr)) / (max(snr) - min(snr))
        orderweights = (1.0 * info_content ** 2 + 1.0 * flux_ratio ** 2 + 1.0 * snr ** 2)
        orderweights /= orderweights.sum()
        logging.debug('Weights:')
        logging.debug(orderweights)

    # Add up the individual CCFs
    total = corrlist[0].copy()
    total_ccfs = CCFContainer(total.x)

    if addmode.lower() == "ml" or addmode.lower() == 'all':
        # use the Maximum Likelihood method from Zucker 2003, MNRAS, 342, 1291
        total.y = np.ones(total.size())
        for i, corr in enumerate(corrlist):
            correlation = spline(corr.x, corr.y, k=1)
            N = data[i].size()
            total.y *= np.power(1.0 - correlation(total.x) ** 2, float(N) / normalization)
        total_ccfs['ml'] = np.sqrt(1.0 - total.y)
    if addmode.lower() == "simple" or addmode.lower() == 'all':
        # do a simple addition
        total.y = np.zeros(total.size())
        for i, corr in enumerate(corrlist):
            correlation = spline(corr.x, corr.y, k=1)
            total.y += correlation(total.x)
        total_ccfs['simple'] = total.y / float(len(corrlist))
    if addmode.lower() == "dc" or addmode.lower() == 'all':
        total.y = np.zeros(total.size())
        for i, corr in enumerate(corrlist):
            N = data[i].size()
            correlation = spline(corr.x, corr.y, k=1)
            total.y += float(N) * correlation(total.x) ** 2 / normalization
        total_ccfs['dc'] = np.sqrt(total.y)
    if addmode.lower() == "weighted" or (addmode.lower() == 'all' and orderweights is not None):
        total.y = np.zeros(total.size())
        for i, corr in enumerate(corrlist):
            w = orderweights[i] / np.sum(orderweights)
            correlation = spline(corr.x, corr.y, k=1)
            total.y += w * correlation(total.x) ** 2 
        total_ccfs['weighted'] = np.sqrt(total.y)
    if addmode.lower() == 'simple-weighted' or (addmode.lower() == 'all' and orderweights is not None):
        total.y = np.zeros(total.size())
        for i, corr in enumerate(corrlist):
            w = orderweights[i] / np.sum(orderweights)
            correlation = spline(corr.x, corr.y, k=1)
            total.y += correlation(total.x) * w
        total_ccfs['simple-weighted'] = total.y / float(len(corrlist))

    if addmode.lower() == 'all':
        return (total_ccfs, corrlist) if debug else total_ccfs
    return (total_ccfs[addmode], corrlist) if debug else total_ccfs[addmode]


class CCFContainer(object):
    """
    A class to store my CCFS. It acts much like a dictionary,
    but I can access the 'x' attribute to do barycentric correction.
    (also probably slightly more memory-efficient, but who cares).

    """
    def __init__(self, x):
        self.x = x
        self.ml = None
        self.dc = None
        self.simple = None
        self.weighted = None
        self.simple_weighted = None
        self.valid_keys = ('ml', 'dc', 'simple', 'weighted', 'simple-weighted')


    def __getitem__(self, item):
        if item not in self.valid_keys:
            raise KeyError('{} not a valid item for CCFContainer!'.format(item))

        if item == 'ml' and self.ml is not None:
            return DataStructures.xypoint(x=self.x, y=self.ml)
        elif item == 'dc' and self.dc is not None:
            return DataStructures.xypoint(x=self.x, y=self.dc)
        elif item == 'simple' and self.simple is not None:
            return DataStructures.xypoint(x=self.x, y=self.simple)
        elif item == 'weighted' and self.weighted is not None:
            return DataStructures.xypoint(x=self.x, y=self.weighted)
        elif item == 'simple-weighted' and self.simple_weighted is not None:
            return DataStructures.xypoint(x=self.x, y=self.simple_weighted)

        return None  # We should never get here...

    def __setitem__(self, key, value):
        if key not in self.valid_keys:
            raise KeyError('{} not a valid item for CCFContainer!'.format(key))

        assert value.shape == self.x.shape

        if key == 'ml':
            self.ml = value
        elif key == 'dc':
            self.dc = value
        elif key == 'simple':
            self.simple = value
        elif key == 'weighted':
            self.weighted = value
        elif key == 'simple-weighted':
            self.simple_weighted = value

        return None

    def keys(self):
        """ Get all the non-None keys
        """
        return [k for k in self.valid_keys if self[k] is not None]



def GetInformationContent(model):
    """
    Returns an array with the information content (right now, the derivative of the model)
    :param model: DataStructures.xypoint instance with the model
    :return: numpy.ndarray with the information content (used as weights)
    """
    info = np.ones(model.size())
    info[1:-1] = np.array(
        [(model.y[i + 1] - model.y[i - 1]) / (model.x[i + 1] - model.x[i - 1]) for i in range(1, model.size() - 1)])
    return info ** 2


def get_rv(vel, corr, Npix=None, **fit_kws):
    """
    Get the best radial velocity, with errors.
    This will only work if the ccf was made with the maximum likelihood method!
    Uses the formula given in Zucker (2003) MNRAS, 342, 4  for the rv error.

    :param vel:   The velocities
    :param corr:  The ccf values. Should be the same size as vel
    :param Npix:  The number of pixels used in the CCF.
    :return: rv, rv_err, ccf(rv)
    """
    vel = np.atleast_1d(vel)
    corr = np.atleast_1d(corr)
    sorter = np.argsort(vel)
    fcn = spline(vel[sorter], corr[sorter])
    fcn_prime = fcn.derivative(1)
    fcn_2prime = fcn.derivative(2)

    guess = vel[np.argmax(corr)]

    def errfcn(v):
        ll = 1e6*fcn_prime(v)**2
        return ll

    out = minimize_scalar(errfcn, bounds=(guess-2, guess+2), method='bounded')
    rv = out.x
    if Npix is None:
        Npix = vel.size
    
    rv_var = -(Npix * fcn_2prime(rv) * (fcn(rv) / (1 - fcn(rv) ** 2))) ** (-1)
    return rv, np.sqrt(rv_var), fcn(rv)

