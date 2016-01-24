
"""
This is a general script for doing the cross-correlations in my companion search.
It is called by several smaller scripts in each of the instrument-specific repositories
"""

import numpy as np

from utils import FittingUtilities, DataStructures
import Correlate
from utils import HelperFunctions
from stellar_models import StellarModel, Broaden


try:
    import pyraf
    pyraf_import = True
except ImportError:
    pyraf_import = False
from astropy.io import fits
from astropy.time import Time
import subprocess
from collections import defaultdict
from utils import StarData
from spectral_type import SpectralTypeRelations
from astropy import units as u
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import re
import sys
import os
import logging
import matplotlib.pyplot as plt
import h5py
import multiprocessing
import functools

if pyraf_import:
    pyraf.iraf.noao()
    pyraf.iraf.noao.rv()


def convert(coord, delim=":"):
    segments = coord.split(delim)
    s = -1.0 if "-" in segments[0] else 1.0
    return s * (abs(float(segments[0])) + float(segments[1]) / 60.0 + float(segments[2]) / 3600.0)


if pyraf_import:
    def HelCorr_IRAF(header, observatory="CTIO", debug=False):
        """
        Get the heliocentric correction for an observation
        """
        jd = header['jd']
        t = Time(jd, format='jd', scale='utc')
        dt = t.datetime
        output = pyraf.iraf.noao.rv.rvcorrect(epoch='INDEF',
                                              epoch_vsun='INDEF',
                                              observatory=observatory,
                                              year=dt.year,
                                              month=dt.month,
                                              day=dt.day,
                                              ut=header['ut'],
                                              ra=header['ra'],
                                              dec=header['dec'],
                                              files="",
                                              images="",
                                              input='no',
                                              Stdout=1)
        vbary = float(output[-1].split()[2])
        if debug:
            for line in output:
                print line
        return vbary
else:
    def HelCorr_IRAF(header, observatory="CTIO", debug=False):
        print "pyraf is not installed! Trying to use the idl version!"
        return 1e-3 * HelCorr(header, observatory=observatory, debug=debug)


def HelCorr(header, observatory="CTIO", idlpath="/Applications/exelis/idl83/bin/idl", debug=False):
    ra = 15.0 * convert(header['RA'])
    dec = convert(header['DEC'])
    jd = float(header['jd'])

    cmd_list = [idlpath,
                '-e',
                ("print, barycorr({:.8f}, {:.8f}, {:.8f}, 0,"
                 " obsname='CTIO')".format(jd, ra, dec)),
    ]
    if debug:
        print "RA: ", ra
        print "DEC: ", dec
        print "JD: ", jd
    output = subprocess.check_output(cmd_list).split("\n")
    if debug:
        for line in output:
            print line
    return float(output[-2])


SMOOTH_FACTOR = 0.25


def Process_Data_parallel(orders, badregions=[], interp_regions=[], extensions=True,
                 trimsize=1, vsini=None, logspacing=False, oversample=1.0, reject_outliers=True, cores=4):

    """
    Use multiprocessing module to parallelize the data processing
    """
    # Set up the multiprocessing stuff
    num_orders = len(orders) / cores + 1
    mp_args = [orders[num_orders*i:num_orders*(i+1)] for i in range(cores)]
    p = multiprocessing.Pool(cores)

    # Call Process_Data
    fcn = functools.partial(Process_Data_serial, badregions=badregions, interp_regions=interp_regions,
                            extensions=extensions, trimsize=trimsize, vsini=vsini, logspacing=logspacing,
                            oversample=oversample, reject_outliers=reject_outliers)
    tmp = p.map(fcn, mp_args)

    # Sort the output
    mp_out = []
    for t in tmp:
        mp_out.extend(t)

    return sorted(mp_out, key=lambda o: o.x[0])


def Process_Data(*args, **kwargs):
    return Process_Data_serial(*args, **kwargs)


def Process_Data_serial(fname, badregions=[], interp_regions=[], extensions=True,
                 trimsize=1, vsini=None, logspacing=False, oversample=1.0, reject_outliers=True):
    """

    :param fname: The filename to read in (should be a fits file)
    :param badregions: a list of regions to exclude (contains strong telluric or stellar line residuals)
    :param interp_regions: a list of regions to interpolate over
    :param extensions: A boolean flag for whether the fits file is separated into extensions
    :param trimsize: The amount to exclude from both ends of every order (where it is very noisy)
    :param vsini: the primary star vsini. If given subtract an estimate of the primary star model obtained by
                  denoising and smoothing with a kernel size set by the vsini.
    :logspacing: If true, interpolate each order into a constant log-spacing.
    :return:
    """
    if isinstance(fname, list) and all([isinstance(f, DataStructures.xypoint) for f in fname]):
        orders = fname
    else:
        if extensions:
            orders = HelperFunctions.ReadExtensionFits(fname)

        else:
            orders = HelperFunctions.ReadFits(fname, errors=2)

    numorders = len(orders)
    for i, order in enumerate(orders[::-1]):
        # Trim data, and make sure the wavelength spacing is constant
        if trimsize > 0:
            order = order[trimsize:-trimsize]

        # Smooth the data
        if vsini is not None:
            # make sure the x-spacing is linear
            xgrid = np.linspace(order.x[0], order.x[-1], order.size())
            order = FittingUtilities.RebinData(order, xgrid)

            dx = order.x[1] - order.x[0]
            """ old method
            theta = GenericSmooth.roundodd(vsini / 3e5 * order.x.mean() / dx * smooth_factor)
            theta = max(theta, 21)
            #denoised = HelperFunctions.Denoise(order.copy())
            denoised = order.copy()
            smooth = FittingUtilities.savitzky_golay(denoised.y, theta, 5)
            order.y = order.y - smooth + order.cont.mean()
            """
            # order.y = order.cont.mean() + HelperFunctions.HighPassFilter(order,
            #                                                             vel=smooth_factor * vsini * u.km.to(u.cm))
            #smoothed = HelperFunctions.astropy_smooth(order, vel=SMOOTH_FACTOR * vsini, linearize=False)
            smoothed = HelperFunctions.astropy_smooth(order, vel=SMOOTH_FACTOR * vsini, linearize=True)
            order.y += order.cont.mean() - smoothed
            order.cont = np.ones(order.size()) * order.cont.mean()

        # Remove bad regions from the data
        for region in badregions:
            left = np.searchsorted(order.x, region[0])
            right = np.searchsorted(order.x, region[1])
            if left > 0 and right < order.size():
                print "Warning! Bad region covers the middle of order %i" % i
                print "Removing full order!"
                left = 0
                right = order.size()
            order.x = np.delete(order.x, np.arange(left, right))
            order.y = np.delete(order.y, np.arange(left, right))
            order.cont = np.delete(order.cont, np.arange(left, right))
            order.err = np.delete(order.err, np.arange(left, right))

        # Interpolate over interp_regions:
        for region in interp_regions:
            left = np.searchsorted(order.x, region[0])
            right = np.searchsorted(order.x, region[1])
            order.y[left:right] = order.cont[left:right]


        # Remove whole order if it is too small
        remove = False
        if order.x.size <= 1:
            remove = True
        else:
            velrange = 3e5 * (np.median(order.x) - order.x[0]) / np.median(order.x)
            if velrange <= 1050.0:
                remove = True
        if remove:
            print "Removing order %i" % (numorders - 1 - i)
            orders.pop(numorders - 1 - i)
        else:
            if reject_outliers:
                # Find outliers from e.g. bad telluric line or stellar spectrum removal.
                order.cont = FittingUtilities.Continuum(order.x, order.y, lowreject=3, highreject=3)
                outliers = HelperFunctions.FindOutliers(order, expand=10, numsiglow=5, numsighigh=5)
                # plt.plot(order.x, order.y / order.cont, 'k-')
                if len(outliers) > 0:
                    # plt.plot(order.x[outliers], (order.y / order.cont)[outliers], 'r-')
                    order.y[outliers] = order.cont[outliers]
                    order.cont = FittingUtilities.Continuum(order.x, order.y, lowreject=3, highreject=3)
                    order.y[outliers] = order.cont[outliers]

            # Save this order
            orders[numorders - 1 - i] = order.copy()

    # plt.show()

    # Rebin the data to a constant log-spacing (if requested)
    if logspacing:
        for i, order in enumerate(orders):
            start = np.log(order.x[0])
            end = np.log(order.x[-1])
            neworder = order.copy()
            neworder.x = np.logspace(start, end, order.size() * oversample, base=np.e)
            neworder = FittingUtilities.RebinData(order, neworder.x)
            orders[i] = neworder

    return orders


def process_model(model, data, vsini_model=None, resolution=None, vsini_primary=None,
                  maxvel=1000.0, debug=False, oversample=1, logspace=True):
    # Read in the model if necessary
    if isinstance(model, str):
        if debug:
            print "Reading in the input model from %s" % model
        x, y = np.loadtxt(model, usecols=(0, 1), unpack=True)
        x = x * u.angstrom.to(u.nm)
        y = 10 ** y
        left = np.searchsorted(x, data[0].x[0] - 10)
        right = np.searchsorted(x, data[-1].x[-1] + 10)
        model = DataStructures.xypoint(x=x[left:right], y=y[left:right])
    elif not isinstance(model, DataStructures.xypoint):
        raise TypeError(
            "Input model is of an unknown type! Must be a DataStructures.xypoint or a string with the filename.")


    # Linearize the x-axis of the model (in log-spacing)
    if logspace:
        if debug:
            print "Linearizing model"
        xgrid = np.logspace(np.log10(model.x[0]), np.log10(model.x[-1]), model.size())
        model = FittingUtilities.RebinData(model, xgrid)
    else:
        xgrid = model.x

    # Broaden
    if vsini_model is not None and vsini_model > 1.0 * u.km.to(u.cm):
        if debug:
            print "Rotationally broadening model to vsini = %g km/s" % (vsini_model * u.cm.to(u.km))
        model = Broaden.RotBroad(model, vsini_model, linear=True)


    # Reduce resolution
    if resolution is not None and 5000 < resolution < 500000:
        if debug:
            print "Convolving to the detector resolution of %g" % resolution
        model = FittingUtilities.ReduceResolutionFFT(model, resolution)

    # Divide by the same smoothing kernel as we used for the data
    if vsini_primary is not None:
        """ old method
        smooth_factor = 0.5
        d_logx = np.log(xgrid[1] / xgrid[0])
        theta = GenericSmooth.roundodd(vsini_primary / 3e5 * smooth_factor / d_logx)
        print "Window size = {}\ndlogx = {}\nvsini = {}".format(theta, d_logx, vsini_primary)
        smooth = FittingUtilities.savitzky_golay(model.y, theta, 5)

        model.y = model.y - smooth
        """
        # model.y = HelperFunctions.HighPassFilter(model, vel=smooth_factor * vsini_primary * u.km.to(u.cm),
        #                                         linearize=True)
        #model.y -= HelperFunctions.astropy_smooth(model, vel=SMOOTH_FACTOR * vsini_primary, linearize=True)
        smoothed = HelperFunctions.astropy_smooth(model, vel=SMOOTH_FACTOR * vsini_primary, linearize=False)
        #minval = min(model.y)
        #model.y += abs(minval)
        #model.cont = abs(minval) * np.ones(model.size())
        model.y += model.cont.mean() - smoothed
        model.cont = np.ones(model.size()) * model.cont.mean()




    # Rebin subsets of the model to the same spacing as the data
    model_orders = []
    model_fcn = spline(model.x, model.y)
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

        segment = DataStructures.xypoint(x=xgrid, y=model_fcn(xgrid))
        # segment = FittingUtilities.RebinData(model[left:right + 1].copy(), xgrid)
        segment.cont = FittingUtilities.Continuum(segment.x, segment.y, lowreject=1.5, highreject=5, fitorder=2)
        model_orders.append(segment)

    print "\n"
    return model_orders


def CompanionSearch(fileList,
                    badregions=[],
                    interp_regions=[],
                    extensions=True,
                    resolution=60000,
                    trimsize=1,
                    vsini_values=(10, 20, 30, 40),
                    Tvalues=range(3000, 6900, 100),
                    metal_values=(-0.5, 0.0, +0.5),
                    logg_values=(4.5,),
                    modeldir=StellarModel.modeldir,
                    hdf5_file=StellarModel.HDF5_FILE,
                    vbary_correct=True,
                    observatory="CTIO",
                    addmode="ML",
                    debug=False):
    model_list = StellarModel.GetModelList(type='hdf5',
                                           hdf5_file=hdf5_file,
                                           temperature=Tvalues,
                                           metal=metal_values,
                                           logg=logg_values)
    modeldict, processed = StellarModel.MakeModelDicts(model_list, type='hdf5', hdf5_file=hdf5_file,
                                                       vsini_values=vsini_values, vac2air=True)

    get_weights = True if addmode.lower() == "weighted" else False
    orderweights = None

    MS = SpectralTypeRelations.MainSequence()

    # Do the cross-correlation
    datadict = defaultdict(list)
    temperature_dict = defaultdict(float)
    vbary_dict = defaultdict(float)
    alpha=0.0
    for temp in sorted(modeldict.keys()):
        for gravity in sorted(modeldict[temp].keys()):
            for metallicity in sorted(modeldict[temp][gravity].keys()):
                for vsini in vsini_values:
                    for fname in fileList:
                        if vbary_correct:
                            if fname in vbary_dict:
                                vbary = vbary_dict[fname]
                            else:
                                vbary = HelCorr_IRAF(fits.getheader(fname), observatory=observatory)
                                vbary_dict[fname] = vbary
                        process_data = False if fname in datadict else True
                        if process_data:
                            orders = Process_Data(fname, badregions, interp_regions=interp_regions,
                                                  extensions=extensions, trimsize=trimsize)
                            header = fits.getheader(fname)
                            spt = StarData.GetData(header['object']).spectype
                            match = re.search('[0-9]', spt)
                            if match is None:
                                spt = spt[0] + "5"
                            else:
                                spt = spt[:match.start() + 1]
                            temperature_dict[fname] = MS.Interpolate(MS.Temperature, spt)
                        else:
                            orders = datadict[fname]

                        output_dir = "Cross_correlations/"
                        outfilebase = fname.split(".fits")[0]
                        if "/" in fname:
                            dirs = fname.split("/")
                            output_dir = ""
                            outfilebase = dirs[-1].split(".fits")[0]
                            for directory in dirs[:-1]:
                                output_dir = output_dir + directory + "/"
                            output_dir = output_dir + "Cross_correlations/"
                        HelperFunctions.ensure_dir(output_dir)

                        model = modeldict[temp][gravity][metallicity][alpha][vsini]
                        pflag = not processed[temp][gravity][metallicity][alpha][vsini]
                        # if pflag:
                        # orderweights = None
                        retdict = Correlate.GetCCF(orders,
                                                   model,
                                                   resolution=resolution,
                                                   vsini=vsini,
                                                   rebin_data=process_data,
                                                   process_model=pflag,
                                                   debug=debug,
                                                   outputdir=output_dir.split("Cross_corr")[0],
                                                   addmode=addmode,
                                                   orderweights=orderweights,
                                                   get_weights=get_weights,
                                                   prim_teff=temperature_dict[fname])
                        corr = retdict["CCF"]
                        if pflag:
                            processed[temp][gravity][metallicity][alpha][vsini] = True
                            modeldict[temp][gravity][metallicity][alpha][vsini] = retdict["model"]
                            # orderweights = retdict['weights']
                        if process_data:
                            datadict[fname] = retdict['data']

                        outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}".format(output_dir, outfilebase,
                                                                                              vsini, temp, gravity,
                                                                                              metallicity)
                        print "Outputting to ", outfilename, "\n"
                        if vbary_correct:
                            corr.x += vbary
                        np.savetxt(outfilename, np.transpose((corr.x, corr.y)), fmt="%.10g")

                        if debug:
                            # Save the individual spectral inputs and CCF orders (unweighted)
                            output_dir2 = output_dir.replace("Cross_correlations", "CCF_inputs")
                            HelperFunctions.ensure_dir(output_dir2)
                            HelperFunctions.ensure_dir("%sCross_correlations/" % (output_dir2))

                            for i, (o, m, c) in enumerate(
                                    zip(retdict['data'], retdict['model'], retdict['CCF_orders'])):
                                print "Saving CCF inputs for order {}".format(i + 1)
                                outfilename = "{0:s}Cross_correlations/{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini,
                                    temp, gravity,
                                    metallicity, i + 1)
                                c.output(outfilename)
                                outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.data.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini,
                                    temp, gravity,
                                    metallicity, i + 1)
                                o.output(outfilename)
                                outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.model.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini,
                                    temp, gravity,
                                    metallicity, i + 1)
                                m.output(outfilename)



                    # Delete the model. We don't need it anymore and it just takes up ram.
                    modeldict[temp][gravity][metallicity][vsini] = []

    return


"""
===============================================================
        SLOW COMPANION SEARCH (BUT MORE ACCURATE)
===============================================================
"""


def slow_companion_search(fileList,
                          primary_vsini,
                          badregions=[],
                          interp_regions=[],
                          extensions=True,
                          resolution=None,
                          trimsize=1,
                          reject_outliers=True,
                          vsini_values=(10, 20, 30, 40),
                          Tvalues=range(3000, 6900, 100),
                          metal_values=(-0.5, 0.0, +0.5),
                          logg_values=(4.5,),
                          modeldir=StellarModel.modeldir,
                          hdf5_file=StellarModel.HDF5_FILE,
                          vbary_correct=True,
                          observatory="CTIO",
                          addmode="ML",
                          output_mode='text',
                          output_file='CCF.hdf5',
                          obstype='real',
                          min_x=None,
                          max_x=None,
                          debug=False,
                          makeplots=False):
    """
    This function runs a companion search, but makes a new model for each file. That is necessary because it
    subtracts the 'smoothed' model from the model spectrum before correlating
    :param fileList: The list of fits data files
    :param primary_vsini: A list of the same length as fileList, which contains the vsini for each star (in km/s)
    :param badregions: A list of wavelength regions to ignore in the CCF (contaminated by tellurics, etc)
    :param interp_regions: A list of wavelength regions to interpolate over in the data. Generally, badregions should be on the edge of the orders or contain the whole order, and interp_regions should contain a small wavelength range.
    :param resolution: The detector resolution in lam/dlam. The default is now to use a pre-broadened grid; do not give a value for resolution unless the grid is un-broadened!
    :param trimsize: The number of pixels to cut from both sides of each order. This is because the  order edges are usually pretty noisy.
    :param reject_outliers: Whether or not to detect and smooth over outliers in the data.
    :param vsini_values: A list of vsini values (in km/s) to apply to each model spectrum before correlation.
    :param Tvalues: A list of model temperatures (in K) to correlate the data against.
    :param metal_values: A list of [Fe/H] values to correlate the model against
    :param logg_values: A list of log(g) values (in cgs units) to correlate the model against
    :param modeldir: The model directory. This is no longer used by default!
    :param hdf5_file: The path to the hdf5 file containing the pre-broadened model grid.
    :param vbary_correct: Correct for the heliocentric motion of the Earth around the Sun?
    :param observatory: The name of the observatory, in a way that IRAF's rvcorrect will understand. Only needed if vbary_correct = True
    :param addmode: The way to add the CCFs for each order. Options are:
         1: 'simple': Do a simple average
         2: 'weighted': Do a weighted average: $C = \sum_i{w_i C_i^2}$ where $w_i$ is the line depth of the each pixel
         3: 'simple-weighted': Same as weighted, but without squaring the CCFs: $C = \sum_i{w_i C_i}$
         4: 'T-weighted': Do a weighted average: $C = \sum_i{w_i C_i}$ where $w_i$ is how fast each pixel changes with temperature
         5: 'dc': $C = \sum_i{C_i^2}$  (basically, weighting by the CCF itself)
         6: 'ml': The maximum likelihood estimate. See Zucker 2003, MNRAS, 342, 1291
         7: 'all': does simple, dc, and ml all at once.
    :param output_mode: How to output. Valid options are:
         1: text, which is just ascii data with a filename convention. This is the default option
         2: hdf5, which ouputs a single hdf5 file with all the metadata necessary to classify the output
    :param output_file: An HDF5 file to output to. Only used if output_mode = 'hdf5'
    :param obstype: Is this a synthetic binary star or real observation? (default is real)
    :param min_x:   The minimum wavelength to use in the model. If not given, the whole model will be used
    :param max_x:   The maximum wavelength to use in the model. If not given, the whole model will be used
    :param debug: Flag to print a bunch of information to screen, and save some intermediate data files
    :param makeplots: A 'higher level' of debug. Will make a plot of the data and model orders for each model.
    """

    # Make sure the temperature, metal, and logg are all at least 1d arrays.
    Tvalues = np.atleast_1d(Tvalues)
    metal_values = np.atleast_1d(metal_values)
    logg_values = np.atleast_1d(logg_values)    

    model_list = StellarModel.GetModelList(type='hdf5',
                                           hdf5_file=hdf5_file,
                                           temperature=Tvalues,
                                           metal=metal_values,
                                           logg=logg_values)
    if addmode.lower() == 't-weighted':
        modeldict, processed, sensitivity = StellarModel.MakeModelDicts(model_list, type='hdf5', hdf5_file=hdf5_file,
                                                       vsini_values=vsini_values, vac2air=True, logspace=True,
                                                       get_T_sens=True)
    else:
        modeldict, processed = StellarModel.MakeModelDicts(model_list, type='hdf5', hdf5_file=hdf5_file,
                                                       vsini_values=vsini_values, vac2air=True, logspace=True)
        sensitivity = None

    get_weights = True if addmode.lower() == "weighted" or addmode.lower() == 'simple-weighted' else False
    orderweights = None

    MS = SpectralTypeRelations.MainSequence()

    # Do the cross-correlation
    datadict = defaultdict(list)
    temperature_dict = defaultdict(float)
    vbary_dict = defaultdict(float)
    alpha = 0.0
    for temp in sorted(modeldict.keys()):
        for gravity in sorted(modeldict[temp].keys()):
            for metallicity in sorted(modeldict[temp][gravity].keys()):
                for vsini_sec in vsini_values:
                    if debug:
                        logging.info('T: {}, logg: {}, [Fe/H]: {}, vsini: {}'.format(temp, gravity,
                                                                                     metallicity, vsini_sec))
                    # broaden the model
                    model = modeldict[temp][gravity][metallicity][alpha][vsini_sec].copy()
                    l_idx = 0 if min_x is None else np.searchsorted(model.x, min_x)
                    r_idx = model.size() if max_x is None else np.searchsorted(model.x, max_x)
                    model = Broaden.RotBroad(model[l_idx:r_idx], vsini_sec * u.km.to(u.cm), linear=True)
                    if resolution is not None:
                        model = FittingUtilities.ReduceResolutionFFT(model, resolution)

                    # Interpolate the temperature weights, if addmode='T-weighted'
                    if addmode.lower() == 't-weighted':
                        x = modeldict[temp][gravity][metallicity][alpha][vsini_sec].x
                        y = sensitivity[temp][gravity][metallicity][alpha][vsini_sec]
                        temperature_weights = spline(x, y)

                    for i, (fname, vsini_prim) in enumerate(zip(fileList, primary_vsini)):
                        if vbary_correct:
                            if fname in vbary_dict:
                                vbary = vbary_dict[fname]
                            else:
                                vbary = HelCorr_IRAF(fits.getheader(fname), observatory=observatory)
                                vbary_dict[fname] = vbary
                        process_data = False if fname in datadict else True
                        if process_data:
                            orders = Process_Data(fname, badregions, interp_regions=interp_regions, logspacing=True,
                                                  extensions=extensions, trimsize=trimsize, vsini=vsini_prim,
                                                  reject_outliers=reject_outliers)
                            header = fits.getheader(fname)
                            try:
                                spt = StarData.GetData(header['object']).spectype
                                if spt == 'Unknown':
                                    temperature_dict[fname] = np.nan  # Unknown
                                    logging.warning('Spectral type retrieval from simbad failed! Entering NaN for primary temperature!')
                                else:
                                    match = re.search('[0-9]', spt)
                                    if match is None:
                                        spt = spt[0] + "5"
                                    else:
                                        spt = spt[:match.start() + 1]
                                    temperature_dict[fname] = MS.Interpolate(MS.Temperature, spt)
                            except AttributeError:
                                temperature_dict[fname] = np.nan  # Unknown
                                logging.warning('Spectral type retrieval from simbad failed! Entering NaN for primary temperature!')
                            datadict[fname] = orders
                        else:
                            orders = datadict[fname]

                        # Now, process the model
                        model_orders = process_model(model.copy(), orders, vsini_primary=vsini_prim, maxvel=1000.0,
                                                     debug=debug, oversample=1, logspace=False)

                        # Get order weights if addmode='T-weighted'
                        if addmode.lower() == 't-weighted':
                            get_weights = False
                            orderweights = [np.sum(temperature_weights(o.x)) for o in orders]
                            addmode = 'simple-weighted'

                        if debug and makeplots:
                            fig = plt.figure('T={}   vsini={}'.format(temp, vsini_sec))
                            for o, m in zip(orders, model_orders):
                                d_scale = np.std(o.y/o.cont)
                                m_scale = np.std(m.y/m.cont)
                                plt.plot(o.x, (o.y/o.cont-1.0)/d_scale, 'k-', alpha=0.4)
                                plt.plot(m.x, (m.y/m.cont-1.0)/m_scale, 'r-', alpha=0.6)
                            plt.show(block=False)

                        # Make sure the output directory exists
                        output_dir = "Cross_correlations/"
                        outfilebase = fname.split(".fits")[0]
                        if "/" in fname:
                            dirs = fname.split("/")
                            outfilebase = dirs[-1].split(".fits")[0]
                            if obstype.lower() == 'synthetic':
                                output_dir = ""
                                for directory in dirs[:-1]:
                                    output_dir = output_dir + directory + "/"
                                output_dir = output_dir + "Cross_correlations/"
                        HelperFunctions.ensure_dir(output_dir)

                        # Save the model and data orders, if debug=True
                        if debug:
                            # Save the individual spectral inputs and CCF orders (unweighted)
                            output_dir2 = output_dir.replace("Cross_correlations", "CCF_inputs")
                            HelperFunctions.ensure_dir(output_dir2)
                            HelperFunctions.ensure_dir("%sCross_correlations/" % (output_dir2))

                            for i, (o, m) in enumerate(zip(orders, model_orders)):
                                outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.data.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini_sec,
                                    temp, gravity,
                                    metallicity, i + 1)
                                o.output(outfilename)
                                outfilename = "{0:s}{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.model.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini_sec,
                                    temp, gravity,
                                    metallicity, i + 1)
                                m.output(outfilename)

                        corr = Correlate.Correlate(orders, model_orders, addmode=addmode, outputdir=output_dir,
                                                   get_weights=get_weights, prim_teff=temperature_dict[fname],
                                                   orderweights=orderweights, debug=debug)
                        if debug:
                            corr, ccf_orders = corr

                        # Barycentric correction
                        if vbary_correct:
                            corr.x += vbary

                        # Output the ccf
                        if obstype.lower() == 'synthetic':
                            pars = {'outdir': output_dir, 'outbase': outfilebase, 'addmode': addmode,
                                    'vsini_prim': vsini_prim, 'vsini': vsini_sec,
                                    'T': temp, 'logg': gravity, '[Fe/H]': metallicity}
                            save_synthetic_ccf(corr, params=pars, mode=output_mode)
                        else:
                            pars = {'outdir': output_dir, 'fname': fname, 'addmode': addmode,
                                    'vsini_prim': vsini_prim, 'vsini': vsini_sec,
                                    'T': temp, 'logg': gravity, '[Fe/H]': metallicity}
                            pars['vbary'] = vbary if vbary_correct else np.nan
                            save_ccf(corr, params=pars, mode=output_mode, hdf_outfilename=output_file)

                        # Save the individual orders, if debug=True
                        if debug:
                            for i, c in enumerate(ccf_orders):
                                print "Saving CCF inputs for order {}".format(i + 1)
                                outfilename = "{0:s}Cross_correlations/{1:s}.{2:.0f}kps_{3:.1f}K{4:+.1f}{5:+.1f}.order{6:d}".format(
                                    output_dir2,
                                    outfilebase, vsini_sec,
                                    temp, gravity,
                                    metallicity, i + 1)
                                c.output(outfilename)



                    # Delete the model. We don't need it anymore and it just takes up ram.
                    modeldict[temp][gravity][metallicity][alpha][vsini_sec] = []

    return


def save_synthetic_ccf(corr, params, mode='text', update=True, hdf_outfilename='CCF.hdf5'):
    """
    Save the cross-correlation function in the given way.
    :param: corr: The DataStructures object holding the cross-correlation function
    :param params: A dictionary describing the metadata to include
    :param mode: See docstring for slow_companion_search, param output_mode
    :param update: If mode='hdf5', DO I NEED THIS?
    """
    if mode.lower() == 'text':
        outfilename = "{0:s}{1:s}_{2:s}-method.{3:.0f}kps_{4:.1f}K{5:+.1f}{6:+.1f}".format(params['outdir'],
                                                                                           params['outbase'],
                                                                                           params['addmode'],
                                                                                           params['vsini'],
                                                                                           params['T'],
                                                                                           params['logg'],
                                                                                           params['[Fe/H]'])
        print('Outputting to {}'.format(outfilename))
        np.savetxt(outfilename, np.transpose((corr.x, corr.y)), fmt="%.10g")

    elif mode.lower() == 'hdf5':
        # Get the hdf5 file
        hdf5_file = os.path.join(params['outdir'], hdf_outfilename)
        print('Saving CCF to {}'.format(hdf5_file))
        f = h5py.File(hdf5_file, 'a')

        # Star combination
        segments = params['outbase'].split('_bright')[0].replace('_', ' ')  # .split('_')[:-1]
        # str = ' '.join(segments)
        star1 = segments.split('+')[0]
        star2 = segments.split('+')[1]

        # Make the heirarchy if the file does not have it
        p = f[star1] if star1 in f.keys() else f.create_group(star1)
        s = p[star2] if star2 in p.keys() else p.create_group(star2)
        g = s[params['addmode']] if params['addmode'] in s.keys() else s.create_group(params['addmode'])

        # Add a new dataset. The name doesn't matter
        current_datasets = g.keys()
        if len(current_datasets) == 0:
            ds = g.create_dataset('ds1', data=corr.y)
        else:
            ds_num = max(int(d[2:]) for d in current_datasets) + 1
            ds = g.create_dataset('ds{}'.format(ds_num), data=corr.y)

        # Add attributes to the dataset
        print(star1, star2)
        print(params)
        ds.attrs['vsini'] = params['vsini']
        ds.attrs['T'] = params['T']
        ds.attrs['logg'] = params['logg']
        ds.attrs['[Fe/H]'] = params['[Fe/H]']
        ds.attrs['velocity'] = corr.x

        p.attrs['vsini'] = params['vsini_prim']

        f.flush()
        f.close()

    else:
        raise ValueError('output mode ({}) not supported!'.format(mode))


def save_ccf(corr, params, mode='text', update=False, hdf_outfilename='CCF.hdf5'):
    """
    Save the cross-correlation function in the given way.
    :param: corr: The DataStructures object holding the cross-correlation function
    :param params: A dictionary describing the metadata to include
    :param mode: See docstring for slow_companion_search, param output_mode
    :param update: If mode='hdf5', DO I NEED THIS?
    """

    # Loop through the add-modes if addmode=all
    if params['addmode'].lower() == 'all':
        for am in corr.keys():
            p = dict(**params)
            p['addmode'] = am
            save_ccf(corr[am], p, mode=mode, update=update, hdf_outfilename=hdf_outfilename)
        return

    if mode.lower() == 'text':
        params['outbase'] = params['fname'].split('/')[-1].split('.fits')[0]
        save_synthetic_ccf(corr, params, mode=mode, update=update)

    elif mode.lower() == 'hdf5':
        # Get the hdf5 file
        hdf5_file = os.path.join(params['outdir'], hdf_outfilename)
        print('Saving CCF to {}'.format(hdf5_file))
        f = h5py.File(hdf5_file, 'a')

        # Star name and date
        header = fits.getheader(params['fname'])
        star = header['OBJECT']
        date = header['DATE-OBS'].split('T')[0]

        print(star, date)
        print(params)
        if star in f.keys():
            s = f[star]
        else:
            star_data = StarData.GetData(star)
            s = f.create_group(star)
            s.attrs['vsini'] = -1 if params['vsini_prim'] is None else params['vsini_prim']
            s.attrs['RA'] = star_data.ra
            s.attrs['DEC'] = star_data.dec
            s.attrs['SpT'] = star_data.spectype

        d = s[date] if date in s.keys() else s.create_group(date)

        # Add a new dataset. The name doesn't matter
        attr_pars = ['vbary'] if 'vbary' in params else []
        attr_pars.extend(['vsini', 'T', 'logg', '[Fe/H]', 'addmode', 'fname'])
        """
        for ds_name in d.keys():
            ds_test = d[ds_name]
            if 'fname' in ds_test.attrs:
                a_pars = attr_pars
            else:
                a_pars = attr_pars[:-1]
            if all([HelperFunctions.is_close(ds_test.attrs[a], params[a]) for a in a_pars]):
                if update:
                    ds = ds_test
                    new_data = np.array((corr.x, corr.y))
                    try:
                        ds.resize(new_data.shape)
                    except TypeError:
                        # Hope for the best...
                        pass
                    ds[:] = np.array((corr.x, corr.y))
                    idx = np.argmax(corr.y)
                    ds.attrs['vel_max'] = corr.x[idx]
                    ds.attrs['ccf_max'] = corr.y[idx]
                    f.flush()
                    f.close()
                return
        """


        # If we get here, no matching dataset was found.
        ds_name = 'T{}_logg{}_metal{}_addmode-{}_vsini{}'.format(params['T'],
                                                                 params['logg'],
                                                                 params['[Fe/H]'],
                                                                 params['addmode'],
                                                                 params['vsini'])
        if ds_name in d.keys():
            if update:
                ds = d[ds_name]
                new_data = np.array((corr.x, corr.y))
                try:
                    ds.resize(new_data.shape)
                except TypeError:
                    # Hope for the best...
                    pass
                ds[:] = np.array((corr.x, corr.y))
                f.flush()
                f.close()
                return
            else:
                i = 1
                while '{}_{}'.format(ds_name, i) in d.keys():
                    i += 1
                ds_name = '{}_{}'.format(ds_name, i)

        ds = d.create_dataset(ds_name, data=np.array((corr.x, corr.y)), maxshape=(2, None))

        # Add attributes to the dataset
        for a in attr_pars:
            ds.attrs[a] = params[a]
        idx = np.argmax(corr.y)
        ds.attrs['vel_max'] = corr.x[idx]
        ds.attrs['ccf_max'] = corr.y[idx]

        f.flush()
        f.close()

    else:
        raise ValueError('output mode ({}) not supported!'.format(mode))

    return
