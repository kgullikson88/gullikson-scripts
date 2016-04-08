from __future__ import print_function, absolute_import, division

import os
import sys
import re
from collections import defaultdict
import warnings
from collections import OrderedDict
import itertools
import logging

from astropy import units
from scipy.interpolate import InterpolatedUnivariateSpline as spline, LinearNDInterpolator, NearestNDInterpolator, \
    interp1d
import numpy as np
import h5py

from kglib.utils import FittingUtilities, DataStructures
from kglib.stellar_models import Broaden
import pandas


"""
This code provides the GetModelList function.
It is used in GenericSearch.py and SensitivityAnalysis.py
"""

# Define file locations on my systems...
if "darwin" in sys.platform:
    modeldir = "/Volumes/DATADRIVE/Stellar_Models/Sorted/Stellar/Vband/"
    HDF5_FILE = '/Volumes/DATADRIVE/PhoenixGrid/Search_Grid.hdf5'
elif "linux" in sys.platform:
    modeldir = "/media/FreeAgent_Drive/SyntheticSpectra/Sorted/Stellar/Vband/"
    HDF5_FILE = '/media/ExtraSpace/PhoenixGrid/Search_Grid.hdf5'
else:
    modeldir = raw_input("sys.platform not recognized. Please enter model directory below: ")
    if not modeldir.endswith("/"):
        modeldir = modeldir + "/"


def GetModelList(type='phoenix',
                 metal=[-0.5, 0, 0.5],
                 logg=[4.5, ],
                 temperature=range(3000, 6900, 100),
                 alpha=[0, 0.2],
                 model_directory=modeldir,
                 hdf5_file=HDF5_FILE):
    """
    This function searches the model directory for stellar
    models with the appropriate parameters

    Parameters:
    ===========
    - type:                string
                           The type of models to get. Right now, only 'phoenix',
                           'kurucz', and 'hdf5' are implemented.

    - metal:               iterable
                           A list of the metallicities to include

    - logg:                iterable
                           A list of the surface gravity values to include

    - temperature:         iterable
                           A list of the temperatures to include

    - alpha:               iterable
                           A list of the [alpha/Fe] values to include

    - model_directory:     string
                           The absolute path to the model directory
                           (only used for type=phoenix or kurucz)

    - hdf5_file:           string
                           The absolute path to the HDF5 file with the models
                           (only used for type=hdf5)

    Returns:
    ========
    A list of filenames for the requested models, or a list of parameters if type='hdf5'
    """

    # Error checking
    metal = np.atleast_1d(metal)
    logg = np.atleast_1d(logg)
    temperature = np.atleast_1d(temperature)
    alpha = np.atleast_1d(alpha)

    if type.lower() == 'phoenix':
        all_models = sorted([f for f in os.listdir(model_directory) if 'phoenix' in f.lower()])
        chosen_models = []
        for model in all_models:
            Teff, gravity, metallicity = ClassifyModel(model)
            if Teff in temperature and gravity in logg and metallicity in metal:
                chosen_models.append("{:s}{:s}".format(model_directory, model))


    elif type.lower() == "kurucz":
        all_models = [f for f in os.listdir(modeldir) if f.startswith("t") and f.endswith(".dat.bin.asc")]
        chosen_models = []
        for model in all_models:
            Teff, gravity, metallicity, a = ClassifyModel(model, type='kurucz')
            if Teff in temperature and gravity in logg and metallicity in metal and a in alpha:
                chosen_models.append("{:s}{:s}".format(model_directory, model))


    elif type.lower() == 'hdf5':
        hdf5_int = HDF5Interface(hdf5_file)
        chosen_models = []
        for par in hdf5_int.list_grid_points:
            if par['temp'] in temperature and par['logg'] in logg and par['Z'] in metal and par['alpha'] in alpha:
                chosen_models.append(par)

    else:
        raise NotImplementedError("Sorry, the model type ({:s}) is not available!".format(type))

    return chosen_models


def ClassifyModel(filename, type='phoenix'):
    """
    Get the effective temperature, log(g), and [Fe/H] of a stellar model from the filename

    Parameters:
    ===========
    - filename:   string
                  The filename to classify

    - type:       string
                  The type of file. Currently, only
                  phoenix and kurucz type files are supported

    Returns:
    ========
    temperature, surface gravity, and metallicity (all floats)
    """
    if not isinstance(filename, basestring):
        raise TypeError("Filename must be a string!")

    if type.lower() == 'phoenix':
        segments = re.split("-|\+", filename.split("/")[-1])
        temp = float(segments[0].split("lte")[-1]) * 100
        gravity = float(segments[1])
        metallicity = float(segments[2][:3])
        if not "+" in filename and metallicity > 0:
            metallicity *= -1
        return temp, gravity, metallicity

    elif type.lower() == 'kurucz':
        fname = filename.split('/')[-1]
        temp = float(fname[1:6])
        gravity = float(fname[8:12])
        metallicity = float(fname[14:16]) / 10.0
        alpha = float(fname[18:20]) / 10.0
        if fname[13] == "m":
            metallicity *= -1
        if fname[17] == "m":
            alpha *= -1
        return temp, gravity, metallicity, alpha

    else:
        raise NotImplementedError("Sorry, the model type ({:s}) is not available!".format(type))

    return temp, gravity, metallicity


def MakeModelDicts(model_list, vsini_values=[10, 20, 30, 40], type='phoenix',
                   vac2air=True, logspace=False, hdf5_file=HDF5_FILE, get_T_sens=False):
    """
    This will take a list of models, and output two dictionaries that are
    used by GenericSearch.py and Sensitivity.py

    Parameters:
    ===========
    - model_list:        iterable
                         A list of model filenames

    - vsini_values:     iterable
                        A list of vsini values to broaden
                        the spectrum by (we do that later!)

    - type:             string
                        The type of models. Currently,
                        phoenix, kurucz, and hdf5 are implemented

    - vac2air:          boolean
                        If true, assumes the model is in
                        vacuum wavelengths and converts to air

    - logspace:         boolean
                        If true, it will rebin the
                        data to a constant log-spacing

    - hdf5_file:        string
                        The absolute path to the HDF5 file
                        with the models. Only used if type=hdf5

    - get_T_sens:       boolean
                        Flag for getting the temperature sensitivity.
                        If true, it finds the derivative of each pixel dF/dT

    Returns:
    ========
     A dictionary containing the model with keys of temperature, gravity,
     metallicity, and vsini,and another one with a processed flag with the same keys
    """
    vsini_values = np.atleast_1d(vsini_values)
    if type.lower() == 'phoenix':
        modeldict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(DataStructures.xypoint))))
        processed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(bool))))
        for fname in model_list:
            temp, gravity, metallicity = ClassifyModel(fname)
            print("Reading in file %s" % fname)
            data = pandas.read_csv(fname,
                                   header=None,
                                   names=["wave", "flux"],
                                   usecols=(0, 1),
                                   sep=' ',
                                   skipinitialspace=True)
            x, y = data['wave'].values, data['flux'].values
            if vac2air:
                n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
                x /= n
            model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm), y=10 ** y)
            if logspace:
                xgrid = np.logspace(np.log(model.x[0]), np.log(model.x[-1]), model.size(), base=np.e)
                model = FittingUtilities.RebinData(model, xgrid)
            for vsini in vsini_values:
                modeldict[temp][gravity][metallicity][vsini] = model
                processed[temp][gravity][metallicity][vsini] = False

    elif type.lower() == 'kurucz':
        modeldict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(DataStructures.xypoint)))))
        processed = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))))
        for fname in model_list:
            temp, gravity, metallicity, a = ClassifyModel(fname)
            print("Reading in file %s" % fname)
            data = pandas.read_csv(fname,
                                   header=None,
                                   names=["wave", "flux"],
                                   usecols=(0, 1),
                                   sep=' ',
                                   skipinitialspace=True)
            x, y = data['wave'].values, data['flux'].values
            if vac2air:
                n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
                x /= n
            model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm), y=10 ** y)
            if logspace:
                xgrid = np.logspace(np.log(model.x[0]), np.log(model.x[-1]), model.size(), base=np.e)
                model = FittingUtilities.RebinData(model, xgrid)
            for vsini in vsini_values:
                modeldict[temp][gravity][metallicity][a][vsini] = model
                processed[temp][gravity][metallicity][a][vsini] = False

    elif type.lower() == 'hdf5':
        hdf5_int = HDF5Interface(hdf5_file)
        x = hdf5_int.wl
        wave_hdr = hdf5_int.wl_header
        if vac2air:
            if not wave_hdr['air']:
                n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
                x /= n
        elif wave_hdr['air']:
            raise GridError(
                'HDF5 grid is in air wavelengths, but you requested vacuum wavelengths. You need a new grid!')
        modeldict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(DataStructures.xypoint)))))
        processed = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))))
        for pars in model_list:
            temp, gravity, metallicity, a = pars['temp'], pars['logg'], pars['Z'], pars['alpha']
            y = hdf5_int.load_flux(pars)
            model = DataStructures.xypoint(x=x * units.angstrom.to(units.nm), y=y)
            for vsini in vsini_values:
                modeldict[temp][gravity][metallicity][a][vsini] = model
                processed[temp][gravity][metallicity][a][vsini] = False

    else:
        raise NotImplementedError("Sorry, the model type ({:s}) is not available!".format(type))

    if get_T_sens:
        # Get the temperature sensitivity. Warning! This assumes the wavelength grid is the same in all models.
        sensitivity = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(DataStructures.xypoint)))))
        Tvals = sorted(modeldict.keys())
        for i, T in enumerate(Tvals):
            gvals = sorted(modeldict[T].keys())
            for gravity in gvals:
                metal_vals = sorted(modeldict[T][gravity].keys())
                for metal in metal_vals:
                    alpha_vals = sorted(modeldict[T][gravity][metal].keys())
                    for alpha in alpha_vals:
                        # get the temperature just under this one
                        lower, l_idx = get_model(modeldict, Tvals, i, gravity, metal, vsini_values[0], alpha, mode='lower')
                        upper, u_idx = get_model(modeldict, Tvals, i, gravity, metal, vsini_values[0], alpha, mode='upper')
                        T_low = Tvals[l_idx]
                        T_high = Tvals[u_idx]
                        slope = (upper.y - lower.y) / (T_high - T_low)
                        for vsini in vsini_values:
                            sensitivity[T][gravity][metal][alpha][vsini] = slope**2
        return modeldict, processed, sensitivity

    return modeldict, processed


def get_model(mdict, Tvals, i, logg, metal, vsini, alpha=None, mode='same'):
    """
    Get the model with the requested parameters

    Parameters:
    ===========
    - mdict:     5x-nested dictionary, such as generated by MakeModelDicts
                 The model dictionary
    - Tvals:     List of floats
                 A list of temperatures in the first level of mdict

    - i:         integer
                 The index of the requested temperature within the Tvals list

    - logg, metal, vsini, alpha:  float
                                  The parameter you want. These index into mdict.

    - mode:      How to get the model. valid options:
                    - 'same': Get the model with the exact requested parameters.
                    - 'lower': Get model with the exact values of everything
                       except temperature (find the next lowest temperature)
                    - 'upper': Get model with the exact values of everything
                       except temperature (find the next highest temperature)
    """
    if mode == 'same':
        if alpha is None:
            mdict[Tvals[i]][logg][metal][vsini]
        else:
            return mdict[Tvals[i]][logg][metal][alpha][vsini]
    elif mode == 'lower':
        done = False
        idx = i - 1
        idx = max(0, idx)
        idx = min(len(Tvals), idx)
        while not done:
            if idx == 0 or idx == len(Tvals) - 1:
                return get_model(mdict, Tvals, idx, logg, metal, vsini, alpha, mode='same'), idx
            try:
                return get_model(mdict, Tvals, idx, logg, metal, vsini, alpha, mode='same'), idx
            except KeyError:
                idx -= 1
    elif mode == 'upper':
        done = False
        idx = i +1
        idx = max(0, idx)
        idx = min(len(Tvals)-1, idx)
        while not done:
            if idx == 0 or idx == len(Tvals) - 1:
                return get_model(mdict, Tvals, idx, logg, metal, vsini, alpha, mode='same'), idx
            try:
                return get_model(mdict, Tvals, idx, logg, metal, vsini, alpha, mode='same'), idx
            except KeyError:
                idx += 1


####################################################################
#         The next several classes/functions are from
#                Ian Czekala's Starfish code
#
####################################################################


class HDF5Interface:
    '''
    Connect to an HDF5 file that stores spectra. Stolen shamelessly from Ian Czekala's Starfish code
    '''

    def __init__(self, filename, ranges={"temp": (0, np.inf),
                                         "logg": (-np.inf, np.inf),
                                         "Z": (-np.inf, np.inf),
                                         "alpha": (-np.inf, np.inf)}):
        '''
            :param filename: the name of the HDF5 file
            :type param: string
            :param ranges: optionally select a smaller part of the grid to use.
            :type ranges: dict
        '''
        self.filename = filename
        self.flux_name = "t{temp:.0f}g{logg:.1f}z{Z:.1f}a{alpha:.1f}"
        grid_parameters = ("temp", "logg", "Z", "alpha")  # Allowed grid parameters
        grid_set = frozenset(grid_parameters)

        with h5py.File(self.filename, "r") as hdf5:
            self.wl = hdf5["wl"][:]
            self.wl_header = dict(hdf5["wl"].attrs.items())

            grid_points = []

            for key in hdf5["flux"].keys():
                # assemble all temp, logg, Z, alpha keywords into a giant list
                hdr = hdf5['flux'][key].attrs

                params = {k: hdr[k] for k in grid_set}

                #Check whether the parameters are within the range
                for kk, vv in params.items():
                    low, high = ranges[kk]
                    if (vv < low) or (vv > high):
                        break
                else:
                    #If all parameters have passed successfully through the ranges, allow.
                    grid_points.append(params)

            self.list_grid_points = grid_points

        # determine the bounding regions of the grid by sorting the grid_points
        temp, logg, Z, alpha = [], [], [], []
        for param in self.list_grid_points:
            temp.append(param['temp'])
            logg.append(param['logg'])
            Z.append(param['Z'])
            alpha.append(param['alpha'])

        self.bounds = {"temp": (min(temp), max(temp)),
                       "logg": (min(logg), max(logg)),
                       "Z": (min(Z), max(Z)),
                       "alpha": (min(alpha), max(alpha))}

        self.points = {"temp": np.unique(temp),
                       "logg": np.unique(logg),
                       "Z": np.unique(Z),
                       "alpha": np.unique(alpha)}

        self.ind = None  #Overwritten by other methods using this as part of a ModelInterpolator

    def load_flux(self, parameters):
        '''
        Load just the flux from the grid, with possibly an index truncation.

        :param parameters: the stellar parameters
        :type parameters: dict

        :raises KeyError: if spectrum is not found in the HDF5 file.

        :returns: flux array
        '''

        key = self.flux_name.format(**parameters)
        with h5py.File(self.filename, "r") as hdf5:
            try:
                if self.ind is not None:
                    fl = hdf5['flux'][key][self.ind[0]:self.ind[1]]
                else:
                    fl = hdf5['flux'][key][:]
            except KeyError as e:
                raise GridError(e)

        # Note: will raise a KeyError if the file is not found.

        return fl


grid_parameters = ("temp", "logg", "Z", "alpha")  # Allowed grid parameters
grid_set = frozenset(grid_parameters)
var_default = {"temp": 5800, "logg": 4.5, "Z": 0.0, "alpha": 0.0,
               "vsini": 0.0, "FWHM": 0.0, "vz": 0.0, "Av": 0.0, "logOmega": 0.0}


class IndexInterpolator:
    '''
    Object to return fractional distance between grid points of a single grid variable.

    :param parameter_list: list of parameter values
    :type parameter_list: 1-D list
    '''

    def __init__(self, parameter_list):
        self.parameter_list = np.unique(parameter_list)
        self.index_interpolator = interp1d(self.parameter_list, np.arange(len(self.parameter_list)), kind='linear')
        pass

    def __call__(self, value):
        '''
        Evaluate the interpolator at a parameter.

        :param value:
        :type value: float
        :raises C.InterpolationError: if *value* is out of bounds.

        :returns: ((low_val, high_val), (frac_low, frac_high)), the lower and higher bounding points in the grid
        and the fractional distance (0 - 1) between them and the value.
        '''
        try:
            index = self.index_interpolator(value)
        except ValueError as e:
            raise InterpolationError("Requested value {} is out of bounds. {}".format(value, e))
        high = np.ceil(index)
        low = np.floor(index)
        frac_index = index - low
        return ((self.parameter_list[low], self.parameter_list[high]), ((1 - frac_index), frac_index))


class Interpolator:
    '''
    Quickly and efficiently interpolate a synthetic spectrum for use in an MCMC simulation. Caches spectra for
    easier memory load.

    :param interface: :obj:`HDF5Interface` (recommended) or :obj:`RawGridInterface` to load spectra
    :param DataSpectrum: data spectrum that you are trying to fit. Used for truncating the synthetic spectra to the relevant region for speed.
    :type DataSpectrum: :obj:`spectrum.DataSpectrum`
    :param cache_max: maximum number of spectra to hold in cache
    :type cache_max: int
    :param cache_dump: how many spectra to purge from the cache once :attr:`cache_max` is reached
    :type cache_dump: int
    :param trilinear: Should this interpolate in temp, logg, and [Fe/H] AND [alpha/Fe], or just the first three parameters.
    :type trilinear: bool

    Setting :attr:`trilinear` to **True** is useful for when you want to do a run with [Fe/H] > 0.0

    '''

    def __init__(self, interface, DataSpectrum, cache_max=256, cache_dump=64, trilinear=False, log=True):
        '''
        Param log decides how to chunk up the returned spectrum. If we are using a pre-instrument convolved grid,
        then we want to use log=True.

        If we are using the raw synthetic grid, then we want to use log=False.
        '''
        self.interface = interface
        self.DataSpectrum = DataSpectrum

        # If alpha only includes one value, then do trilinear interpolation
        (alow, ahigh) = self.interface.bounds['alpha']
        if (alow == ahigh) or trilinear:
            self.parameters = grid_set - set(("alpha",))
        else:
            self.parameters = grid_set

        self.wl = self.interface.wl
        self.wl_dict = self.interface.wl_header
        if log:
            self._determine_chunk_log()
        else:
            self._determine_chunk()
        self.setup_index_interpolators()
        self.cache = OrderedDict([])
        self.cache_max = cache_max
        self.cache_dump = cache_dump  #how many to clear once the maximum cache has been reached


    def _determine_chunk_log(self, tol=50):
        '''
        Using the DataSpectrum, determine the minimum chunksize that we can use and then truncate the synthetic
        wavelength grid and the returned spectra.

        Assumes HDF5Interface is LogLambda spaced, because otherwise you shouldn't need a grid with 2^n points,
        because you would need to interpolate in wl space after this anyway.
        '''

        wave_grid = self.interface.wl
        wl_min, wl_max = np.min(self.DataSpectrum.wls) - tol, np.max(self.DataSpectrum.wls) + tol
        # Length of the raw synthetic spectrum
        len_wg = len(wave_grid)
        #ind_wg = np.arange(len_wg) #Labels of pixels
        #Length of the data
        len_data = np.sum(
            (self.wl > wl_min - tol) & (self.wl < wl_max + tol))  # How much of the synthetic spectrum do we need?

        #Find the smallest length synthetic spectrum that is a power of 2 in length and larger than the data spectrum
        chunk = len_wg
        self.interface.ind = (0, chunk)  #Set to be the full spectrum

        while chunk > len_data:
            if chunk / 2 > len_data:
                chunk = chunk // 2
            else:
                break

        assert type(chunk) == np.int, "Chunk is no longer integer!. Chunk is {}".format(chunk)

        if chunk < len_wg:
            # Now that we have determined the length of the chunk of the synthetic spectrum, determine indices
            # that straddle the data spectrum.

            # What index corresponds to the wl at the center of the data spectrum?
            median_wl = np.median(self.DataSpectrum.wls)
            median_ind = (np.abs(wave_grid - median_wl)).argmin()

            #Take the chunk that straddles either side.
            ind = [median_ind - chunk // 2, median_ind + chunk // 2]
            if ind[0] < 0:
                ind[1] -= ind[0]
                ind[0] = 0
            elif ind[1] >= len_wg:
                ind[0] -= (ind[1] - len_wg - 1)
                ind[1] -= (ind[1] - len_wg - 1)
            ind = tuple(ind)

            self.wl = self.wl[ind[0]:ind[1]]
            assert min(self.wl) < wl_min and max(self.wl) > wl_max, "ModelInterpolator chunking ({:.2f}, {:.2f}) " \
                                                                    "didn't encapsulate full DataSpectrum range ({:.2f}, {:.2f}).".format(
                min(self.wl),
                max(self.wl), wl_min, wl_max)

            self.interface.ind = ind

        print("Determine Chunk Log: Wl is {}".format(len(self.wl)))

    def _determine_chunk(self):
        '''
        Using the DataSpectrum, set the bounds of the interpolator to +/- 50 Ang
        '''

        wave_grid = self.interface.wl
        wl_min, wl_max = np.min(self.DataSpectrum.wls), np.max(self.DataSpectrum.wls)

        ind_low = (np.abs(wave_grid - (wl_min - 50.))).argmin()
        ind_high = (np.abs(wave_grid - (wl_max + 50.))).argmin()

        self.wl = self.wl[ind_low:ind_high]

        assert min(self.wl) < wl_min and max(self.wl) > wl_max, "ModelInterpolator chunking ({:.2f}, {:.2f}) " \
                                                                "didn't encapsulate full DataSpectrum range ({:.2f}, {:.2f}).".format(
            min(self.wl),
            max(self.wl), wl_min, wl_max)

        self.interface.ind = (ind_low, ind_high)
        print("Wl is {}".format(len(self.wl)))


    def __call__(self, parameters):
        '''
        Interpolate a spectrum

        :param parameters: stellar parameters
        :type parameters: dict

        Automatically pops :attr:`cache_dump` items from cache if full.
        '''
        if len(self.cache) > self.cache_max:
            [self.cache.popitem(False) for i in range(self.cache_dump)]
            self.cache_counter = 0
        try:
            return self.interpolate(parameters)
        except:
            logging.warning('Warning! Interpolation error found! Returning ones array!')
            return np.ones_like(self.wl)

    def setup_index_interpolators(self):
        # create an interpolator between grid points indices. Given a temp, produce fractional index between two points
        self.index_interpolators = {key: IndexInterpolator(self.interface.points[key]) for key in self.parameters}

        lenF = self.interface.ind[1] - self.interface.ind[0]
        self.fluxes = np.empty((2 ** len(self.parameters), lenF))  #8 rows, for temp, logg, Z

    def interpolate(self, parameters):
        '''
        Interpolate a spectrum without clearing cache. Recommended to use :meth:`__call__` instead.

        :param parameters: stellar parameters
        :type parameters: dict
        :raises C.InterpolationError: if parameters are out of bounds.

        Now the interpolator also returns the 24 error spectra along with weights.
        '''

        # Here it really would be nice to return things in a predictable order
        # (temp, logg, Z)
        odict = OrderedDict()
        for key in ("temp", "logg", "Z"):
            odict[key] = parameters[key]
        try:
            edges = OrderedDict()
            for key, value in odict.items():
                edges[key] = self.index_interpolators[key](value)
        except InterpolationError as e:
            raise InterpolationError("Parameters {} are out of bounds. {}".format(parameters, e))

        #Edges is a dictionary of {"temp": ((6000, 6100), (0.2, 0.8)), "logg": (())..}
        names = [key for key in edges.keys()]  #list of ["temp", "logg", "Z"],
        params = [edges[key][0] for key in names]  #[(6000, 6100), (4.0, 4.5), ...]
        weights = [edges[key][1] for key in names]  #[(0.2, 0.8), (0.4, 0.6), ...]

        param_combos = itertools.product(*params)  #Selects all the possible combinations of parameters
        #[(6000, 4.0, 0.0), (6100, 4.0, 0.0), (6000, 4.5, 0.0), ...]
        weight_combos = itertools.product(*weights)
        #[(0.2, 0.4, 1.0), (0.8, 0.4, 1.0), ...]

        parameter_list = [dict(zip(names, param)) for param in param_combos]
        if "alpha" not in parameters.keys():
            [param.update({"alpha": var_default["alpha"]}) for param in parameter_list]
        key_list = [self.interface.flux_name.format(**param) for param in parameter_list]
        weight_list = np.array([np.prod(weight) for weight in weight_combos])

        assert np.allclose(np.sum(weight_list), np.array(1.0)), "Sum of weights must equal 1, {}".format(
            np.sum(weight_list))

        #Assemble flux vector from cache
        for i, param in enumerate(parameter_list):
            key = key_list[i]
            if key not in self.cache.keys():
                try:
                    fl = self.interface.load_flux(param)  #This method allows loading only the relevant region from HDF5
                except KeyError as e:
                    raise InterpolationError("Parameters {} not in master HDF5 grid. {}".format(param, e))
                self.cache[key] = fl
                #Note: if we are dealing with a ragged grid, a C.GridError will be raised here because a Z=+1, alpha!=0 spectrum can't be found.

            self.fluxes[i, :] = self.cache[key] * weight_list[i]

        return np.sum(self.fluxes, axis=0)


class DataSpectrum:
    '''
    Object to manipulate the data spectrum.

    :param wls: wavelength (in AA)
    :type wls: 1D or 2D np.array
    :param fls: flux (in f_lam)
    :type fls: 1D or 2D np.array
    :param sigmas: Poisson noise (in f_lam)
    :type sigmas: 1D or 2D np.array
    :param masks: Mask to blot out bad pixels or emission regions.
    :type masks: 1D or 2D np.array of boolean values

    If the wl, fl, are provided as 1D arrays (say for a single order), they will be converted to 2D arrays with length 1
    in the 0-axis.

    .. note::

       For now, the DataSpectrum wls, fls, sigmas, and masks must be a rectangular grid. No ragged Echelle orders allowed.

    '''

    def __init__(self, wls, fls, sigmas, masks=None, orders='all', name=None):
        self.wls = np.atleast_2d(wls)
        self.fls = np.atleast_2d(fls)
        self.sigmas = np.atleast_2d(sigmas)
        self.masks = np.atleast_2d(masks) if masks is not None else np.ones_like(self.wls, dtype='b')

        self.shape = self.wls.shape
        assert self.fls.shape == self.shape, "flux array incompatible shape."
        assert self.sigmas.shape == self.shape, "sigma array incompatible shape."
        assert self.masks.shape == self.shape, "mask array incompatible shape."

        if orders != 'all':
            # can either be a numpy array or a list
            orders = np.array(orders)  #just to make sure
            self.wls = self.wls[orders]
            self.fls = self.fls[orders]
            self.sigmas = self.sigmas[orders]
            self.masks = self.masks[orders]
            self.shape = self.wls.shape
            self.orders = orders
        else:
            self.orders = np.arange(self.shape[0])

        self.name = name


class GridError(Exception):
    '''
    Raised when a spectrum cannot be found in the grid.
    '''

    def __init__(self, msg):
        self.msg = msg


class InterpolationError(Exception):
    '''
    Raised when the :obj:`Interpolator` or :obj:`IndexInterpolator` cannot properly interpolate a spectrum,
    usually grid bounds.
    '''

    def __init__(self, msg):
        self.msg = msg


####################################################################
#                       Back to my code.
#
####################################################################


class KuruczGetter():
    def __init__(self, modeldir, rebin=True, T_min=7000, T_max=9000, logg_min=3.5, logg_max=4.5, metal_min=-0.5,
                 metal_max=0.5, alpha_min=0.0, alpha_max=0.4, wavemin=0, wavemax=np.inf, debug=False):
        """
        This class will read in a directory with Kurucz models

        The associated methods can be used to interpolate a model at any
        temperature, gravity, metallicity, and [alpha/Fe] value that
        falls within the grid

        modeldir: The directory where the models are stored. Can be a list of model directories too!
        rebin: If True, it will rebin the models to a constant x-spacing
        other args: The minimum and maximum values for the parameters to search.
                    You need to keep this as small as possible to avoid memory issues!
                    The whole grid would take about 36 GB of RAM!
        """
        from kglib.utils import HelperFunctions
        self.rebin = rebin
        self.debug = debug

        # First, read in the grid
        if HelperFunctions.IsListlike(modeldir):
            # There are several directories to combine
            Tvals = []
            loggvals = []
            metalvals = []
            alphavals = []
            for i, md in enumerate(modeldir):
                if i == 0:
                    T, G, Z, A, S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max, logg_min=logg_min,
                                                   logg_max=logg_max, metal_min=metal_min, metal_max=metal_max,
                                                   alpha_min=alpha_min, alpha_max=alpha_max, wavemin=wavemin,
                                                   wavemax=wavemax,
                                                   xaxis=None)
                    spectra = np.array(S)
                else:
                    T, G, Z, A, S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max, logg_min=logg_min,
                                                   logg_max=logg_max, metal_min=metal_min, metal_max=metal_max,
                                                   alpha_min=alpha_min, alpha_max=alpha_max, wavemin=wavemin,
                                                   wavemax=wavemax,
                                                   xaxis=self.xaxis)
                    S = np.array(S)
                    spectra = np.vstack((spectra, S))

                Tvals = np.hstack((Tvals, T))
                loggvals = np.hstack((loggvals, G))
                metalvals = np.hstack((metalvals, Z))
                alphavals = np.hstack((alphavals, A))
        else:
            Tvals, loggvals, metalvals, alphavals, spectra = self.read_grid(modeldir,
                                                                            rebin=rebin,
                                                                            T_min=T_min,
                                                                            T_max=T_max,
                                                                            logg_min=logg_min,
                                                                            logg_max=logg_max,
                                                                            metal_min=metal_min,
                                                                            metal_max=metal_max,
                                                                            alpha_min=alpha_min,
                                                                            alpha_max=alpha_max,
                                                                            wavemin=wavemin,
                                                                            wavemax=wavemax,
                                                                            xaxis=None)

        # Check if there are actually two different values of alpha/Fe
        alpha_varies = True if max(alphavals) - min(alphavals) > 0.1 else False

        # Scale the variables so they all have about the same range
        self.T_scale = ((max(Tvals) + min(Tvals)) / 2.0, max(Tvals) - min(Tvals))
        self.metal_scale = ((max(metalvals) + min(metalvals)) / 2.0, max(metalvals) - min(metalvals))
        self.logg_scale = ((max(loggvals) + min(loggvals)) / 2.0, max(loggvals) - min(loggvals))
        if alpha_varies:
            self.alpha_scale = ((max(alphavals) + min(alphavals)) / 2.0, max(alphavals) - min(alphavals))
        Tvals = (np.array(Tvals) - self.T_scale[0]) / self.T_scale[1]
        loggvals = (np.array(loggvals) - self.logg_scale[0]) / self.logg_scale[1]
        metalvals = (np.array(metalvals) - self.metal_scale[0]) / self.metal_scale[1]
        if alpha_varies:
            alphavals = (np.array(alphavals) - self.alpha_scale[0]) / self.alpha_scale[1]
        print(self.T_scale)
        print(self.metal_scale)
        print(self.logg_scale)
        if alpha_varies:
            print(self.alpha_scale)

        # Make the grid and interpolator instances
        if alpha_varies:
            self.grid = np.array((Tvals, loggvals, metalvals, alphavals)).T
        else:
            self.grid = np.array((Tvals, loggvals, metalvals)).T
        self.spectra = np.array(spectra)
        self.interpolator = LinearNDInterpolator(self.grid, self.spectra)  # , rescale=True)
        self.NN_interpolator = NearestNDInterpolator(self.grid, self.spectra)  # , rescale=True)
        self.alpha_varies = alpha_varies


    def read_grid(self, modeldir, rebin=True, T_min=7000, T_max=9000, logg_min=3.5, logg_max=4.5, metal_min=-0.5,
                  metal_max=0.5, alpha_min=0.0, alpha_max=0.4, wavemin=0, wavemax=np.inf, xaxis=None):
        Tvals = []
        loggvals = []
        metalvals = []
        alphavals = []
        spectra = []
        firstkeeper = True
        modelfiles = [f for f in os.listdir(modeldir) if f.startswith("t") and f.endswith(".dat.bin.asc")]
        for i, fname in enumerate(modelfiles):
            T = float(fname[1:6])
            logg = float(fname[8:12])
            metal = float(fname[14:16]) / 10.0
            alpha = float(fname[18:20]) / 10.0
            if fname[13] == "m":
                metal *= -1
            if fname[17] == "m":
                alpha *= -1

            # Read in and save file if it falls in the correct parameter range
            if (T_min <= T <= T_max and
                            logg_min <= logg <= logg_max and
                            metal_min <= metal <= metal_max and
                            alpha_min <= alpha <= alpha_max):

                if self.debug:
                    print("Reading in file {:s}".format(fname))
                data = pandas.read_csv("{:s}/{:s}".format(modeldir, fname),
                                       header=None,
                                       names=["wave", "norm"],
                                       usecols=(0, 3),
                                       sep=' ',
                                       skipinitialspace=True)
                x, y = data['wave'].values, data['norm'].values
                # x, y = np.loadtxt("{:s}/{:s}".format(modeldir, fname), usecols=(0, 3), unpack=True)
                x *= units.angstrom.to(units.nm)
                y[np.isnan(y)] = 0.0

                left = np.searchsorted(x, wavemin)
                right = np.searchsorted(x, wavemax)
                x = x[left:right]
                y = y[left:right]

                if rebin:
                    if firstkeeper:
                        xgrid = np.logspace(np.log10(x[0]), np.log10(x[-1]), x.size)
                    else:
                        xgrid = self.xaxis
                    fcn = spline(x, y)
                    x = xgrid
                    y = fcn(xgrid)

                if firstkeeper:
                    self.xaxis = x if xaxis is None else xaxis
                    firstkeeper = False
                elif np.max(np.abs(self.xaxis - x) > 1e-4):
                    warnings.warn("x-axis for file {:s} is different from the master one! Not saving!".format(fname))
                    continue

                Tvals.append(T)
                loggvals.append(logg)
                metalvals.append(metal)
                alphavals.append(alpha)
                spectra.append(y)

        return Tvals, loggvals, metalvals, alphavals, spectra


    def __call__(self, T, logg, metal, alpha, vsini=0.0, return_xypoint=True, **kwargs):
        """
        Given parameters, return an interpolated spectrum

        If return_xypoint is False, then it will only return
          a numpy.ndarray with the spectrum

        Before interpolating, we will do some error checking to make
        sure the requested values fall within the grid
        """

        # Scale the requested values
        if self.debug:
            print(T, logg, metal, alpha, vsini)
        T = (T - self.T_scale[0]) / self.T_scale[1]
        logg = (logg - self.logg_scale[0]) / self.logg_scale[1]
        metal = (metal - self.metal_scale[0]) / self.metal_scale[1]
        if self.alpha_varies:
            alpha = (alpha - self.alpha_scale[0]) / self.alpha_scale[1]


        # Get the minimum and maximum values in the grid
        T_min = min(self.grid[:, 0])
        T_max = max(self.grid[:, 0])
        logg_min = min(self.grid[:, 1])
        logg_max = max(self.grid[:, 1])
        metal_min = min(self.grid[:, 2])
        metal_max = max(self.grid[:, 2])
        alpha_min = min(self.grid[:, 3]) if self.alpha_varies else 0.0
        alpha_max = max(self.grid[:, 3]) if self.alpha_varies else 0.0
        if self.alpha_varies:
            input_list = (T, logg, metal, alpha)
        else:
            input_list = (T, logg, metal)

        # Check to make sure the requested values fall within the grid
        if (T_min <= T <= T_max and
                        logg_min <= logg <= logg_max and
                        metal_min <= metal <= metal_max and
                (not self.alpha_varies or alpha_min <= alpha <= alpha_max)):

            y = self.interpolator(input_list)
        else:
            if self.debug:
                warnings.warn("The requested parameters fall outside the model grid. Results may be unreliable!")
            # print T, T_min, T_max
            # print logg, logg_min, logg_max
            #print metal, metal_min, metal_max
            #print alpha, alpha_min, alpha_max
            y = self.NN_interpolator(input_list)

        # Test to make sure the result is valid. If the requested point is
        # outside the Delaunay triangulation, it will return NaN's
        if np.any(np.isnan(y)):
            if self.debug:
                warnings.warn("Found NaNs in the interpolated spectrum! Falling back to Nearest Neighbor")
            y = self.NN_interpolator(input_list)

        model = DataStructures.xypoint(x=self.xaxis, y=y)
        vsini *= units.km.to(units.cm)
        model = Broaden.RotBroad(model, vsini, linear=self.rebin)


        # Return the appropriate object
        if return_xypoint:
            return model
        else:
            return model.y





class PhoenixGetter():
    def __init__(self, modeldir, rebin=True, T_min=3000, T_max=6800, metal_min=-0.5,
                 metal_max=0.5, wavemin=0, wavemax=np.inf, debug=False):
        """
        This class will read in a directory with Phoenix models

        The associated methods can be used to interpolate a model at any
        temperature, and metallicity value that
        falls within the grid

        modeldir: The directory where the models are stored. Can be a list of model directories too!
        rebin: If True, it will rebin the models to a constant x-spacing
        other args: The minimum and maximum values for the parameters to search.
                    You need to keep this as small as possible to avoid memory issues!
        """
        from kglib.utils import HelperFunctions
        self.rebin = rebin
        self.debug = debug

        # First, read in the grid
        if HelperFunctions.IsListlike(modeldir):
            # There are several directories to combine
            Tvals = []
            metalvals = []
            for i, md in enumerate(modeldir):
                if i == 0:
                    T, Z, S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max,
                                             metal_min=metal_min, metal_max=metal_max,
                                             wavemin=wavemin, wavemax=wavemax, xaxis=None)
                    spectra = np.array(S)
                else:
                    T, Z, S = self.read_grid(md, rebin=rebin, T_min=T_min, T_max=T_max,
                                             metal_min=metal_min, metal_max=metal_max,
                                             wavemin=wavemin, wavemax=wavemax, xaxis=self.xaxis)
                    S = np.array(S)
                    spectra = np.vstack((spectra, S))

                Tvals = np.hstack((Tvals, T))
                metalvals = np.hstack((metalvals, Z))
        else:
            Tvals, metalvals, spectra = self.read_grid(modeldir, rebin=rebin,
                                                       T_min=T_min, T_max=T_max,
                                                       metal_min=metal_min, metal_max=metal_max,
                                                       wavemin=wavemin, wavemax=wavemax, xaxis=None)

        # Scale the variables so they all have about the same range
        self.T_scale = ((max(Tvals) + min(Tvals)) / 2.0, max(Tvals) - min(Tvals))
        self.metal_scale = ((max(metalvals) + min(metalvals)) / 2.0, max(metalvals) - min(metalvals))
        Tvals = (np.array(Tvals) - self.T_scale[0]) / self.T_scale[1]
        metalvals = (np.array(metalvals) - self.metal_scale[0]) / self.metal_scale[1]

        # Make the grid and interpolator instances
        self.grid = np.array((Tvals, metalvals)).T
        self.spectra = np.array(spectra)
        self.interpolator = LinearNDInterpolator(self.grid, self.spectra)  # , rescale=True)
        self.NN_interpolator = NearestNDInterpolator(self.grid, self.spectra)  # , rescale=True)


    def read_grid(self, modeldir, rebin=True, T_min=3000, T_max=6800, metal_min=-0.5,
                  metal_max=0.5, wavemin=0, wavemax=np.inf, xaxis=None, debug=False):
        Tvals = []
        metalvals = []
        spectra = []
        firstkeeper = True
        modelfiles = [f for f in os.listdir(modeldir) if
                      f.startswith("lte") and "PHOENIX" in f and f.endswith(".sorted")]
        for i, fname in enumerate(modelfiles):
            T, logg, metal = ClassifyModel(fname)

            # Read in and save file if it falls in the correct parameter range
            if (T_min <= T <= T_max and
                            metal_min <= metal <= metal_max and
                        logg == 4.5):

                if self.debug:
                    print("Reading in file {:s}".format(fname))
                data = pandas.read_csv("{:s}{:s}".format(modeldir, fname),
                                       header=None,
                                       names=["wave", "flux", "continuum"],
                                       usecols=(0, 1, 2),
                                       sep=' ',
                                       skipinitialspace=True)
                x, y, c = data['wave'].values, data['flux'].values, data['continuum'].values
                n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
                x /= n
                x *= units.angstrom.to(units.nm)
                y = 10 ** y / 10 ** c

                left = np.searchsorted(x, wavemin)
                right = np.searchsorted(x, wavemax)
                x = x[left:right]
                y = y[left:right]

                if rebin:
                    if firstkeeper:
                        xgrid = np.logspace(np.log10(x[0]), np.log10(x[-1]), x.size)
                    else:
                        xgrid = self.xaxis
                    fcn = spline(x, y)
                    x = xgrid
                    y = fcn(xgrid)

                if firstkeeper:
                    self.xaxis = x if xaxis is None else xaxis
                    firstkeeper = False
                elif np.max(np.abs(self.xaxis - x) > 1e-4):
                    warnings.warn("x-axis for file {:s} is different from the master one! Not saving!".format(fname))
                    continue

                Tvals.append(T)
                metalvals.append(metal)
                spectra.append(y)

        return Tvals, metalvals, spectra


    def __call__(self, T, metal, vsini=0.0, return_xypoint=True, **kwargs):
        """
        Given parameters, return an interpolated spectrum

        If return_xypoint is False, then it will only return
          a numpy.ndarray with the spectrum

        Before interpolating, we will do some error checking to make
        sure the requested values fall within the grid
        """

        # Scale the requested values
        T = (T - self.T_scale[0]) / self.T_scale[1]
        metal = (metal - self.metal_scale[0]) / self.metal_scale[1]

        # Get the minimum and maximum values in the grid
        T_min = min(self.grid[:, 0])
        T_max = max(self.grid[:, 0])
        metal_min = min(self.grid[:, 1])
        metal_max = max(self.grid[:, 1])
        input_list = (T, metal)

        # Check to make sure the requested values fall within the grid
        if (T_min <= T <= T_max and
                        metal_min <= metal <= metal_max):

            y = self.interpolator(input_list)
        else:
            if self.debug:
                warnings.warn("The requested parameters fall outside the model grid. Results may be unreliable!")
            print(T, T_min, T_max)
            print(metal, metal_min, metal_max)
            y = self.NN_interpolator(input_list)

        # Test to make sure the result is valid. If the requested point is
        # outside the Delaunay triangulation, it will return NaN's
        if np.any(np.isnan(y)):
            if self.debug:
                warnings.warn("Found NaNs in the interpolated spectrum! Falling back to Nearest Neighbor")
            y = self.NN_interpolator(input_list)

        model = DataStructures.xypoint(x=self.xaxis, y=y)
        vsini *= units.km.to(units.cm)
        model = Broaden.RotBroad(model, vsini, linear=self.rebin)


        # Return the appropriate object
        if return_xypoint:
            return model
        else:
            return model.y
