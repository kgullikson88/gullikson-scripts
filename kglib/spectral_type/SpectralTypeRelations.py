"""
  Usage:
         Make instance of class (currently only MainSequence class available
         call instance.Interpolate(instance.dict, SpT) where dict is the name of the dictionary you want to interpolate (Temperature, Radius, or Mass) and SpT is the spectral type of what you wish to interpolate to.

# Provides relations for temperature, luminosity, radius, and mass for varius spectral types
#Data comes from Carroll and Ostlie book, or interpolated from it
#ALL RELATIONS ARE FOR MAIN SEQUENCE ONLY!
"""

from __future__ import print_function, absolute_import

from collections import defaultdict
import re
import logging
import os

from scipy.interpolate import UnivariateSpline
import numpy as np
import pandas

from kglib.utils import DataStructures


_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)

SPT_PATTERN = '[A-Z]([0-9]\.?[0-9]*)'  # regular expression pattern for identifying spectral types


def fill_dict(row, d, key, makefloat=True):
    val = row[key].strip()
    if makefloat:
        if val != '':
            d[row['SpT'].strip()[:-1]] = float(val)
    else:
        d[row['SpT'].strip()[:-1]] = val


class FitVals():
    def __init__(self, coeffs, xmean=0.0, xscale=1.0, logscale=False, intercept=0.0, valid=(-5.0, 5.0)):
        self.coeffs = coeffs
        self.order = len(coeffs) - 1.0
        self.xmean = xmean
        self.xscale = xscale
        self.log = logscale
        self.intercept = intercept
        self.valid = valid


class FunctionFits():
    def __init__(self, MS=None):
        self.MS = MainSequence() if MS is None else MS

        # Mass fits, made using the old MainSequence dictionaries
        self.sptnum_to_mass = FitVals(coeffs=np.array([0.11679476, -0.51168936, 0.27332682, 1.42616918,
                                                       -1.56182261, -1.21786221, 1.8851773, -0.04980108,
                                                       -0.30105226, -0.38423188, -0.17182606]),
                                      xmean=26.681818181818183, xscale=19.342337838478862, logscale=True,
                                      intercept=0.46702748509563452, valid=[5, 65])

        # Radius fit, made using the old MainSequence dictionaries
        self.sptnum_to_radius = FitVals(coeffs=np.array([0.02250148, 0.06041591, -0.21719815, -0.2087987,
                                                         0.55373813, 0.13635043, -0.50930703, -0.07293512,
                                                         0.3132073, -0.24671561, -0.08480404]),
                                        xmean=34.5, xscale=20.702656834329261, logscale=True,
                                        intercept=0.16198349185993394, valid=[5, 67])

        # Absolute magnitude fit, using the old MainSequence dictionaries
        self.sptnum_to_absmag = FitVals(coeffs=np.array([0.35215153, -0.2924717, -0.95804462, 1.74295661,
                                                         -0.41864979, 2.50954236, 0.45854428]),
                                        xmean=32.44, xscale=18.456608572541164,
                                        intercept=2.8008819709959134, valid=[5, 65])


        # Color fits from Boyajian et al 2013
        color_relations = defaultdict(lambda: defaultdict(FitVals))
        color_relations['B']['V'] = FitVals(coeffs=np.array((9552, -17443, 44350, 68940, 57338, -24072, 4009)),
                                            valid=[-0.1, 1.8])
        color_relations['V']['J'] = FitVals(coeffs=np.array((9052, -3972, 1039, -101))[::-1], valid=[-0.12, 4.24])
        color_relations['V']['H'] = FitVals(coeffs=np.array((8958, -3023, 632, -52.9))[::-1], valid=[-0.13, 4.77])
        color_relations['V']['K'] = FitVals(coeffs=np.array((8984, -2914, 588, -47.4))[::-1], valid=[-0.15, 5.04])
        color_relations['V']['R_j'] = FitVals(coeffs=np.array((9335, -9272, 5579, -1302.5))[::-1], valid=[0.0, 1.69])
        color_relations['V']['I_j'] = FitVals(coeffs=np.array((9189, -5372, 1884, -245.1))[::-1], valid=[-0.02, 2.77])
        color_relations['V']['R_c'] = FitVals(coeffs=np.array((9317, -13886, 12760, -4468.7))[::-1],
                                              valid=[-0.01, 1.24])
        color_relations['V']['I_c'] = FitVals(coeffs=np.array((9354, -7178, 3226, -518.2))[::-1], valid=[-0.02, 2.77])
        color_relations['V']['R_k'] = FitVals(coeffs=np.array((7371, -7940, 6947, -2557.8))[::-1], valid=[-0.21, 1.32])
        color_relations['V']['I_k'] = FitVals(coeffs=np.array((7694, -5142, 2412, -428.4))[::-1], valid=[-0.33, 2.42])
        color_relations['R_j']['J'] = FitVals(coeffs=np.array((8718, -6740, 3164, -547.0))[::-1], valid=[-0.12, 2.21])
        color_relations['R_j']['H'] = FitVals(coeffs=np.array((8689, -4292, 1356, -180.8))[::-1], valid=[-0.13, 2.80])
        color_relations['R_j']['K'] = FitVals(coeffs=np.array((8787, -4287, 1383, -187.0))[::-1], valid=[-0.15, 3.06])
        color_relations['R_c']['J'] = FitVals(coeffs=np.array((9019, -5767, 2209, -310.3))[::-1], valid=[-0.11, 3.00])
        color_relations['R_c']['H'] = FitVals(coeffs=np.array((9035, -4354, 1334, -160.9))[::-1], valid=[-0.12, 3.53])
        color_relations['R_c']['K'] = FitVals(coeffs=np.array((9077, -4054, 1133, -124.1))[::-1], valid=[-0.14, 3.80])
        color_relations['R_k']['J'] = FitVals(coeffs=np.array((10087, -7219, 2903, -433.7))[::-1], valid=[0.09, 2.58])
        color_relations['R_k']['H'] = FitVals(coeffs=np.array((9695, -4791, 1432, -175.0))[::-1], valid=[0.07, 3.17])
        color_relations['R_k']['K'] = FitVals(coeffs=np.array((9683, -4479, 1268, -147.8))[::-1], valid=[0.06, 3.43])
        color_relations['g']['z'] = FitVals(coeffs=np.array((7089, -2760, 804, -95.2))[::-1], valid=[-0.58, 3.44])
        color_relations['g']['i'] = FitVals(coeffs=np.array((7279, -3356, 1112, -153.9))[::-1], valid=[-0.23, 1.40])
        color_relations['g']['r'] = FitVals(coeffs=np.array((7526, -5570, 3750, -1332.9))[::-1], valid=[-0.23, 1.40])
        color_relations['g']['J'] = FitVals(coeffs=np.array((8576, -2710, 548, -44.0))[::-1], valid=[-0.02, 5.06])
        color_relations['g']['H'] = FitVals(coeffs=np.array((8589, -2229, 380, -27.5))[::-1], valid=[-0.12, 5.59])
        color_relations['g']['K'] = FitVals(coeffs=np.array((8526, -2084, 337, -23.3))[::-1], valid=[-0.1, 5.86])
        color_relations['V']['W3'] = FitVals(coeffs=np.array((9046, -3005, 602, -45.3))[::-1], valid=[0.76, 5.50])
        color_relations['V']['W4'] = FitVals(coeffs=np.array((9008, -2881, 565, -42.3))[::-1], valid=[0.03, 5.62])
        color_relations['R_j']['W4'] = FitVals(coeffs=np.array((9055, -4658, 1551, -199.8))[::-1], valid=[0.03, 3.56])
        color_relations['I_j']['W4'] = FitVals(coeffs=np.array((9140, -7347, 3981, -873.1))[::-1], valid=[0.04, 2.13])
        color_relations['R_c']['W4'] = FitVals(coeffs=np.array((9015, -3833, 1004, -98.5))[::-1], valid=[0.20, 4.38])
        color_relations['I_c']['W4'] = FitVals(coeffs=np.array((8971, -5296, 1997, -298.1))[::-1], valid=[0.14, 2.85])
        color_relations['R_k']['W4'] = FitVals(coeffs=np.array((9753, -4530, 1271, -137.7))[::-1], valid=[0.17, 3.93])
        color_relations['I_k']['W4'] = FitVals(coeffs=np.array((10576, -7103, 2887, -461.5))[::-1], valid=[0.23, 2.83])

        self.color_relations = color_relations
        self.interpolator = Interpolator(MS=self.MS)


    def evaluate(self, fv, independent_var, is_spt=True):
        """
        Evaluate the function defined by fv (which is a FitVals instance) for the given spectral type
        :param fv: A FitVals object that specifies the requested fit
        :param independent_var: The variable you want to evaluate the function at
        :keyword is_spt: A boolean flag for whether the independent variable is a spectral type
        """
        from kglib.utils import HelperFunctions
        if is_spt:
            if HelperFunctions.IsListlike(independent_var):
                independent_var = [re.search(SPT_PATTERN, s).group() for s in independent_var]
                x = np.array([self.MS.SpT_To_Number(s) for s in independent_var])
                if not all([fv.valid[0] < n < fv.valid[1] for n in x]):
                    logging.warn('Evaluating function outside of valid range!\n'
                                 'Value = {}\nRange = {} - {}'.format(x, fv.valid[0], fv.valid[1]))
            else:
                independent_var = re.search(SPT_PATTERN, independent_var).group()
                x = self.MS.SpT_To_Number(independent_var)
                if not fv.valid[0] < x < fv.valid[1]:
                    logging.warn('Evaluating function outside of valid range!\n'
                                 'Value = {}\nRange = {} - {}'.format(x, fv.valid[0], fv.valid[1]))
        else:
            x = independent_var

        # Normalize the sptnum
        x = (x - fv.xmean) / fv.xscale

        # Evaluate
        retval = np.poly1d(fv.coeffs)(x) + fv.intercept
        if fv.log:
            retval = 10**retval
        return retval

    def get_color(self, fv, temperature, search_range='valid'):
        """
        Get the color, given the temperature (root-finding)
        :param fv: The FitVals object to use. Should be one of the self.color_relations
        :param temperature: The temperature for which you want a color
        :param search_range: The range of colors to search. The default is the full valid range of the fit.
                             You can extend it if you want by giving a list-like object, but it will give
                             you a warning if the best fit is an extrapolation.
        :return: The color corresponding to the requested temperature
        """
        from kglib.utils import HelperFunctions
        # Determine the test values from search_range
        if isinstance(search_range, str) and search_range.lower() == 'valid':
            test_values = np.linspace(fv.valid[0], fv.valid[1], 1000)
        else:
            test_values = np.linspace(search_range[0], search_range[1], 1000)

        # Evaluate the function at each of the test colors
        test_temperatures = self.evaluate(fv, test_values, is_spt=False)

        # Determine the 'best fit' solution
        temperature = np.array(temperature)
        differences = (temperature.reshape(1, -1) - test_temperatures.reshape(-1, 1))
        idx = np.abs(differences).argmin(axis=0)
        color = test_values[idx]

        # Check if the best-fit solution is an extrapolation
        if HelperFunctions.IsListlike(search_range):
            if HelperFunctions.IsListlike(color):
                if not all([fv.valid[0] < c < fv.valid[1] for c in color]):
                    logging.warn('Best-fit color is an extrapolation from the valid range. Be very careful!')
            elif not fv.valid[0] < color < fv.valid[1]:
                logging.warn('Best-fit color is an extrapolation from the valid range. Be very careful!')
        return color

    def __call__(self, fv, spt):
        return self.evaluate(fv, spt)


class Interpolator():
    def __init__(self, MS=None):
        self.MS = MainSequence() if MS is None else MS

        # Spectral type to temperature converter
        #fname = '{}/Dropbox/School/Research/Databases/SpT_Relations/sptnum_to_teff.interp'.format(os.environ['HOME'])
        #fileid = open(fname, 'r')
        #self.sptnum_to_teff = pickle.load(fileid)
        #fileid.close()
        self.make_new_interpolator()

    def make_new_interpolator(self, filename=get_data('Pecaut2013.tsv')):
        df = pandas.read_csv(filename, skiprows=55, sep='|', engine='python')[2:-1]
        sptnum = [self.MS.SpT_To_Number(s.strip()[:-1]) for s in df.SpT.values]
        self.sptnum_to_teff = UnivariateSpline(sptnum, df.Teff.values, s=0)

    def evaluate(self, interp, spt):
        from kglib.utils import HelperFunctions
        if HelperFunctions.IsListlike(spt):
            spt = [re.search(SPT_PATTERN, s).group() for s in spt]
            sptnum = np.array([self.MS.SpT_To_Number(s) for s in spt])
        else:
            spt = re.search(SPT_PATTERN, spt).group()
            sptnum = self.MS.SpT_To_Number(spt)

        return interp(sptnum)

    def __call__(self, interp, spt):
        return self.evaluate(interp, spt)



class MainSequence:
    def __init__(self):
        self.FunctionFitter = FunctionFits(self)
        self.Interpolator = self.FunctionFitter.interpolator

        # TODO: Remove all these dictionaries!
        self.Temperature = defaultdict(float)
        self.Radius = defaultdict(float)
        self.Mass = defaultdict(float)
        self.Lifetime = defaultdict(float)
        self.BC = defaultdict(float)
        self.BmV = defaultdict(float)  #B-V color
        self.UmB = defaultdict(float)  # U-B color
        self.VmR = defaultdict(float)  #V-R color
        self.VmI = defaultdict(float)  #V-I color
        self.VmJ = defaultdict(float)  #V-J color
        self.VmH = defaultdict(float)  #V-H color
        self.VmK = defaultdict(float)  #V-K color
        self.AbsMag = defaultdict(float)  #Absolute Magnitude in V band

        # Read in the data from Pecaut & Mamajek 2013 for Teff and color indices
        pfilename = get_data('Pecaut2013.tsv')
        # pdata = pandas.read_csv(pfilename, skiprows=55, sep="|")[2:-1]
        pdata = pandas.read_csv(pfilename, sep="|", skip_blank_lines=True, comment='#')[2:]
        pdata.apply(fill_dict, axis=1, args=(self.Temperature, 'Teff', True))
        pdata.apply(fill_dict, axis=1, args=(self.UmB, 'U-B', True))
        pdata.apply(fill_dict, axis=1, args=(self.BmV, 'B-V', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmR, 'V-Rc', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmI, 'V-Ic', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmJ, 'V-J', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmH, 'V-H', True))
        pdata.apply(fill_dict, axis=1, args=(self.VmK, 'V-Ks', True))

        self.Radius['O5'] = 13.4
        self.Radius['O6'] = 12.2
        self.Radius['O7'] = 11.0
        self.Radius['O8'] = 10.0
        self.Radius['B0'] = 6.7
        self.Radius['B1'] = 5.4  #Malkov et al. 2007
        self.Radius['B2'] = 4.9  #Malkov et al. 2007
        self.Radius['B3'] = 3.9  #Malkov et al. 2007
        self.Radius['B4'] = 3.6  #Malkov et al. 2007
        self.Radius['B5'] = 3.3  #Malkov et al. 2007
        self.Radius['B6'] = 3.1  #Malkov et al. 2007
        self.Radius['B7'] = 2.85  #Malkov et al. 2007
        self.Radius['B8'] = 2.57  #Malkov et al. 2007
        self.Radius['B9'] = 2.3
        self.Radius['A0'] = 2.2
        self.Radius['A1'] = 2.1
        self.Radius['A2'] = 2.0
        self.Radius['A5'] = 1.8
        self.Radius['A8'] = 1.5
        self.Radius['F0'] = 1.4
        self.Radius['F2'] = 1.3
        self.Radius['F5'] = 1.2
        self.Radius['F8'] = 1.1
        self.Radius['G0'] = 1.06
        self.Radius['G2'] = 1.03
        self.Radius['G8'] = 0.96
        self.Radius['K0'] = 0.93
        self.Radius['K1'] = 0.92
        self.Radius['K3'] = 0.86
        self.Radius['K4'] = 0.83
        self.Radius['K5'] = 0.80
        self.Radius['K7'] = 0.74
        self.Radius['M0'] = 0.63
        self.Radius['M1'] = 0.56
        self.Radius['M2'] = 0.48
        self.Radius['M3'] = 0.41
        self.Radius['M4'] = 0.35
        self.Radius['M5'] = 0.29
        self.Radius['M6'] = 0.24
        self.Radius['M7'] = 0.20

        self.Mass['O5'] = 60
        self.Mass['O6'] = 37
        self.Mass['O8'] = 23
        self.Mass['B0'] = 17.5
        self.Mass['B1'] = 10.5  #Malkov et al. 2007
        self.Mass['B2'] = 8.9  #Malkov et al. 2007
        self.Mass['B3'] = 6.4  #Malkov et al. 2007
        self.Mass['B4'] = 5.4  #Malkov et al. 2007
        self.Mass['B5'] = 4.5  #Malkov et al. 2007
        self.Mass['B6'] = 4.0  #Malkov et al. 2007
        self.Mass['B7'] = 3.5  #Malkov et al. 2007
        self.Mass['B8'] = 3.2  #Malkov et al. 2007
        self.Mass['A0'] = 2.9
        self.Mass['A5'] = 2.0
        self.Mass['F0'] = 1.6
        self.Mass['F5'] = 1.4
        self.Mass['G0'] = 1.05
        self.Mass['K0'] = 0.79
        self.Mass['K5'] = 0.67
        self.Mass['M0'] = 0.51
        self.Mass['M2'] = 0.40
        self.Mass['M5'] = 0.21

        self.Lifetime['O9.1'] = 8
        self.Lifetime['B1.1'] = 11
        self.Lifetime['B2.5'] = 16
        self.Lifetime['B4.2'] = 26
        self.Lifetime['B5.3'] = 43
        self.Lifetime['B6.7'] = 94
        self.Lifetime['B7.7'] = 165
        self.Lifetime['B9.7'] = 350
        self.Lifetime['A1.6'] = 580
        self.Lifetime['A5'] = 1100
        self.Lifetime['A8.4'] = 1800
        self.Lifetime['F2.6'] = 2700

        #From Allen's Astrophysical Quantities and Binney & Merrifield (marked with 'BM')
        self.AbsMag['O5'] = -5.7
        self.AbsMag['O8'] = -4.9  #BM
        self.AbsMag['O9'] = -4.5
        self.AbsMag['B0'] = -4.0
        self.AbsMag['B2'] = -2.45
        self.AbsMag['B3'] = -1.6  #BM
        self.AbsMag['B5'] = -1.2
        self.AbsMag['B8'] = -0.25
        self.AbsMag['A0'] = 0.65
        self.AbsMag['A2'] = 1.3
        self.AbsMag['A5'] = 1.95
        self.AbsMag['F0'] = 2.7
        self.AbsMag['F2'] = 3.6
        self.AbsMag['F5'] = 3.5
        self.AbsMag['F8'] = 4.0
        self.AbsMag['G0'] = 4.4
        self.AbsMag['G2'] = 4.7
        self.AbsMag['G5'] = 5.1
        self.AbsMag['G8'] = 5.5
        self.AbsMag['K0'] = 5.9
        self.AbsMag['K2'] = 6.4
        self.AbsMag['K5'] = 7.35
        self.AbsMag['M0'] = 8.8
        self.AbsMag['M2'] = 9.9
        self.AbsMag['M5'] = 12.3

    def SpT_To_Number(self, SpT):
        SpT_match = re.search(SPT_PATTERN, SpT)
        if SpT_match is None or SpT_match.group()[1:] == '':
            basenum = 5.0
        else:
            SpT = SpT_match.group()
            basenum = float(SpT[1:])
        SpectralClass = SpT[0]
        if SpectralClass == "O":
            return basenum
        elif SpectralClass == "B":
            return basenum + 10
        elif SpectralClass == "A":
            return basenum + 20
        elif SpectralClass == "F":
            return basenum + 30
        elif SpectralClass == "G":
            return basenum + 40
        elif SpectralClass == "K":
            return basenum + 50
        elif SpectralClass == "M":
            return basenum + 60
        elif SpectralClass == "L":
            return basenum + 70
        elif SpectralClass == "T":
            return basenum + 80
        elif SpectralClass == "Y":
            return basenum + 90
        else:
            print( "Something weird happened! Spectral type = ", SpT)
            return -1

    def Number_To_SpT(self, number):
        tens_index = 0
        num = float(number)
        while num >= 0:
            num -= 10
            tens_index += 1
        tens_index = tens_index - 1
        if abs(num) < 1e-5:
            tens_index += 1
            number = 10 * tens_index
        if tens_index == 0:
            spt_class = "O"
        elif tens_index == 1:
            spt_class = "B"
        elif tens_index == 2:
            spt_class = "A"
        elif tens_index == 3:
            spt_class = "F"
        elif tens_index == 4:
            spt_class = "G"
        elif tens_index == 5:
            spt_class = "K"
        elif tens_index == 6:
            spt_class = "M"
        subclass = str(number - 10 * tens_index)
        return spt_class + subclass

    def Interpolate(self, parameter, SpT):
        """
        A new version that uses pre-made interpolations and function fits
        :param parameter: The string name of the value you want. Valid options are (case-insensitive):
             + Mass
             + Radius
             + Temperature
             + Absmag  (gives the absolute V magnitude. Use GetAbsoluteMagnitude to get other colors!)

             Note: You can still give a dictionary like before, but that is discouraged and will spit out a warning
        :param SpT: The spectral type to interpolate at. If you give parameters as strings, these can now be list-like!
        :return: The value of the requested parameter, at the requested spectral type(s)
        """
        if isinstance(parameter, dict):
            logging.warn('Dictionary input is deprecated! Use string names instead!')
            return self.Interpolate_Old(parameter, SpT)

        # If we get here, we are using the new method!
        if parameter.lower().strip() == 'temperature':
            return self.Interpolator(self.Interpolator.sptnum_to_teff, SpT)
        elif parameter.lower().strip() == 'mass':
            return self.FunctionFitter(self.FunctionFitter.sptnum_to_mass, SpT)
        elif parameter.lower().strip() == 'radius':
            return self.FunctionFitter(self.FunctionFitter.sptnum_to_radius, SpT)
        elif parameter.lower().strip() == 'absmag':
            return self.FunctionFitter(self.FunctionFitter.sptnum_to_absmag, SpT)

    def Interpolate_Old(self, dictionary, SpT):
        #First, we must convert the relations above into a monotonically increasing system
        #Just add ten when we get to each new spectral type
        relation = DataStructures.xypoint(len(dictionary))

        # Strip the spectral type of the luminosity class information
        SpT = re.search('[A-Z]([0-9]\.?[0-9]*)', SpT).group()

        xpoints = []
        ypoints = []
        for key, index in zip(dictionary, range(len(dictionary))):
            #Convert key to a number
            xpoints.append(self.SpT_To_Number(key))
            ypoints.append(dictionary[key])

        sorting_indices = [i[0] for i in sorted(enumerate(xpoints), key=lambda x: x[1])]
        for index in range(len(dictionary)):
            i = sorting_indices[index]
            relation.x[index] = xpoints[i]
            relation.y[index] = ypoints[i]

        RELATION = UnivariateSpline(relation.x, relation.y, s=0)

        spnum = self.SpT_To_Number(SpT)
        if spnum > 0:
            return RELATION(spnum)
        else:
            return np.nan

    def GetAbsoluteMagnitude(self, spt, color='V'):
        """
        Return the absolute magnitude of the requested spectral type, in the requested band.
        :param spt: The spectral type you want
        :param color: The band you want. Valid options are:
           + V, B, J, H, K (Johnson bands)
           + R_c, I_c (Cousins bands)
           + R_k, I_k (??)
           + R_j, I_j (Johnson bands)
           + g,r,i,z (SDSS bands)
           + W3, W4 (WISE bands)
        :return: The absolute magnitude of the given spectral type. IGNORES the luminosity class!
        """
        Vmag = self.Interpolate('Absmag', spt)
        if color.upper() == "V":
            return Vmag
        else:
            valid_colors = ['B', 'J', 'H', 'K', 'R_C', 'I_C', 'R_K', 'I_K',
                            'R_J', 'I_J', 'G', 'R', 'I', 'Z', 'W3', 'W4']
            if color.upper() not in valid_colors:
                print('Valid colors: ')
                print(valid_colors)
                raise ValueError('Must give a color in the list above!')
            temperature = self.Interpolate('temperature', spt)
            if color.upper() == 'B':
                color_diff = self.FunctionFitter.get_color(self.FunctionFitter.color_relations['B']['V'],
                                                           temperature, search_range=[-3.0, 8.0])
                return color_diff + Vmag
            elif color.upper() in ['J', 'H', 'K', 'R_C', 'I_C', 'R_K', 'I_K', 'R_J', 'I_J', 'W3', 'W4']:
                color_diff = self.FunctionFitter.get_color(self.FunctionFitter.color_relations['V'][color],
                                                           temperature, search_range=[-3.0, 8.0])
                return Vmag - color_diff
            else:
                # The color is one of the SDSS bands We need g, which we can get from J, H, and K. Get all and take median!
                Jmag = self.GetAbsoluteMagnitude(spt, color='J')
                Hmag = self.GetAbsoluteMagnitude(spt, color='H')
                Kmag = self.GetAbsoluteMagnitude(spt, color='K')
                gmag = []
                gmag.append(Jmag + self.FunctionFitter.get_color(self.FunctionFitter.color_relations['g']['J'],
                                                                 temperature, search_range=[-3.0, 8.0]))
                gmag.append(Hmag + self.FunctionFitter.get_color(self.FunctionFitter.color_relations['g']['H'],
                                                                 temperature, search_range=[-3.0, 8.0]))
                gmag.append(Kmag + self.FunctionFitter.get_color(self.FunctionFitter.color_relations['g']['K'],
                                                                 temperature, search_range=[-3.0, 8.0]))
                gmag = np.median(gmag)
                if color.upper() == 'G':
                    return gmag
                else:
                    color_diff = self.FunctionFitter.get_color(self.FunctionFitter.color_relations['g'][color],
                                                               temperature, search_range=[-3.0, 8.0])
                    return gmag - color_diff


    def GetSpectralType_FromAbsMag(self, value, color='V', prec=1.0):
        """
        Given an absolute magnitude in some band, return the spectral type that best matches it
        :param value: The absolute magnitude
        :param color: The band the magnitude is measured in
        :param prec: The precision you want in the returned spectral type.
                     prec=1.0 means spectral type subclass (returns things like 'G4').
                     prec=0.1 would mean returning things like 'G4.3'
        :return: The spectral type that best matches the given absolute magnitude
        """
        spt_num = np.arange(10, 70, prec)
        spt_values = np.array([self.Number_To_SpT(n) for n in spt_num])
        absmag = self.GetAbsoluteMagnitude(spt_values, color=color)

        dm = (np.array(value).reshape(1, -1) - absmag.reshape(-1, 1))
        idx = np.abs(dm).argmin(axis=0)
        spt = spt_values[idx]
        return spt


    def GetSpectralType(self, parameter, value, prec=1.0):
        """
        Returns the spectral type that is closest to the requested value of the requested parameter
        :param parameter: A string containing any of the valid parameters (see Interpolate docstring).
                          A dictionary can still be given, but we will now throw a warning
        :param value: The value of the parameter
        :param prec: The precision you want in the returned spectral type.
                     prec=1.0 means spectral type subclass (returns things like 'G4').
                     prec=0.1 would mean returning things like 'G4.3'
        :return: The spectral type that best matches the given value of the requested parameter
        """

        if isinstance(parameter, dict):
            logging.warn('Dictionary input is deprecated! Use string names instead!')
            interpolate = True if prec < 1 else False
            return self.GetSpectralType_Old(parameter, value, interpolate=interpolate)

        # If we get here, we can vectorize things
        spt_num = np.arange(10, 70, prec)
        spt_values = np.array([self.Number_To_SpT(n) for n in spt_num])
        test_values = self.Interpolate(parameter, spt_values)

        difference = (np.array(value).reshape(1, -1) - test_values.reshape(-1, 1))
        idx = np.abs(difference).argmin(axis=0)
        spt = spt_values[idx]
        return spt


    def GetSpectralType_Old(self, dictionary, value, interpolate=False):
        """
        Returns the spectral type that is closest to the requested value of the requested parameter. Deprecated!
        :param parameter: One of the MS class dictionaries.
        :param value: The value of the parameter
        :param interpolate: If True, it will return a spectral type at ridiculously high precision
        :return: The spectral type that best matches the given value of the requested parameter
        """
        testgrid = np.arange(self.SpT_To_Number("O1"), self.SpT_To_Number("M9"), 0.1)
        besttype = "O1"
        best_difference = 9e9
        for num in testgrid:
            num = round(num, 2)
            spt = self.Number_To_SpT(num)
            difference = np.abs(value - self.Interpolate(dictionary, spt))
            if difference < best_difference:
                best_difference = difference
                besttype = spt
        if not interpolate:
            return besttype
        else:
            bestvalue = self.Interpolate(dictionary, besttype)
            num = self.SpT_To_Number(besttype)
            spt = self.Number_To_SpT(num - 0.1)
            secondvalue = self.Interpolate(dictionary, spt)
            slope = 0.1 / (bestvalue - secondvalue)
            num2 = slope * (bestvalue - value) + num
            return self.Number_To_SpT(num2)
