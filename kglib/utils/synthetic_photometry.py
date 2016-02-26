import pysynphot
from astropy import units as u, constants
from kglib.spectral_type import SpectralTypeRelations, Mamajek_Table
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import logging
from scipy.optimize import fmin, brute
import numpy as np



class CompanionFitter(object):
    def __init__(self, filt, T_low=3500, T_high=12000, dT=10, feh=0.0):
        """
        Initialize a CompanionFitter instance. It will pre-tabulate
        synthetic photometry for main-sequence stars with Temperatures
        ranging from T_low to T_high, in steps on dT K. All the 
        models with have [Fe/H] = feh. Finally, we will interpolate
        the relationship between temperature and magnitude so that
        additional photometry points are made quickly.

        :param filt: A pysynphot bandpass encoding the filter information.
        :param T_low, T_high, dT: Parameters describing the temperature grid 
                                  to interpolate  
        :param feh: The metallicity [Fe/H] to use for the models
        """
        # Use the Mamajek table to get main-sequence relationships
        MT = Mamajek_Table.MamajekTable()
        MT.mam_df['radius'] = 10**(0.5*MT.mam_df.logL - 2.0*MT.mam_df.logT + 2.0*3.762)
        MT.mam_df['logg'] = 4.44 + np.log10(MT.mam_df.Msun) - 2.0*np.log10(MT.mam_df.radius)
        teff2radius = MT.get_interpolator('Teff', 'radius')
        teff2mass = MT.get_interpolator('Teff', 'Msun')
        teff2logg = MT.get_interpolator('Teff', 'logg')

        # Pre-calculate the magnitude at each temperature
        self.temperature = np.arange(T_low, T_high, dT)
        self.magnitude = np.zeros(self.temperature.size)
        for i, T in enumerate(self.temperature):
            logging.info('i = {}/{}: T = {:.1f}'.format(i+1, self.temperature.size, T))
            logg = teff2logg(T)
            R = teff2radius(T)
            spec = pysynphot.Icat('ck04models', T, feh, logg) * R**2
            obs = pysynphot.Observation(spec, filt)
            self.magnitude[i] = obs.effstim('abmag')

        # Interpolate the T-mag curve
        self.interpolator = spline(self.temperature, self.magnitude)


    def fit(self, T_prim, delta_mag, delta_mag_error, T_range=(3500, 9000)):
        """
        Fit for the companion temperature given a primary temperature and delta-magnitude measurement
        """

        def lnlike(T2, T1, dm, dm_err):
            dm_synth = self.__call__(T2) - self.__call__(T1)
            logging.debug('T2 = {}: dm = {}'.format(T2, dm_synth))
            return 0.5 * (dm - dm_synth)**2 / dm_err**2

        #T_sec = fmin(lnlike, guess_T, args=(T_prim, delta_mag, delta_mag_error))
        T_sec = brute(lnlike, [T_range], args=(T_prim, delta_mag, delta_mag_error))
        return T_sec



    def __call__(self, T):
        """
         Evaluate the spline at the given temperature, returning the interpolated magnitude
        """
        return self.interpolator(T)

    @classmethod
    def make_box_filter(cls, center, width):
        """
        Make a box filter with the given parameters. Both center and width should either be in angstroms, 
        or be astropy quantities.
        """
        if not isinstance(center, u.quantity.Quantity):
            center *= u.angstrom
        if not isinstance(width, u.quantity.Quantity):
            width *= u.angstrom

        return pysynphot.Box(center.to(u.angstrom).value, width.to(u.angstrom).value)




