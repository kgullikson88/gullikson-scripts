"""
This is a module to read in an HDF5 file with CCFs.
Use this to determine the best parameters, and plot the best CCF for each star/date
"""
from __future__ import print_function, division, absolute_import

from collections import defaultdict
import logging

import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

import pandas as pd


class CCF_Interface(object):
    """
    Provides an interface to the HDF5 cross-correlation function file

    Parameters:
    ===========
    - filename:         string
                        The path to the HDF5 cross-correlation function file.

    - vel:              numpy.ndarray
                        What velocities should we interpolate the CCFs onto?
                        You should provide every velocity point you want sampled.
    """
    def __init__(self, filename, vel=np.arange(-900, 900, 1)):
        self.hdf5 = h5py.File(filename, 'r')
        self.velocities = vel
        self._df = None


    def __getitem__(self, path):
        return self.hdf5[path]


    def list_stars(self, print2screen=False):
        """
        List the stars available in the HDF5 file, and the dates available for each

        Parameters:
        ===========
        - print2screen:     bool
                            Should we print the stars and dates to screen?

        Returns:
        =========
        - star_list:        list
                            A list of every star in the file, sorted by name.
        """
        if print2screen:
            for star in sorted(self.hdf5.keys()):
                print(star)
                for date in sorted(self.hdf5[star].keys()):
                    print('\t{}'.format(date))
        return sorted(self.hdf5.keys())


    def list_dates(self, star, print2screen=False):
        """
        List the dates available for the given star

        Parameters:
        ===========
        - star:             string
                            The name of the star to search. See self.list_stars() for the valid names.

        - print2screen:     bool
                            Should we print the stars and dates to screen?

        Returns:
        =========
        - date_list:        list
                            A sorted list of every date the given star was observed.
        """
        if star not in self.hdf5:
            logging.warn('Star ({}) not found in HDF5 file!'.format(star))
            return

        if print2screen:
            for date in sorted(self.hdf5[star].keys()):
                print(date)
        return sorted(self.hdf5[star].keys())

    def load_cache(self, addmode='simple'):
        """
        Read in the whole HDF5 file. This will take a while and take a few Gb of memory,
        but will speed things up considerably

        Parameters:
        ===========
        - addmode:    string
                      The way the individual CCFs were added. Options are:
                          - 'simple'
                          - 'ml'
                          - 'dc'
                          - 'all'  (saves all addmodes)
        """
        self._df = self._compile_data(addmode=addmode)


    def _compile_data(self, starname=None, date=None, addmode='simple', read_ccf=True):
        """
        This reads in all the datasets for the given star and date

        Parameters:
        ===========
        - starname:     string
                        The name of the star to search. See self.list_stars() for the valid names.

        - date:         string
                        The date to search. Must be in self.hdf5[star]

        - addmode:      string
                        The way the individual CCFs were added. Options are:
                            - 'simple'
                            - 'ml'
                            - 'dc'
                            - 'all'  (saves all addmodes)

        Returns:
        =========
        - data:         pandas DataFrame
                        A dataframe with the following columns:
                            - star
                            - date
                            - temperature
                            - log(g)
                            - [Fe/H]
                            - vsini
                            - addmode
                            - rv (at maximum CCF value)
                            - CCF height (maximum)
        """
        if starname is None:
            df_list = []
            star_list = self.list_stars()
            for star in star_list:
                date_list = self.list_dates(star)
                for date in date_list:
   
                    logging.debug('Reading in metadata for star {}, date {}'.format(star, date))
                    df_list.append(self._compile_data(star, date, addmode=addmode, read_ccf=read_ccf))
            return pd.concat(df_list, ignore_index=True)
            
        elif starname is not None and date is None:
            df_list = []
            date_list = self.list_dates(starname)
            for date in date_list:
                logging.debug('Reading in metadata for date {}'.format(date))
                df_list.append(self._compile_data(starname, date, addmode=addmode, read_ccf=read_ccf))
            return pd.concat(df_list, ignore_index=True)
            
        else:
            if self._df is not None:
                return self._df.loc[(self._df['Star'] == starname) & (self._df['Date'] == date)].copy()
            #print('Stars: ', self.list_stars())
            datasets = self.hdf5[starname][date].keys()
            data = defaultdict(list)
            for ds_name, ds in self.hdf5[starname][date].iteritems():  # in datasets:
                #ds = self.hdf5[starname][date][ds_name]
                try:
                    am = ds.attrs['addmode']
                    if addmode == 'all' or addmode == am:
                        data['T'].append(ds.attrs['T'])
                        data['logg'].append(ds.attrs['logg'])
                        data['[Fe/H]'].append(ds.attrs['[Fe/H]'])
                        data['vsini'].append(ds.attrs['vsini'])
                        data['addmode'].append(am)
                        data['name'].append(ds.name)
                        try:
                            data['ccf_max'].append(ds.attrs['ccf_max'])
                            data['vel_max'].append(ds.attrs['vel_max'])
                        except KeyError:
                            vel, corr = ds.value
                            idx = np.argmax(corr)
                            data['ccf_max'].append(corr[idx])
                            data['vel_max'].append(vel[idx])

                        if read_ccf:
                            v = ds.value
                            vel, corr = v[0], v[1]
                            sorter = np.argsort(vel)
                            fcn = spline(vel[sorter], corr[sorter])
                            data['ccf'].append(fcn(self.velocities))
                except:
                    raise IOError('Something weird happened with dataset {}!'.format(ds.name))

            data['Star'] = [starname] * len(data['T'])
            data['Date'] = [date] * len(data['T'])
            df = pd.DataFrame(data=data)
            return df

    def get_temperature_run(self, starname=None, date=None, df=None):
        """
        Return the maximum ccf height for each temperature. Either starname AND date, or df must be given

        Parameters:
        ===========
        - starname:      string
                         The name of the star

        - param date:    string
                         The date of the observation

        - param df:      pandas DataFrame
                         Input dataframe, such as from _compile_data. Overrides starname and date, if given

        Returns:
        ========
        - temp_run:      pandas DataFrame
                         Contains all the best parameters for each temperature
        """
        # Get the dataframe if it isn't given
        if df is None:
            if starname is None or date is None:
                raise ValueError('Must give either starname or date to get_temperature_run!')
            df = self._compile_data(starname, date)

        # Find the maximum CCF for each set of parameters
        fcn = lambda row: (np.max(row), self.velocities[np.argmax(row)])
        vals = df['ccf'].map(fcn)
        df['ccf_max'] = vals.map(lambda l: l[0])
        df['rv'] = vals.map(lambda l: l[1])

        # Find the best parameters for each temperature
        d = defaultdict(list)
        temperatures = pd.unique(df['T'])
        for T in temperatures:
            good = df.loc[df['T'] == T]
            best = good.loc[good.ccf_max == good.ccf_max.max()]
            d['vsini'].append(best['vsini'].item())
            d['logg'].append(best['logg'].item())
            d['[Fe/H]'].append(best['[Fe/H]'].item())
            d['rv'].append(best['rv'].item())
            d['ccf_value'].append(best.ccf_max.item())
            d['T'].append(T)
            d['metal'].append(best['[Fe/H]'].item())

        return pd.DataFrame(data=d)

    def get_ccf(self, params, df=None):
        """
        Get the ccf with the given parameters.

        Parameters:
        ===========
        - params:    dictionary:
                     All the parameters necessary to define a single ccf. This should be
                     a python dictionary with the keys:
                         - 'starname': The name of the star. Try self.list_stars() for the options.
                         - 'date': The UT date of the observations. Try self.list_dates() for the options.
                         - 'T': temperature of the model
                         - 'logg': the log(g) of the model
                         - 'vsini': the vsini by which the model was broadened before correlation
                         - '[Fe/H]': the metallicity of the model
                         - 'addmode': The way the order CCFs were added to make a total one. Can be:
                             - 'simple'
                             - 'ml'
                             - 'weighted'
                             - 'dc'


        - df:        a pandas DataFrame such as outputted by _compile_data

        Returns:
        ========
        -ccf:        pandas DataFrame
                     Holds columns of velocity and CCF power
        """
        if df is None:
            try:
                df = self._compile_data(params['starname'], params['date'])
            except KeyError:
                raise KeyError('Must give get_ccf params with starname and date keywords, if df is not given!')

        Tvals = df['T'].unique()
        T = Tvals[np.argmin(abs(Tvals - params['T']))]
        good = df.loc[(df['T'] == T) & (df.logg == params['logg']) & (df.vsini == params['vsini']) \
                      & (df['[Fe/H]'] == params['[Fe/H]']) & (df.addmode == params['addmode'])]

        return pd.DataFrame(data={'velocity': self.velocities, 'CCF': good['ccf'].item()})



