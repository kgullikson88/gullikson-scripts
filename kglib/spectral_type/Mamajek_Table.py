from __future__ import print_function, division, absolute_import

import os

from scipy.interpolate import InterpolatedUnivariateSpline as spline
import pandas as pd

from kglib.spectral_type import SpectralTypeRelations

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)

home = os.environ['HOME']
TABLE_FILENAME = os.path.join(get_data('Mamajek_Table.txt'))

class MamajekTable(object):
    """
    Class to interact with the table that Eric mamajek has online at
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    """
    def __init__(self, filename=TABLE_FILENAME):
        MS = SpectralTypeRelations.MainSequence()

        # Read in the table.
        colspecs=[[0,7], [7,14], [14,21], [21,28], [28,34], [34,40], [40,47], [47,55],
                  [55,63], [63,70], [70,78], [78,86], [86,94], [94,103], [103,110],
                  [110,116], [116,122], [122,130], [130,137], [137,144], [144,151],
                  [151,158]]
        mam_df = pd.read_fwf(filename, header=20, colspecs=colspecs, na_values=['...'])[:92]

        # Strip the * from the logAge column. Probably shouldn't but...
        mam_df['logAge'] = mam_df['logAge'].map(lambda s: s.strip('*') if isinstance(s, basestring) else s)

        # Convert everything to floats
        for col in mam_df.columns:
            mam_df[col] = pd.to_numeric(mam_df[col], errors='ignore')

        # Add the spectral type number for interpolation
        mam_df['SpTNum'] = mam_df['SpT'].map(MS.SpT_To_Number)
        
        self.mam_df = mam_df

    def get_columns(self, print_keys=True):
        """
        Get the column names in a list, and optionally print them to the screen.

        Parameters:
        ===========
        - print_keys:    boolean
                         Print the keys to the screen?

        Returns:
        ========
        A sorted list of column names.
        """
        if print_keys:
            for k in self.mam_df.keys():
                print(k)
        return list(self.mam_df.keys())

    def get_interpolator(self, xcolumn, ycolumn, extrap='nearest'):
        """
        Get an interpolator instance between the two columns

        Parameters:
        ===========
        - xcolumn:    string
                      The name of the x column to interpolate between

        - ycolumn:    string
                      The name of the value you want to interpolate

        - extrap:     string
                      How to treat extrapolation. Options are:
                          1. 'nearest': Default behavior. It will return the nearest match
                             to the given 'x' value
                          2. 'extrapolate': Extrapolate the spline. This is probably only
                             safe for very small extrapolations

        Returns:
        ========
        A callable interpolator.
        """
        # Make sure the column names are correct
        assert xcolumn in self.mam_df.keys() and ycolumn in self.mam_df.keys()

        # Sort the dataframe by the x column, and drop any duplicates or nans it might have
        sorted_df = self.mam_df.sort_values(by=xcolumn).dropna(subset=[xcolumn, ycolumn], how='any').drop_duplicates(xcolumn)

        # Make an interpolator
        ext_value = {'nearest': 3, 'extrapolate': 0}
        fcn = spline(sorted_df[xcolumn].values, sorted_df[ycolumn].values, ext=ext_value[extrap])
        return fcn

