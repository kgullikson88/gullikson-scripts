from __future__ import print_function

import os
import logging

from astropy.io import fits
import numpy as np
from astroquery.simbad import Simbad
import pandas as pd

from kglib import stellar_data
from .HelperFunctions import convert_to_hex


Simbad.TIMEOUT = 60
try:
    Simbad.add_votable_fields('sp', 'flux(V)', 'flux(K)', 'plx')
except KeyError:
    pass

data_cache = {}

class stardata:
    def __init__(self):
        self.main_id = ''
        self.spectype = ""
        self.Vmag = 0.0
        self.Kmag = 0.0
        self.ra = ""
        self.dec = ""
        self.par = 0.0  # parallax in milli-arcseconds


def GetData(starname, safe_spt=False):
    """
    Search simbad for information about the given star.

    Parameters:
    ===========
    - starname:   string
                  A simbad-queryable name for the star

    - safe_spt:   boolean
                  If True, convert spectral types with 'm' in them to '5': eg. 'Am' --> 'A5'
    """
    logging.info('Getting stellar data for {}'.format(starname))
    if starname in data_cache:
        return data_cache[starname]

    data = stardata()
    # Try the pre-downloaded database first
    dr = stellar_data.DatabaseReader()
    star = dr.query_object(starname)
    dr.db_con.close()
    if len(star) > 0:
        star = star.ix[0]
        data.main_id = star.main_id
        data.spectype = star.spectral_type
        if safe_spt:
            data.spectype = data.spectype.replace('m', '5')
        data.Vmag = star.Vmag
        data.Kmag = star.Kmag
        data.ra = convert_to_hex(star.RA, delimiter=':')
        data.dec = convert_to_hex(star.DEC, delimiter=':', force_sign=True)
        data.par = star.parallax
        data_cache[starname] = data

        return data

    # If we get here, the star was not found in the stellar_data database
    # Fall back on astroquery.
    try:
        Simbad.SIMBAD_URL = 'http://simbak.cfa.harvard.edu/simbad/sim-script'
        star = Simbad.query_object(starname)
    except:
        Simbad.SIMBAD_URL = 'http://simbad.u-strasbg.fr/simbad/sim-script'
        star = Simbad.query_object(starname)
    if star is None:
        logging.warn('Simbad query for object "{}" failed!'.format(starname))
        data.main_id = starname
        data.spectype = 'Unknown'
        data.Vmag = np.nan
        data.Kmag = np.nan
        data.ra = 'Unknown'
        data.dec = 'Unknown'
        data.par = np.nan
        return data

    data.main_id = star['MAIN_ID'].item()
    data.spectype = star['SP_TYPE'].item()
    if safe_spt:
        data.spectype = data.spectype.replace('m', '5')
    data.Vmag = star['FLUX_V'].item()
    data.Kmag = star['FLUX_K'].item()
    data.ra = star['RA'].item().strip().replace(' ', ':')
    data.dec = star['DEC'].item().strip().replace(' ', ':')
    data.par = star['PLX_VALUE'].item()
    data_cache[starname] = data
    return data


homedir = os.environ['HOME']
VSINI_FILE = "{}/School/Research/Useful_Datafiles/Vsini.csv".format(homedir)


def get_vsini(file_list, vsini_filename=VSINI_FILE):
    """
    Get the vsini for every fits file in file_list. Uses the OBJECT keyword and a pre-tabulated vsini table.
    This is really only useful for my project...
    :param file_list:
    :return:
    """
    vsini = pd.read_csv(vsini_filename, sep='|', skiprows=8)[1:]
    vsini_dict = {}
    prim_vsini = []
    for fname in file_list:
        root = fname.split('/')[-1][:9]
        if root in vsini_dict:
            prim_vsini.append(vsini_dict[root])
        else:
            header = fits.getheader(fname)
            star = header['OBJECT']
            try:
                v = vsini.loc[vsini.Identifier.str.strip() == star]['vsini(km/s)'].values[0]
                prim_vsini.append(float(v) * 0.8)
                vsini_dict[root] = float(v) * 0.8
            except IndexError:
                logging.warn('No vsini found for star {}! No primary star removal will be attempted!'.format(star))
                prim_vsini.append(None)

    return prim_vsini
