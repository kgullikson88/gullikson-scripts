from __future__ import division, print_function
import os
import os.path
import pickle

import numpy as np
from pkg_resources import resource_filename
from scipy.interpolate import LinearNDInterpolator as interpnd


try:
    import pandas as pd
except ImportError:
    pd = None

from isochrones.isochrone import Isochrone

DATADIR = os.getenv('ISOCHRONES',
                    os.path.expanduser(os.path.join('~', '.isochrones')))
if not os.path.exists(DATADIR):
    os.mkdir(DATADIR)

MASTERFILE = '{}/Feiden.h5'.format(DATADIR)
TRI_FILE = '{}/Feiden.tri'.format(DATADIR)

MAXAGES = np.load(resource_filename('isochrones', 'data/dartmouth_maxages.npz'))
MAXAGE = interpnd(MAXAGES['points'], MAXAGES['maxages'])


# def _download_h5():
#    """
#    Downloads HDF5 file containing Dartmouth grids from Zenodo.
#    """
#    #url = 'http://zenodo.org/record/12800/files/dartmouth.h5'
#    url = 'http://zenodo.org/record/15843/files/dartmouth.h5'
#    from six.moves import urllib
#    print('Downloading Dartmouth stellar model data (should happen only once)...')
#    if os.path.exists(MASTERFILE):
#        os.remove(MASTERFILE)
#    urllib.request.urlretrieve(url,MASTERFILE)

#def _download_tri():
#    """
#    Downloads pre-computed triangulation for Dartmouth grids from Zenodo.
#    """
#    #url = 'http://zenodo.org/record/12800/files/dartmouth.tri'
#    #url = 'http://zenodo.org/record/15843/files/dartmouth.tri'
#    url = 'http://zenodo.org/record/17627/files/dartmouth.tri'
#    from six.moves import urllib
#    print('Downloading Dartmouth isochrone pre-computed triangulation (should happen only once...)')
#    if os.path.exists(TRI_FILE):
#        os.remove(TRI_FILE)
#    urllib.request.urlretrieve(url,TRI_FILE)

#if not os.path.exists(MASTERFILE):
#    _download_h5()

#if not os.path.exists(TRI_FILE):
#    _download_tri()

#Check to see if you have the right dataframe and tri file
#import hashlib

#DF_SHASUM = '0515e83521f03cfe3ab8bafcb9c8187a90fd50c7'
#TRI_SHASUM = 'e05a06c799abae3d526ac83ceeea5e6df691a16d'

#if hashlib.sha1(open(MASTERFILE, 'rb').read()).hexdigest() != DF_SHASUM:
#    raise ImportError('You have a wrong/corrupted/outdated Dartmouth DataFrame!' +
#                      ' Delete {} and try re-importing to download afresh.'.format(MASTERFILE))
#if hashlib.sha1(open(TRI_FILE, 'rb').read()).hexdigest() != TRI_SHASUM:
#    raise ImportError('You have a wrong/corrupted/outdated Dartmouth triangulation!' +
#                      ' Delete {} and try re-importing to download afresh.'.format(TRI_FILE))

#


if pd is not None:
    MASTERDF = pd.read_hdf(MASTERFILE, 'df').dropna()  #temporary hack
else:
    MASTERDF = None


class Feiden_Isochrone(Isochrone):
    """Dotter (2008) Stellar Models, at solar a/Fe and He abundances.
    :param bands: (optional)
        List of desired photometric bands.  Must be a subset of
        ``['U','B','V','R','I','J','H','K','g','r','i','z','Kepler','D51',
        'W1','W2','W3']``, which is the default.  W4 is not included
        because it does not have a well-measured A(lambda)/A(V).
    """

    def __init__(self, bands=None, **kwargs):

        df = MASTERDF
        log_ages = np.log10(df['Age'])
        minage = log_ages.min()
        maxage = log_ages.max()

        # make copies that claim to have different metallicities. This is a lie, but makes things work.
        lowmet = df.copy()
        lowmet['feh'] = -0.1
        highmet = df.copy()
        highmet['feh'] = 0.1
        df = pd.concat((df, lowmet, highmet))

        mags = {}
        if bands is not None:
            for band in bands:
                try:
                    if band in ['g', 'r', 'i', 'z']:
                        mags[band] = df['sdss_{}'.format(band)]
                    else:
                        mags[band] = df[band]
                except:
                    if band == 'kep' or band == 'Kepler':
                        mags[band] = df['Kp']
                    elif band == 'K':
                        mags['K'] = df['Ks']
                    else:
                        raise

        tri = None
        try:
            f = open(TRI_FILE, 'rb')
            tri = pickle.load(f)
        except:
            f = open(TRI_FILE, 'rb')
            tri = pickle.load(f, encoding='latin-1')
        finally:
            f.close()

        Isochrone.__init__(self, m_ini=df['Msun'], age=np.log10(df['Age']),
                           feh=df['feh'], m_act=df['Msun'], logL=df['logL'],
                           Teff=10 ** df['logT'], logg=df['logg'], mags=mags,
                           tri=tri, minage=minage, maxage=maxage, **kwargs)


