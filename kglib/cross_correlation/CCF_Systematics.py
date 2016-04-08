from __future__ import print_function, division, absolute_import

import os
import re
from collections import defaultdict
from operator import itemgetter
import logging
import sys

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, fmin
import matplotlib.pyplot as plt
import numpy as np
import emcee
import h5py

import pandas as pd
from george import kernels
import george
from kglib import fitters
from kglib.utils import StarData
from kglib.utils.HelperFunctions import mad, integral
from kglib.spectral_type import SpectralTypeRelations


def classify_filename(fname, type='bright'):
    """
    Given a CCF filename, classify the star combination, temperature, metallicity, and vsini
    :param fname:
    :return:
    """
    # First, remove any leading directories
    fname = fname.split('/')[-1]

    # Star combination
    m1 = re.search('\.[0-9]+kps', fname)
    stars = fname[:m1.start()]
    star1 = stars.split('+')[0].replace('_', ' ')
    star2 = stars.split('+')[1].split('_{}'.format(type))[0].replace('_', ' ')

    # secondary star vsini
    vsini = float(fname[m1.start() + 1:].split('kps')[0])

    # Temperature
    m2 = re.search('[0-9]+\.0K', fname)
    temp = float(m2.group()[:-1])

    # logg
    m3 = re.search('K\+[0-9]\.[0-9]', fname)
    logg = float(m3.group()[1:])

    # metallicity
    metal = float(fname.split(str(logg))[-1])

    return star1, star2, vsini, temp, logg, metal


def get_ccf_data(basedir, primary_name=None, secondary_name=None, vel_arr=np.arange(-900.0, 900.0, 0.1), type='bright'):
    """
    Searches the given directory for CCF files, and classifies
    by star, temperature, metallicity, and vsini
    :param basedir: The directory to search for CCF files
    :keyword primary_name: Optional keyword. If given, it will only get the requested primary star data
    :keyword secondary_name: Same as primary_name, but only reads ccfs for the given secondary
    :keyword vel_arr: The velocities to interpolate each ccf at
    :return: pandas DataFrame
    """
    if not basedir.endswith('/'):
        basedir += '/'
    all_files = ['{}{}'.format(basedir, f) for f in os.listdir(basedir) if type in f.lower()]
    primary = []
    secondary = []
    vsini_values = []
    temperature = []
    gravity = []
    metallicity = []
    ccf = []
    for fname in all_files:
        star1, star2, vsini, temp, logg, metal = classify_filename(fname, type=type)
        if primary_name is not None and star1.lower() != primary_name.lower():
            continue
        if secondary_name is not None and star2.lower() != secondary_name.lower():
            continue
        vel, corr = np.loadtxt(fname, unpack=True)
        fcn = spline(vel, corr)
        ccf.append(fcn(vel_arr))
        primary.append(star1)
        secondary.append(star2)
        vsini_values.append(vsini)
        temperature.append(temp)
        gravity.append(logg)
        metallicity.append(metal)

    # Make a pandas dataframe with all this data
    df = pd.DataFrame(data={'Primary': primary, 'Secondary': secondary, 'Temperature': temperature,
                                'vsini': vsini_values, 'logg': gravity, '[Fe/H]': metallicity, 'CCF': ccf})
    return df


def get_ccf_summary(hdf5_filename, vel_arr=np.arange(-900.0, 900.0, 0.1), excel_filename=None,
                    velocity='highest', addmode='simple', Tmin=3000, Tmax=7000, N_best=1, debug=False):
    """
    Goes through the given HDF5 file, and finds the best set of parameters for each combination of primary/secondary star
    :param hdf5_filename: The HDF5 file containing the CCF data
    :keyword excel_filename: The filename of an MS excel file giving the velocity for each secondary star.
                             The data must be in the first sheet, and three must be columns labeled 
                             'Star' and 'CCF RV'. Only used if velocity='excel'
    :keyword velocity: The velocity to measure the CCF at. Options are:
                       - 'highest' (default): uses the maximum of the ccf
                       - value: A numeric type giving the velocity to to use.
                       - 'excel': Search the filename excel_filename for the velocity of each secondary star
    :keyword vel_arr: The velocities to interpolate each ccf at
    :keyword addmode: The way the CCF orders were added while generating the ccfs
    :keyword debug: If True, it prints the progress. Otherwise, does its work silently and takes a while
    :keyword Tmin, Tmax: The minimum and maximum temperatures to include in the output.
    :keyword N_best: Passed to find_best_pars()
    :return: pandas DataFrame summarizing the best parameters.
             This is the type of dataframe to give to the other function here
    """
    if velocity.lower() == 'excel':
        table = pd.read_excel(excel_filename, 0)

    summary_dfs = []
    with h5py.File(hdf5_filename, 'r') as f:
        primaries = f.keys()
        for p in primaries:
            secondaries = f[p].keys()
            for s in secondaries:
                if addmode not in f[p][s].keys():
                    continue
                logging.info('Primary: {}\tSecondary: {}'.format(p, s))
                if velocity.lower() == 'excel':
                    try:
                        vel_max = table.loc[table.Star.str.lower().str.contains(s.strip().lower())]['CCF RV'].item()
                    except ValueError:
                        logging.warning('No entry found for star "{}" in table {}'.format(s, excel_filename))
                        continue
                else:
                    vel_max = velocity
                datasets = f[p][s][addmode].keys()
                vsini_values = []
                temperature = []
                gravity = []
                metallicity = []
                ccf = []
                for i, d in enumerate(datasets):
                    if debug:
                        sys.stdout.write('\r\t\tDataset {}/{}'.format(i+1, len(datasets)))
                        sys.stdout.flush()
                    ds = f[p][s][addmode][d]
                    if Tmin <= ds.attrs['T'] <= Tmax:
                        if ds.value.shape[0] == 2:
                            vel, corr = ds.value
                        elif 'velocity' in ds.attrs:
                            vel, corr = ds.attrs['velocity'], ds.value
                        else:
                            raise KeyError('Cannot find velocity information for dataset {}'.format(ds.name))
                        fcn = spline(vel, corr)
                        
                        vsini_values.append(ds.attrs['vsini'])
                        temperature.append(ds.attrs['T'])
                        gravity.append(ds.attrs['logg'])
                        metallicity.append(ds.attrs['[Fe/H]'])
                        ccf.append(fcn(vel_arr))
                data = pd.DataFrame(data={'Primary': [p]*len(ccf), 'Secondary': [s]*len(ccf),
                                          'Temperature': temperature, 'vsini': vsini_values,
                                          'logg': gravity, '[Fe/H]': metallicity, 'CCF': ccf})
                data.drop_duplicates(subset=('Temperature', 'vsini', 'logg', '[Fe/H]', 'Primary', 'Secondary'),
                                     inplace=True)
                summary_dfs.append(find_best_pars(data, velocity=vel_max, vel_arr=vel_arr, N=N_best))
                del data

    return pd.concat(summary_dfs, ignore_index=True)


def find_best_pars(df, velocity='highest', vel_arr=np.arange(-900.0, 900.0, 0.1), N=1):
    """
    Find the 'best-fit' parameters for each combination of primary and secondary star
    :param df: the dataframe to search in
    :keyword velocity: The velocity to measure the CCF at. The default is 'highest', and uses the maximum of the ccf
    :keyword vel_arr: The velocities to interpolate each ccf at
    :keyword N: The number of parameters to return
    :return: a dataframe with keys of primary, secondary, and the parameters
    """
    # Make sure N is odd
    if N % 2 == 0:
        logging.warn('N must be an odd number. Changing N from {} --> {}'.format(N, N + 1))
        N += 1


    # Get the names of the primary and secondary stars
    primary_names = pd.unique(df.Primary)
    secondary_names = pd.unique(df.Secondary)

    # Find the ccf value at the given velocity
    def val_fcn(ccf, idx=None, search_indices=None):
        if idx is None:
            if search_indices is None:
                idx = np.argmax(ccf)
            else:
                idx = np.argmax(ccf[search_indices])
                idx = search_indices[idx]
        rv = vel_arr[idx]
        sigma = np.std(ccf[np.abs(vel_arr - rv) > 200])
        return ccf[idx], ccf[idx] / sigma, rv
    if velocity == 'highest':
        vals = df['CCF'].map(val_fcn)
        df['ccf_max'] = vals.map(lambda l: l[0])
        df['significance'] = vals.map(lambda l: l[1])
        df['rv'] = vals.map(lambda l: l[2])
    else:
        # idx = np.argmin(np.abs(vel_arr - velocity))
        idx = np.where(np.abs(vel_arr - velocity) <= 5)[0]
        vals = df['CCF'].map(lambda c: val_fcn(c, search_indices=idx))
        df['ccf_max'] = vals.map(lambda l: l[0])
        df['significance'] = vals.map(lambda l: l[1])
        df['rv'] = vals.map(lambda l: l[2])
        #print(df[['Secondary', 'rv']])

    # Find the best parameter for each combination
    d = defaultdict(list)
    groups = df.groupby(('Primary', 'Secondary'))
    for group in groups.groups.keys():
        primary = group[0]
        secondary = group[1]
        g = groups.get_group(group)
        best = g.loc[g.ccf_max == g.ccf_max.max()]
        T = best['Temperature'].item()
        vsini = best['vsini'].item()
        logg = best['logg'].item()
        metal = best['[Fe/H]'].item()
        rv = best['rv'].item()
        Tmin = T - (N - 1) * 50
        Tmax = T + (N - 1) * 50
        for Ti in range(Tmin, Tmax + 1, 100):
            good = g.loc[
                (g['Temperature'] == Ti) & (g['vsini'] == vsini) & (g['logg'] == logg) & (g['[Fe/H]'] == metal)]
            if len(good) == 0:
                logging.warn('No matches for T = {} with primary/secondary = {}/{}!'.format(Ti, primary, secondary))
                d['Primary'].append(primary)
                d['Secondary'].append(secondary)
                d['Temperature'].append(Ti)
                d['vsini'].append(vsini)
                d['logg'].append(logg)
                d['[Fe/H]'].append(metal)
                d['rv'].append(rv)
                d['CCF'].append(np.nan)
                d['significance'].append(np.nan)
                continue
            # print len(good)
            best = good.loc[good.ccf_max == good.ccf_max.max()]
            #best = good
            if len(best) != 1 or any(np.isnan(best['CCF'].item())):
                print(best)
                print(good)
                print(good.ccf_max)
                print(good.ccf_max.max())
                continue

            # Save the best parameters for this temperature
            d['Primary'].append(primary)
            d['Secondary'].append(secondary)
            d['Temperature'].append(best['Temperature'].item())
            d['vsini'].append(best['vsini'].item())
            d['logg'].append(best['logg'].item())
            d['[Fe/H]'].append(best['[Fe/H]'].item())
            idx = np.argmin(np.abs(vel_arr - rv))
            d['rv'].append(rv)
            d['CCF'].append(best['CCF'].item()[idx])
            # d['rv'].append(best['rv'].item())
            #d['CCF'].append(best.ccf_max.item())

            # Measure the detection significance
            std = mad(best.CCF.item())
            mean = np.median(best.CCF.item())
            d['significance'].append((d['CCF'][-1] - mean) / std)

    return pd.DataFrame(data=d)


def get_detected_objects(df, tol=1.0, debug=False):
    """
    Takes a summary dataframe with RV information. Finds the median rv for each star,
      and removes objects that are more than 'tol' km/s from the median value
    :param df: A summary dataframe, such as created by get_ccf_summary or find_best_pars
    :param tol: The tolerance, in km/s, to accept an observation as detected
    :return: a dataframe containing only detected companions
    """
    secondary_names = pd.unique(df.Secondary)
    secondary_to_rv = defaultdict(float)
    for secondary in secondary_names:
        rv = df.loc[df.Secondary == secondary]['rv'].median()
        secondary_to_rv[secondary] = rv

    if debug:
        for secondary in sorted(secondary_to_rv.keys()):
            print ('RV for {}: {:.2f} km/s'.format(secondary, secondary_to_rv[secondary]))

    keys = df.Secondary.values
    good = df.loc[abs(df.rv.values - np.array(itemgetter(*keys)(secondary_to_rv))) < tol]
    return good


def get_detected_objects_new(df, siglim=5, Terr_lim=3, Toffset=2000):
    """
    Get a dataframe with only the detected objects.
    :param df: A DataFrame such as one output by get_ccf_summary with N > 1
    :param siglim: The minimum significance to count as detected
    :param Terr_lim: The maximum number of standard deviations of (Measured - Actual) to allow for detected objects
    :param Toffset: The absolute difference to allow between the true and measured temperature.
    :return: A dataframe similar to df, but with fewer rows
    """
    S = get_initial_uncertainty(df)
    S['Tdiff'] = S.Tmeas - S.Tactual
    mean, std = S.Tdiff.mean(), S.Tdiff.std()
    detected = S.loc[(S.significance > siglim) & (S.Tdiff - mean < Terr_lim * std) & (abs(S.Tdiff) < Toffset)]
    return pd.merge(detected[['Primary', 'Secondary']], df, on=['Primary', 'Secondary'], how='left')


def add_actual_temperature(df, method='excel', filename='SecondaryStar_Temperatures.xls'):
    """
    Add the actual temperature to a given summary dataframe
    :param df: The dataframe to which we will add the actual secondary star temperature
    :keyword method: How to get the actual temperature. Options are:
                   - 'spt': Use main-sequence relationships to go from spectral type --> temperature
                   - 'excel': Use tabulated data, available in the file 'SecondaryStar_Temperatures.xls'
    :keyword filename: The filename of the excel spreadsheet containing the literature temperatures.
                       Needs to have the right format! Ignored if method='spt'
    :return: copy of the original dataframe, with an extra column for the secondary star temperature
    """
    # First, get a list of the secondary stars in the data
    secondary_names = pd.unique(df.Secondary)
    secondary_to_temperature = defaultdict(float)
    secondary_to_error = defaultdict(float)

    if method.lower() == 'spt':
        MS = SpectralTypeRelations.MainSequence()
        for secondary in secondary_names:
            star_data = StarData.GetData(secondary)
            spt = star_data.spectype[0] + re.search('[0-9]\.*[0-9]*', star_data.spectype).group()
            T_sec = MS.Interpolate(MS.Temperature, spt)
            secondary_to_temperature[secondary] = T_sec

    elif method.lower() == 'excel':
        table = pd.read_excel(filename, 0)
        for secondary in secondary_names:
            T_sec = table.loc[table.Star.str.lower().str.contains(secondary.strip().lower())]['Literature_Temp'].item()
            T_error = table.loc[table.Star.str.lower().str.contains(secondary.strip().lower())][
                'Literature_error'].item()
            secondary_to_temperature[secondary] = T_sec
            secondary_to_error[secondary] = T_error

    df['Tactual'] = df['Secondary'].map(lambda s: secondary_to_temperature[s])
    df['Tact_err'] = df['Secondary'].map(lambda s: secondary_to_error[s])
    return


def make_gaussian_process_samples(df):
    """
    Make a gaussian process fitting the Tactual-Tmeasured relationship
    :param df: pandas DataFrame with columns 'Temperature' (with the measured temperature)
                 and 'Tactual' (for the actual temperature)
    :return: emcee sampler instance
    """
    Tmeasured, Tactual, error, lit_err = get_values(df)
    for i, e in enumerate(error):
        if e < 1:
            e = fit_sigma(df, i)
        error[i] = np.sqrt(e**2 + lit_err[i]**2)
    for Tm, Ta, e in zip(Tmeasured, Tactual, error):
        print(Tm, Ta, e)
    plt.figure(1)
    limits = [3000, 7000]
    plt.errorbar(Tmeasured, Tactual, yerr=error, fmt='.k', capsize=0)
    plt.plot(limits, limits, 'r--')
    plt.xlabel('Measured Temperature')
    plt.ylabel('Actual Temperature')
    plt.xlim(limits)
    plt.ylim(limits)

    # Define some functions to use in the GP fit
    def model(pars, T):
        #polypars = pars[2:]
        #return np.poly1d(polypars)(T)
        return T

    def lnlike(pars, Tact, Tmeas, Terr):
        a, tau = np.exp(pars[:2])
        gp = george.GP(a * kernels.ExpSquaredKernel(tau))
        gp.compute(Tmeas, Terr)
        return gp.lnlikelihood(Tact - model(pars, Tmeas))

    def lnprior(pars):
        lna, lntau = pars[:2]
        polypars = pars[2:]
        if -20 < lna < 20 and 12 < lntau < 20:
            return 0.0
        return -np.inf

    def lnprob(pars, x, y, yerr):
        lp = lnprior(pars)
        return lp + lnlike(pars, x, y, yerr) if np.isfinite(lp) else -np.inf

    # Set up the emcee fitter
    initial = np.array([0, 14])#, 1.0, 0.0])
    ndim = len(initial)
    nwalkers = 100
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Tactual, Tmeasured, error))

    print('Running first burn-in')
    p1, lnp, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running second burn-in...")
    p_best = p1[np.argmax(lnp)]
    p2 = [p_best + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    p3, _, _ = sampler.run_mcmc(p2, 250)
    sampler.reset()

    print("Running production...")
    sampler.run_mcmc(p3, 1000)

    # We now need to increase the spread of the posterior distribution so that it encompasses the right number of data points
    # This is because the way we have been treating error bars here is kind of funky...
    # First, generate a posterior distribution of Tactual for every possible Tmeasured
    print('Generating posterior samples at all temperatures...')
    N = 10000  # This is 1/10th of the total number of samples!
    idx = np.argsort(-sampler.lnprobability.flatten())[:N]  # Get N 'best' curves
    par_vals = sampler.flatchain[idx]
    Tvalues = np.arange(3000, 6900, 100)
    gp_posterior = []
    for pars in par_vals:
        a, tau = np.exp(pars[:2])
        gp = george.GP(a * kernels.ExpSquaredKernel(tau))
        gp.compute(Tmeasured, error)
        s = gp.sample_conditional(Tactual - model(pars, Tmeasured), Tvalues) + model(pars, Tvalues)
        gp_posterior.append(s)

    # Get the median and spread in the pdf
    gp_posterior = np.array(gp_posterior)
    medians = np.median(gp_posterior, axis=0)
    sigma_pdf = np.std(gp_posterior, axis=0)

    # Correct the data and get the residual spread
    df['Corrected_Temperature'] = df['Temperature'].map(lambda T: medians[np.argmin(abs(T - Tvalues))])
    sigma_spread = np.std(df.Tactual - df.Corrected_Temperature)

    # Increase the spread in the pdf to reflect the residual spread
    ratio = np.maximum(np.ones(sigma_pdf.size), sigma_spread / sigma_pdf)
    gp_corrected = (gp_posterior - medians) * ratio + medians

    # Make confidence intervals
    l, m, h = np.percentile(gp_corrected, [16.0, 50.0, 84.0], axis=0)
    conf = pd.DataFrame(data={'Measured Temperature': Tvalues, 'Actual Temperature': m,
                              'Lower Bound': l, 'Upper bound': h})
    conf.to_csv('Confidence_Intervals.csv', index=False)


    # Finally, plot a bunch of the fits
    print("Plotting...")
    N = 300
    Tvalues = np.arange(3000, 7000, 20)
    idx = np.argsort(-sampler.lnprobability.flatten())[:N]  # Get N 'best' curves
    par_vals = sampler.flatchain[idx]
    plot_posterior = []
    for i, pars in enumerate(par_vals):
        a, tau = np.exp(pars[:2])
        gp = george.GP(a * kernels.ExpSquaredKernel(tau))
        gp.compute(Tmeasured, error)
        s = gp.sample_conditional(Tactual - model(pars, Tmeasured), Tvalues) + model(pars, Tvalues)
        plot_posterior.append(s)
    plot_posterior = np.array(plot_posterior)
    medians = np.median(plot_posterior, axis=0)
    sigma_pdf = np.std(plot_posterior, axis=0)

    # Increase the spread in the pdf to reflect the residual spread
    ratio = np.maximum(np.ones(sigma_pdf.size), sigma_spread / sigma_pdf)
    plot_posterior = (plot_posterior - medians) * ratio + medians
    plt.plot(Tvalues, plot_posterior.T, 'b-', alpha=0.05)

    plt.draw()
    plt.savefig('Temperature_Correspondence.pdf')

    return sampler, gp_corrected


def check_posterior(df, posterior, Tvalues=np.arange(3000, 6900, 100)):
    """
    Checks the posterior samples: Are 95% of the measurements within 2-sigma of the prediction?
    :param df: The summary dataframe
    :param posterior: The MCMC predicted values
    :param Tvalues: The measured temperatures the posterior was made with
    :return: boolean, as well as some warning messages if applicable
    """
    # First, make 2-sigma confidence intervals
    l, m, h = np.percentile(posterior, [5.0, 50.0, 95.0], axis=0)

    Ntot = []  # The total number of observations with the given measured temperature
    Nacc = []  # The number that have actual temperatures within the confidence interval

    g = df.groupby('Temperature')
    for i, T in enumerate(Tvalues):
        if T in g.groups.keys():
            Ta = g.get_group(T)['Tactual']
            low, high = l[i], h[i]
            Ntot.append(len(Ta))
            Nacc.append(len(Ta.loc[(Ta >= low) & (Ta <= high)]))
            p = float(Nacc[-1]) / float(Ntot[-1])
            if p < 0.95:
                logging.warn(
                    'Only {}/{} of the samples ({:.2f}%) were accepted for T = {} K'.format(Nacc[-1], Ntot[-1], p * 100,
                                                                                            T))
                print(low, high)
                print(sorted(Ta))
        else:
            Ntot.append(0)
            Nacc.append(0)

    p = float(sum(Nacc)) / float(sum(Ntot))
    if p < 0.95:
        logging.warn('Only {:.2f}% of the total samples were accepted!'.format(p * 100))
        return False
    return True


def get_Tmeas(d, include_actual=True):
    d = d.dropna(subset=['CCF'])
    corr = d.CCF.values
    corr += 1.0 - corr.max()
    T = d.Temperature.values
    w = corr / corr.sum()
    Tmeas = np.average(T, weights=w)
    var_T = np.average((T - Tmeas) ** 2, weights=w)

    # Get factor to go from biased --> unbiased variance
    V1 = np.sum(w)
    V2 = np.sum(w ** 2)
    f = V1 / (V1 - V2 / V1)

    # Finally, get the peak significance
    sig = d['significance'].max()

    if include_actual:
        return pd.DataFrame(data={'[Fe/H]': d['[Fe/H]'].values[0], 'logg': d.logg.values[0],
                                  'rv': d.rv.values[0], 'vsini': d.vsini.values[0],
                                  'Tactual': d.Tactual.values[0], 'Tact_err': d.Tact_err.values[0],
                                  'Tmeas': Tmeas, 'Tmeas_err': np.sqrt(f * var_T), 'significance': sig}, index=[0])
    else:
        return pd.DataFrame(data={'[Fe/H]': d['[Fe/H]'].values[0], 'logg': d.logg.values[0],
                                  'rv': d.rv.values[0], 'vsini': d.vsini.values[0],
                                  'Tmeas': Tmeas, 'Tmeas_err': np.sqrt(f * var_T), 'significance': sig}, index=[0])


def get_initial_uncertainty(df):
    """
    Take a dataframe such as one output from get_ccf_summary with N > 1, and get the temperature and initial
    estimate for the temperature uncertainty.
    :param df:
    :return:
    """

    # Get the measured temperature as the weighted average of the temperatures (weight by normalized CCF value)


    summary = df.groupby(('Primary', 'Secondary')).apply(get_Tmeas).reset_index()

    return summary


class GPFitter(fitters.Bayesian_LS):
    def _lnlike(self, pars):
        """
        likelihood function. This uses the class variables for x,y,xerr, and yerr, as well as the 'model' instance.
        """
        y_pred = self.x

        a, tau = np.exp(pars[:2])
        gp = george.GP(a * kernels.ExpSquaredKernel(tau))
        gp.compute(self.x, self.yerr)
        return gp.lnlikelihood(self.y - y_pred)

    def lnprior(self, pars):
        lna, lntau = pars[:2]
        polypars = pars[2:]
        if -20 < lna < 30 and 0 < lntau < 30:
            return 0.0
        return -np.inf

    def guess_fit_parameters(self):
        return [0, 10]

    def predict(self, x, N=100, highest=False):
        """
        Predict the y value for the given x values.
        """
        if self.sampler is None:
            logging.warn('Need to run the fit method before predict!')
            return

        # Find the N best walkers
        if N == 'all':
            N = self.sampler.flatchain.shape[0]
        else:
            N = min(N, self.sampler.flatchain.shape[0])
        
        if highest:
            indices = np.argsort(self.sampler.flatlnprobability)[:N]
            pars = self.sampler.flatchain[indices]
        else:
            pars = self.sampler.flatchain[:N]

        yvals = []
        for i, p in enumerate(pars):
            logging.info('Generating GP samples for iteration {}/{}'.format(i+1, len(pars)))
            a, tau = np.exp(p[:2])
            gp = george.GP(a * kernels.ExpSquaredKernel(tau))
            gp.compute(self.x, self.yerr)
            s = gp.sample_conditional(self.y - self.x, x) + x
            yvals.append(s)

        return np.array(yvals)


class ModifiedPolynomial(fitters.Bayesian_LS):
    def model(self, p, x):
        s, m = 10**p[:2]
        polypars = p[2:]
        return np.poly1d(polypars)(x)  * np.exp(-s*(x-m)**2)  + x

    def guess_fit_parameters(self, fitorder=1):
        polypars = np.zeros(fitorder + 1)
        polypars[-2] = 1.0
        pars = [-7, 3.5]
        pars.extend(polypars)
        min_func = lambda p, xi, yi, yerri: np.sum((yi - self.model(p, xi)) ** 2 / yerri ** 2)

        best_pars = fmin(min_func, x0=pars, args=(self.x, self.y, self.yerr))
        self.guess_pars = best_pars
        return best_pars



def fit_act2tmeas(df, nwalkers=500, n_burn=200, n_prod=500, fitorder=1, fitter_class=None):
    """
    Fit a function to go from actual to measured temperature. Use Bayes' Theorem to get the reverse!
    :param df: A pandas DataFrame such as one output by get_ccf_summary with N > 1
    :param fitorder: The order of the fit
    :return:
    """

    # Get the measured temperature for each primary/secondary combination
    summary = get_initial_uncertainty(df)

    # Get the average and standard deviation of the measured temperature, for a given actual temperature.
    def get_avg_T(df):
        Tm = df.Tmeas.values
        Tm_err = df.Tmeas_err.values
        w = 1.0 / Tm_err ** 2
        w /= w.sum()
        T_avg = np.average(Tm, weights=w)
        var_T = np.average((Tm - T_avg) ** 2, weights=w)

        # Get factor to go from biased --> unbiased variance
        V1 = np.sum(w)
        V2 = np.sum(w ** 2)
        f = V1 / (V1 - V2 / V1)

        # return Tmeas, np.sqrt(f*var_T)
        return pd.DataFrame(data={'Tactual': df.Tactual.values[0], 'Tact_err': df.Tact_err.values[0],
                                  'Tmeas': T_avg, 'Tmeas_err': np.sqrt(f * var_T)}, index=[0])

    final = summary.groupby('Secondary').apply(get_avg_T).reset_index()

    # Don't let the error bars get smaller than 50 K
    final['Tmeas_err'] = np.maximum(final.Tmeas_err, 100.0)

    # Save the measurement data
    final[['Tactual', 'Tmeas', 'Tact_err', 'Tmeas_err']].to_csv('Systematics_Data.csv')

    # Fit to a polynomial with a gaussian process noise model.
    if fitter_class is None:
        #fitter = GPFitter(final.Tactual, final.Tmeas, final.Tmeas_err)
        fitter = ModifiedPolynomial(final.Tactual, final.Tmeas, final.Tmeas_err)
    else:
        fitter = fitter_class(final.Tactual, final.Tmeas, final.Tmeas_err)
    fitter.fit(nwalkers=nwalkers, n_burn=n_burn, n_prod=n_prod, fitorder=fitorder)
    par_samples = fitter.sampler.flatchain


    # Plot
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0.15, bottom=0.18)
    ax.errorbar(final.Tactual, final.Tmeas, xerr=final.Tact_err, yerr=final.Tmeas_err, fmt='ko')
    lim = [3000, 7000]
    xplot = np.linspace(lim[0], lim[1], 100)
    ypred = fitter.predict(xplot, N=300)
    for i in range(ypred.shape[0]):
        yplot = ypred[i]
        ax.plot(xplot, yplot, 'b-', alpha=0.03)
    ax.set_xlabel('Literature Temperature (K)')
    ax.set_ylabel('Measured Temperature (K)')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.plot(lim, lim, 'r--')
    plt.savefig('Tact2Tmeas.pdf')

    # Now, plot the fractional difference vs Tactual
    fig2, ax2 = plt.subplots(1, 1)
    delta = (final.Tmeas - final.Tactual)/final.Tactual
    delta_var = (final.Tmeas_err/final.Tactual)**2 + (final.Tmeas/final.Tactual**2 * final.Tact_err)**2
    ax2.errorbar(final.Tactual, delta, xerr=final.Tact_err, yerr=np.sqrt(delta_var), fmt='ko')
    for i in range(ypred.shape[0]):
        #ypred = np.poly1d(par_samples[i])(xplot)
        del_plot = (ypred[i] - xplot)/xplot
        ax2.plot(xplot, del_plot, 'b-', alpha=0.03)
    ax2.set_xlabel('Literature Temperature')
    ax2.set_ylabel('Fractional Error')
    lim = ax2.get_xlim()
    ax2.plot(lim, [0, 0], 'r--')
    plt.savefig('Tact2Tmeas_Residual.pdf')

    plt.show()

    return fitter


def get_actual_temperature(fitter, Tmeas, Tmeas_err, cache=None, ret_cache=None, summarize=True):
    """
    Get the actual temperature from the measured temperature
    :param fitter: a Bayesian_TLS instance which has already been fit
    :param Tmeas: the measured temperature. Either a float or a numpy array with independent temperatures
    :param Tmeas_err: uncertainty on the measured temperature. Same shape as Tmeas.
    :return: posterior samples for the actual temperature
    """

    # First, build up a cache of the MCMC predicted measured temperatures for lots of actual temperatures
    if cache is None:
        logging.info('Generating cache...')
        Ta_arr = np.arange(2000, 10000, 1.0)
        Tmeas_pred = fitter.predict(Ta_arr, N=10000)
        cache = pd.DataFrame(Tmeas_pred, columns=Ta_arr)
        ret_cache = True if ret_cache is None else False
        del Tmeas_pred

    # Get the probability of each value in the cache
    Tmeas = np.atleast_1d(Tmeas).astype(np.float)
    Tmeas_err = np.atleast_1d(Tmeas_err).astype(np.float)
    def get_prob(Tm_pred, Tm, Tm_err):
        return np.exp(-((Tm_pred - Tm) / Tm_err)**2)

    probs = np.array([get_prob(cache.values, Tm, Tme) for Tm, Tme in zip(Tmeas, Tmeas_err)])

    # Get the posterior probability distribution
    tmp = np.mean(probs, axis=1)
    tmp /= np.sum(tmp, axis=1)[:, np.newaxis]
    P = np.prod(tmp, axis=0)

    # Find the maximum and FWHM of the probabilities
    #best_T = cache.columns.values[np.argmax(P)]
    #roots = fwhm(cache.columns.values, P, k=0, ret_roots=True)
    #h, l = max(roots), min(roots)
    l, best_T, h = integral(cache.columns.values, P, [0.16, 0.5, 0.84], k=0)


    print('$T = {}^{{+{}}}_{{-{}}}$'.format(best_T, h-best_T, best_T-l))

    # Return the requested things.
    if ret_cache:
        if summarize:
            return best_T, h - best_T, best_T - l, cache
        return cache.columns.values, P, cache
    if summarize:
        return best_T, h - best_T, best_T - l
    return cache.columns.values, P

def correct_measured_temperature(df, fitter, cache=None):
    """
    Given a dataframe such as output by get_ccf_data (with N > 1), correct the temperatures to
    account for the measurement bias.
    :param df: A dataframe with the CCF data
    :param fitter: A fitters.Bayesian_TLS instance that contains fitted parameters for the measurement bias
    :param cache: A pandas dataframe that gives MCMC samples of the temperature measurement
                 for various actual temperatures.
    :return:
    """

    # First, get the measurement values and estimated uncertainty
    data = get_initial_uncertainty(df)

    # Make a cache for get_actual_temperature if it is not given.
    if cache is None:
        logging.info('Generating cache...')
        Ta_arr = np.arange(2000, 10000, 1.0)
        Tmeas_pred = fitter.predict(Ta_arr, N=10000)
        cache = pd.DataFrame(Tmeas_pred, columns=Ta_arr)

    # Correct the measured temperatures to account for the bias.
    out = data.apply(lambda r: get_actual_temperature(fitter, r['Tmeas'], r['Tmeas_err'], cache=cache), axis=1)
    data['Corrected Temperature'] = out.map(lambda l: l[0])
    data['T_uperr'] = out.map(lambda l: l[1])
    data['T_lowerr'] = out.map(lambda l: l[2])

    return data


def get_uncertainty_scalefactor(df):
    """
    Find the factor by which to multiply the 1-sigma measurement uncertainties
    so that they agree with the literature values 68% of the time.

    :param df: A pandas DataFrame with corrected temperatures, such as output by correct_measured_temperature
    :return: The scaling factor. Multiply df['T_uperr'] and df['T_lowerr'] by this to get more realistic uncertainties.
    """

    def get_zscore(x, y, xerr, yerr, f=1.0):
        delta = x - y
        sigma = np.sqrt(f * xerr ** 2 + yerr ** 2)
        return delta / sigma

    def min_func(f, x, y, xerr, yerr):
        zscores = get_zscore(x, y, xerr, yerr, f)
        return (len(zscores[zscores ** 2 > 1]) / float(len(zscores)) - 0.32) ** 2

    df['T_err'] = np.minimum(df['T_uperr'], df['T_lowerr'])  # Be conservative and use the smaller error.

    fitresult = minimize_scalar(min_func, bounds=[0, 10], method='bounded', args=(df['Corrected Temperature'],
                                                                                  df['Tactual'],
                                                                                  df['T_err'],
                                                                                  df['Tact_err']))

    logging.info('Uncertainty scale factor = {:.2g}'.format(fitresult.x))

    return fitresult.x


def get_values(df):
    temp = df.groupby('Temperature')
    Tmeasured = temp.groups.keys()
    Tactual_values = [temp.get_group(Tm)['Tactual'].values for Tm in Tmeasured]
    Tactual = np.array([np.mean(Ta) for Ta in Tactual_values])
    spread = np.nan_to_num([np.std(Ta, ddof=1) for Ta in Tactual_values])
    literr_values = [temp.get_group(Tm)['Tact_err'].values for Tm in Tmeasured]
    lit_err = np.array([np.sqrt(np.sum(literr**2)) for literr in literr_values])
    return Tmeasured, Tactual, spread, lit_err


def integrate_gauss(x1, x2, amp, mean, sigma):
    """
    Integrate a gaussian between the points x1 and x2
    """
    gauss = lambda x, A, mu, sig: A*np.exp(-(x-mu)**2 / (2.0*sig**2))
    if x1 < -1e6:
        x1 = -np.inf
    if x2 > 1e6:
        x2 = np.inf
    result = quad(gauss, x1, x2, args=(amp, mean, sigma))
    return result[0]


def get_probability(x1, x2, x3, x4, N, mean, sigma, debug=False):
    """
    Get the probability of the given value of sigma
    x1-x4 are the four limits, which are the bin edges of the possible values Tactual can take
    N is the number of entries in the single bin, and mean what it sounds like
    """
    if x2 < 100:
        x2 = x3 - (x4-x3)
    if x4 > 1e6:
        x4 = x3 + (x3-x2)
    int1 = integrate_gauss(x2, x3, 1.0, mean, sigma)
    A = float(N) / int1
    int2 = 0 if x1 < 100 else integrate_gauss(x1, x2, A, mean, sigma)
    int3 = 0 if x4 > 1e6 else integrate_gauss(x3, x4, A, mean, sigma)
    if debug:
        print('\n')
        print(x1, x2, x3, x4, N, mean, sigma)
        print(int1)
        print(A)
        print(int2)
        print(int3)
    if int2 > 1 or int3 > 1:
        return 0
    return 1


def fit_sigma(df, i):
    """
    Find the largest allowable standard deviation, given the possible values Tactual can take.
    """
    Tmeasured, Tactual, _, _ = get_values(df)
    Tm = Tmeasured[i]
    
    # Get the possible values, and bin those with this measured value
    possible_values = sorted(pd.unique(df.Tactual))
    edges = [(possible_values[i] + possible_values[i+1])/2 for i in range(len(possible_values)-1)]
    bins = [0] + edges + [9e9]
    good = df.loc[df.Temperature == Tm]
    values, _= np.histogram(good.Tactual.values, bins=bins)
    
    mean = np.mean(good.Tactual.values)
    std = np.std(good.Tactual.values, ddof=1)
    if std > 0:
        return std
    
    sigma_test = np.arange(500, 10, -10) #Just test a bunch of values
    idx = np.searchsorted(bins, mean)
    idx = np.argmin(abs(np.array(bins) - mean))
    x1 = bins[idx-2] if idx > 2 else -1
    x2 = bins[idx-1]
    x3 = bins[idx]
    x4 = bins[idx+1] if idx < len(bins)-2 else np.inf
    N = len(good)
    probs = [get_probability(x1, x2, x3, x4, N, mean, s) for s in sigma_test]
    for s, p in zip(sigma_test, probs):
        if p > 0.5:
            return s
    
    # If we get here, just return a guess value
    return 200.0

    #raise ValueError('No probability > 0!')


if __name__ == '__main__':
    pass

