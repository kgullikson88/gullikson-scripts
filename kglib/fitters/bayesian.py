""" 
Implements Bayesian_LS, a class the makes MCMC sampling with either emcee or pymultinest
object-oriented, and requires only defining the model to fit and priors on the parameters.
"""

from __future__ import print_function, division, absolute_import

import logging
import os
import glob
import json
import warnings

import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import pandas as pd

from kglib.utils.HelperFunctions import ensure_dir


try:
    import corner
except ImportError:
    import triangle as corner

try:
    import emcee
except ImportError:
    emcee = None


class Bayesian_LS(object):
    """
    Fit a polynomial to data using either MCMC sampling with emcee, or
    Importance Nested Sampling with MultiNest. This class is mostly
    meant to be inherited for more interesting models.
    """

    def __init__(self, x=1, y=1, yerr=1, param_names=None):
        """
        Parameters:
        ===========
        - x:           numpy.ndarray
                       The independent variable

        - y:           numpy.ndarray
                       The dependent variable

        - yerr:        numpy.ndarray
                       The uncertainty in the dependent variable

        - param_names: Any iterable
                       A list of the parameter names. You MUST give this if using the
                       multinest backend.
        """
        self.x = x
        self.y = y
        self.yerr = yerr
        self.sampler = None
        self.samples = None
        self.n_params = None
        self.param_names = None
        if param_names is not None:
            self.n_params = len(param_names)
            self.param_names = param_names
        return

    def model(self, p, x):
        """
        A parameteric model to fit y = f(x, p)
        This can be overridden in a class that inherits from this one to make a new model

        Parameters:
        ===========
        - p:       Any iterable
                   The model parameters

        - x:       numpy.ndarray
                   The dependent variable

        Returns:
        ========
        A model at each of the x locations
        """
        return np.poly1d(p)(x)


    def lnlike(self, pars):
        """
        likelihood function. This uses the class variables for x,y,xerr, and yerr, as well as the 'model' instance.
        The model parameters are given in the `pars` iterable.
        """
        y_pred = self.model(pars, self.x)  # Predict the y value
        # Make the log-likelihood
        return -0.5 * np.sum((self.y - y_pred) ** 2 / self.yerr * 2 + np.log(2 * np.pi * self.yerr ** 2))


    def _lnlike(self, pars):
        warnings.warn('Use fitter.lnlike instead of fitter._lnlike!', DeprecationWarning)
        return self.lnlike(pars)


    def lnprior(self, pars):
        """
        Log of the prior for the parameters. This can be overridden to make custom priors
        """
        return 0.0


    def _lnprob(self, pars):
        """
        Log of the posterior probability of pars given the data.
        """
        lp = self.lnprior(pars)
        return lp + self._lnlike(pars) if np.isfinite(lp) else -np.inf


    def mnest_prior(self, cube, ndim, nparams):
        """
        This pretty much MUST be overridden for any practical use!
        Transform the 'cube' parameter, which holds everything being fit,
        from a uniform distibution on [0,1] to the prior probability distribution.
        (Use the inverse cumulative distribution function)

        Typical usage:
        ==============
        cube[1] = cube[1] * 10 - 5  # Uniform distribution from -5 to 5
        cube[2] = scipy.special.erf(cube[2])  # Standard Normal distribution
        """
        return


    def mnest_lnlike(self, cube, ndim, nparams):
        """
        This is probably okay as it is. You may (but probably not) need to override
        _lnlike, but not this one.
        """
        pars = np.array([cube[i] for i in range(nparams)])
        return self.lnlike(pars)


    def guess_fit_parameters(self, fitorder=1):
        """
        Do a normal (non-bayesian) fit to the data.
        The result will be saved for use as initial guess parameters in the full MCMC fit.
        If you use a custom model, you will probably have to override this method as well.
        """
        pars = np.zeros(fitorder + 1)
        pars[-2] = 1.0
        min_func = lambda p, xi, yi, yerri: np.sum((yi - self.model(p, xi)) ** 2 / yerri ** 2)

        best_pars = fmin(min_func, x0=pars, args=(self.x, self.y, self.yerr))
        self.guess_pars = best_pars
        return best_pars

    def fit(self, backend='emcee', *args, **kwargs):
        """
        Perform the full MCMC fit. This function calls either fit_multinest or fit_emcee, depending on the backend.
        See the doc-strings for those methods to learn what args and kwargs should be for each backend. The
        backend keyword can be either 'emcee' (the default) or 'multinest'. It is case-insensitive.

        """
        if backend.lower() == 'emcee':
            return self.fit_emcee(*args, **kwargs)
        elif backend.lower() == 'multinest':
            return self.fit_multinest(*args, **kwargs)


    def fit_emcee(self, nwalkers=100, n_burn=200, n_prod=1000, guess=True, initial_pars=None, **guess_kws):
        """
        Perform the full MCMC fit using emcee.

        Parameters:
        ===========
        - nwalkers:         integer
                            The number of walkers to use in the MCMC sampler

        - n_burn:           integer
                            The number of samples to discard for the burn-in portion

        - n_prod:           integer
                            The number of MCMC samples to take in the final production sampling

        - guess:            boolean
                            Flag for whether the data should be fit in a normal way first,
                            to get decent starting parameters.
                            If true, it uses self.guess_fit_parameters and passes guess_kws to the function.
                            If false, it uses initial_pars. You MUST give initial_pars if guess=False!

        - guess_kws:        Any additional keyword arguments to pass to `guess_fit_parameters()`

        - initial_pars:     Any iterable
                            Initial parameters to use. Should be either a 1d array with the guess pars
                            for each parameter, or a 2d array giving the range each parameter can take.
                            If 1d, the sampler will be initialized in a small ball near the guess values.
                            If 2d, the sampler will be initialized uniformly filling the volume.
        """

        # Make sure emcee exists
        if emcee is None:
            logging.warn('You must install emcee to use this backend!')
            return None

        if guess:
            initial_pars = self.guess_fit_parameters(**guess_kws)
        elif initial_pars is None:
            raise ValueError('Must give initial pars if guess = False!')

        # Give generic parameter names so that the triangle method works
        if self.param_names is None:
            self.n_params = len(initial_pars)
            self.param_names = ['c{}'.format(i) for i in range(self.n_params)]

        # Set up the MCMC sampler
        pars = np.array(initial_pars)
        if pars.ndim == 1:
            ndim = pars.size
            p0 = emcee.utils.sample_ball(pars, std=[1e-6] * ndim, size=nwalkers)
        elif pars.ndim == 2:
            ndim = pars.shape[0]
            p0 = np.random.uniform(low=pars[:, 0], high=pars[:, 1], size=(nwalkers, ndim))
        else:
            raise TypeError('initial_pars should be either 1d or 2d. You gave a {}d array!'.format(pars.ndim))

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)

        # Burn-in
        logging.info('Running burn-in')
        i = 0
        for p1, lnp, rstate in sampler.sample(p0, iterations=n_burn):
            if i % 10 == 0:
                logging.info('Done with burn-in iteration {} / {}'.format(i + 1, n_burn))
            i += 1

        logging.info('Running production')
        i = 0
        for p1, lnp, _ in sampler.sample(p1, lnprob0=lnp, rstate0=rstate, iterations=n_prod):
            if i % 10 == 0:
                logging.info('Done with production iteration {} / {}'.format(i + 1, n_prod))
            i += 1

        # Save the sampler instance as a class variable
        self.sampler = sampler

        # Put the chain in a pandas array for easier access/manipulation
        self.make_emcee_samples(n_burn)
        return


    def make_emcee_samples(self, n_burn):
        """ Make a pandas DataFrame with the emcee samples. Discard the first n_burn samples. """

        ndim = self.sampler.chain.shape[2]
        samples = self.sampler.chain[:, n_burn:, :].reshape((-1, ndim))
        lnprob = self.sampler.lnprobability[:, n_burn:].flatten()
        chain_dict = {self.param_names[i]: samples[:, i] for i in range(self.n_params)}
        chain_dict['lnprob'] = lnprob
        self.samples = pd.DataFrame(data=chain_dict)
        return


    def fit_multinest(self, n_live_points=1000, basename='chains/single-',
                      verbose=True, refit=False, overwrite=False,
                      **kwargs):
        """
        Fits model using MultiNest, via pymultinest. This function was taken almost entirely
        form Timothy Morton's 'isochrones' code on github.

        Parameters:
        ===========
        - n_live_points:    integer
                            Number of live points to use for MultiNest fit.

        - basename:         string
                            Where the MulitNest-generated files will live.
                            By default this will be in a folder named `chains`
                            in the current working directory.  Calling this method
                            will define a `_mnest_basename` attribute for
                            this object.

        - verbose:          boolean
                            Whether you want MultiNest to talk to you.

        - refit, overwrite: boolean
                            Set either of these to true if you want to
                            delete the MultiNest files associated with the
                            given basename and start over.

        - **kwargs:
            Additional keyword arguments will be passed to
            `pymultinest.run)_`.
        """

        try:
            import pymultinest
        except ImportError:
            logging.warn('You must install pymultinest (and MultiNest) to use this backend!')
            return None

        # Make sure the output directory exists
        ensure_dir(basename)

        # If previous fit exists, see if it's using the same
        # observed properties
        prop_nomatch = False
        propfile = '{}properties.json'.format(basename)
        if os.path.exists(propfile):
            with open(propfile) as f:
                props = json.load(f)
            if set(props) != set(self.param_names):
                prop_nomatch = True

        if prop_nomatch and not overwrite:
            raise ValueError('Properties not same as saved chains ' +
                             '(basename {}*). '.format(basename) +
                             'Use overwrite=True to fit.')

        if refit or overwrite:
            files = glob.glob('{}*'.format(basename))
            [os.remove(f) for f in files]

        self._mnest_basename = basename

        pymultinest.run(self.mnest_lnlike, self.mnest_prior, self.n_params,
                        n_live_points=n_live_points, outputfiles_basename=basename,
                        verbose=verbose,
                        **kwargs)

        with open(propfile, 'w') as f:
            json.dump(self.param_names, f, indent=2)

        self._make_mn_samples()

        return

    def _make_mn_samples(self):
        """
        Make MCMC samples out of a multinest run. MUST call fit() method before this!
        """
        chain = np.loadtxt('{}post_equal_weights.dat'.format(self._mnest_basename))
        chain_dict = {self.param_names[i]: chain[:, i] for i in range(self.n_params)}
        chain_dict['lnprob'] = chain[:, -1]

        self.samples = pd.DataFrame(data=chain_dict)
        return


    def predict(self, x, N=100, highest=False):
        """
        predict the y value for the given x values. Use the N most probable MCMC chains if highest=True,
        otherwise use the first N chains.
        """
        if self.samples is None:
            logging.warn('Need to run the fit method before predict!')
            return

        # Find the N best walkers
        if N == 'all':
            N = self.samples.shape[0]
        else:
            N = min(N, self.samples.shape[0])

        if highest:
            samples = self.samples.sort('lnprob', ascending=False)[:N]
        else:
            indices = np.random.randint(0, self.samples.shape[0], N)
            samples = self.samples.ix[indices]

        pars = samples[self.param_names].as_matrix()
        y = np.array([self.model(p, x) for p in pars])
        return y


    def plot_samples(self, x, N=100, ax=None, highest=False, *plot_args, **plot_kws):
        """
        Plot sample curves.

        Parameters:
        ===========
        - x:         numpy.ndarray
                     The x-values to plot at

        - N:         integer
                     The number of samples to take

        - highest:   boolean
                     If True, use the N most probable samples.
                     Otherwise, just take N random samples.

        - ax:        matplotlib axis object
                     The axis to plot the curves on. A new figure will be made if no axis is given.

        Returns:
        =========
        matplotlib axis object, with which to plot other stuff, labels, etc
        """

        y = self.predict(x, N=N)
        if ax is None:
            ax = plt.gca()

        for i in range(N):
            ax.plot(x, y[i], *plot_args, **plot_kws)

        return ax

    def spoof_sampler(self, flatchain, flatlnprobability, force=False):
        """
        Create a sampler object with the flatchain and lnprobability attributes so self.predict will work.
        This is useful for predicting values from pre-tabulated MCMC parameter fits.

        Parameters:
        ===========
        - flatchain:        numpy.ndarray
                            The original sampler.flatchain property

        - lnprobability:    numpy.ndarray
                            The original sampler.lnprobabiliity property

        - force:            boolean
                            Force creation of a sampler object, even if one already exists.
        """
        if self.sampler is not None and not force:
            logging.warn('sampler instance already exists! Use force=True to overwrite.')
            return

        self.sampler = MCSampler_Spoof(flatchain, flatlnprobability)

        # Make samples
        if self.n_params is None:
            self.n_params = flatchain.shape[1]
        if self.param_names is None:
            self.param_names = ['a{}'.format(i) for i in range(self.n_params)]
        chain_dict = {self.param_names[i]: flatchain[:, i] for i in range(self.n_params)}
        chain_dict['lnprob'] = flatlnprobability
        self.samples = pd.DataFrame(data=chain_dict)

        return

    def triangle(self, **kws):
        """
        Make a triangle plot from the samples. All keyword arguments are passed to
        corner.corner()
        """
        if self.samples is None:
            logging.warn('Need to run the fit method first!')
            return

        samples = self.samples[self.param_names].as_matrix()
        corner.corner(samples, labels=self.param_names, **kws)
        return


    @property
    def mnest_analyzer(self):
        """
        PyMultiNest Analyzer object associated with fit.
        See PyMultiNest documentation for more.
        """
        try:
            import pymultinest
        except ImportError:
            print('pymultinest import failed!')
            return {'global evidence': None, 'global evidence error': None}

        return pymultinest.Analyzer(self.n_params, self._mnest_basename)

    @property
    def evidence(self):
        """
        Log(evidence) from multinest fit
        """
        s = self.mnest_analyzer.get_stats()
        return (s['global evidence'], s['global evidence error'])


class MCSampler_Spoof(object):
    def __init__(self, flatchain, flatlnprobability):
        self.flatchain = flatchain
        self.flatlnprobability = flatlnprobability
        return



