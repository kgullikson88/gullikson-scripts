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
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir


try:
    import corner
except ImportError:
    import triangle as corner

try:
	import emcee
except ImportError:
	emcee = None
try:
	import pymultinest
except ImportError:
	pymultinest = None

class Bayesian_LS(object):
        def __init__(self, x=1, y=1, yerr=1, param_names=None):
            """
            Class to perform a bayesian least squares fit to data with errors in only the y-axis.
            :param x:  A numpy ndarray with the independent variable
            :param y:  A numpy ndarray with the dependent variable
            :param yerr:  A numpy ndarray with the uncertainty in the dependent variable
            :param param_names: An iterable of the parameter names. You MUST give this if using the
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
            """
            return np.poly1d(p)(x)


        def lnlike(self, pars):
            """
            likelihood function. This uses the class variables for x,y,xerr, and yerr, as well as the 'model' instance.
            """
            y_pred = self.model(pars, self.x)  # Predict the y value
            # Make the log-likelihood
            return -0.5 * np.sum((self.y - y_pred) ** 2 / self.yerr * 2 + np.log(2*np.pi*self.yerr**2))


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
            """
            return


        def mnest_lnlike(self, cube, ndim, nparams):
            """
            This is probably okay as it is. You may (but probably not) need to override
            _lnlike, but not this one.
            """
            pars = np.array([cube[i] for i in range(nparams)])
            return self._lnlike(pars)


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
            See the doc-strings for those methods to learn what args and kwargs should be for each backend.

            :param backend: string - either 'emcee' or 'multinest'.
            :param args:   A list of arguments to pass to either fit_multinest or fit_emcee
            :param kwargs: A dict of keyword arguments to pass to either fit_multinest or fit_emcee
            :return: None
            """
            if backend.lower() == 'emcee':
                return self.fit_emcee(*args, **kwargs)
            elif backend.lower() == 'multinest':
                return self.fit_multinest(*args, **kwargs)


        def fit_emcee(self, nwalkers=100, n_burn=200, n_prod=1000, guess=True, initial_pars=None, **guess_kws):
            """
            Perform the full MCMC fit using emcee.

            :param nwalkers:  The number of walkers to use in the MCMC sampler
            :param n_burn:   The number of samples to discard for the burn-in portion
            :param n_prod:   The number of MCMC samples to take in the final production sampling
            :param guess:    Flag for whether the data should be fit in a normal way first, to get decent starting parameters.
                             If true, it uses self.guess_fit_parameters and passes guess_kws to the function.
                             If false, it uses initial_pars. You MUST give initial_pars if guess=False!
            :param initial_pars: Initial parameters to use. Should be either a 1d array with the guess pars
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
                    logging.info('Done with burn-in iteration {} / {}'.format(i+1, n_burn))
                i += 1

            logging.info('Running production')
            i = 0
            for p1, lnp, _ in sampler.sample(p1, lnprob0=lnp, rstate0=rstate, iterations=n_prod):
                if i % 10 == 0:
                    logging.info('Done with production iteration {} / {}'.format(i+1, n_prod))
                i += 1

            # Save the sampler instance as a class variable
            self.sampler = sampler

            # Put the chain in a pandas array for easier access/manipulation
            self.make_emcee_samples(n_burn)
            return

        def make_emcee_samples(self, n_burn):
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

            :param n_live_points:  Number of live points to use for MultiNest fit.
            :param basename:
                Where the MulitNest-generated files will live.
                By default this will be in a folder named `chains`
                in the current working directory.  Calling this
                will define a `_mnest_basename` attribute for
                this object.
            :param verbose:
                Whether you want MultiNest to talk to you.
            :param refit, overwrite:
                Set either of these to true if you want to
                delete the MultiNest files associated with the
                given basename and start over.
            :param **kwargs:
                Additional keyword arguments will be passed to
                :func:`pymultinest.run`.
            """
            
            # Make sure pymultinest exists
            if pymultinest is None:
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
            predict the y value for the given x values. Use the N most probable MCMC chains if highest=False,
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


        def plot_samples(self, x, N=100, ax=None, *plot_args, **plot_kws):
            """
            Plot N best-fit curves at x-values x, on axis ax (if given)
            :param x:
            :param N:
            :param ax:
            :return: matplotlib axis object, with which to plot other stuff, label, etc
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
            This is useful for predicting values from pre-tabulated MCMC parameter fits
            :param flatchain: The original sampler.flatchain property
            :param lnprobability: The original sampler.lnprobabiliity property
            :keyword force: Force creation of a sampler object, even if one already exists.
            :return: None
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