from __future__ import print_function, division, absolute_import

from george import kernels, GP

import numpy as np
from kglib import fitters
from scipy.integrate import quad
from scipy.optimize import minimize


class HistFitter(fitters.Bayesian_LS):
    def __init__(self, mcmc_samples, bin_edges):
        """
        Histogram Inference a la Dan Foreman-Mackey

        Parameters:
        ===========
        - mcmc_samples:      numpy array of shape (Nobs, Nsamples)
                             MCMC samples for the thing you want to histogram

        - bin_edges:         numpy.ndarray array
                             The edges of the histogram bins to use.

        """
        self.mcmc_samples = mcmc_samples
        self.bin_edges = bin_edges
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_widths = np.diff(self.bin_edges)
        self.Nbins = self.bin_widths.size
        self.Nobs = self.mcmc_samples.shape[0]

        # Find which bin each q falls in
        self.bin_idx = np.digitize(self.mcmc_samples, self.bin_edges) - 1

        # Determine the censoring function for each bin (used in the integral)
        self.censor_integrals = np.array([quad(func=self.censoring_fcn,
                                               a=left, b=right)[0] for (left, right) in
                                          zip(self.bin_edges[:-1], self.bin_edges[1:])])

        # Set values needed for multinest fitting
        self.n_params = self.Nbins
        self.param_names = [r'$\theta_{}$'.format(i) for i in range(self.Nbins)]


    def lnlike(self, pars):
        # Pull theta out of pars
        theta = pars[:self.Nbins]

        # Generate the inner summation
        gamma = np.ones_like(self.bin_idx) * np.nan
        good = (self.bin_idx < self.Nbins) & (self.bin_idx >= 0)  # nans in q get put in nonexistent bins
        gamma[good] = self.Nobs * self.censoring_fcn(self.mcmc_samples[good]) * theta[self.bin_idx[good]]
        summation = np.nanmean(gamma, axis=1)

        # Calculate the integral
        I = self._integral_fcn(theta)

        # Generate the log-likelihood
        ll = -I + np.nansum(np.log(summation))
        return ll


    def lnprior(self, pars):
        """ Override this if you want to set a better prior on the bin heights. """

        if all([p > 0 and p < 10 for p in pars]):
            return 0
        return -np.inf


    def lnprob(self, pars):
        lp = self.lnprior(pars)
        return lp + self.lnlike(pars) if np.isfinite(lp) else -np.inf


    def _integral_fcn(self, theta):
        return np.sum(theta * self.censor_integrals) * self.Nobs


    def censoring_fcn(self, value):
        """
        Censoring function. This should return the completeness of your survey to the given value.
        """
        return 1.0


    def guess_fit(self):

        def errfcn(pars):
            ll = self.lnprob(pars)
            return -ll

        initial_guess = np.ones_like(self.bin_centers)
        bounds = [[1e-3, None] for p in initial_guess]
        out = minimize(errfcn, initial_guess, bounds=bounds)
        return out.x


    def mnest_prior(self, cube, ndim, nparams):
        # All bins are in the range (0, 10)
        for i in range(self.Nbins):
            cube[i] *= 10

        return


class CensoredHistFitter(HistFitter):
    """
    Inherits from HistFitter, but actually defines the censoring function
    """

    def censoring_fcn(self, val, alpha=40, beta=0.25):
        # sigmoid censoring function. Change this for the real deal!
        return 1.0 / (1.0 + np.exp(-alpha * (val - beta)))


class SmoothHistFitter(CensoredHistFitter):
    """
    A subclass of HistogramFitter that puts a gaussian process smoothing prior on the bin heights
    """

    def __init__(self, *args, **kwargs):
        super(SmoothHistFitter, self).__init__(*args, **kwargs)
        self.smoothing = self.mcmc_samples.shape[0] / self.Nbins
        self.n_params = self.Nbins + 4
        self.param_names = [r'$\theta_{}$'.format(i) for i in range(self.Nbins)]
        self.param_names.extend(('lna', 'lntau', 'lnerr', 'mean'))

    def lnprior(self, pars):
        """
        Smoothing prior using gaussian process.
        We will learn the hyperparameters and marginalize over them.
        """
        theta = pars[:self.Nbins]
        if np.any(theta < 0):
            return -np.inf
        a, tau, err = np.exp(pars[self.Nbins:-1])
        mean = pars[-1]
        kernel = a * kernels.ExpSquaredKernel(tau)
        gp = GP(kernel, mean=mean)
        gp.compute(self.bin_centers, yerr=err)
        return gp.lnlikelihood(theta) / self.smoothing

    def guess_fit(self):
        """
        This doesn't work too great, but the full MCMC fit looks good.
        """
        def errfcn(pars):
            ll = self.lnprob(pars)
            return -ll

        # Set up initial guesses
        initial_guess = np.ones(self.bin_centers.size + 4)
        initial_guess[-4] = 0.0
        initial_guess[-3] = -0.25
        initial_guess[-2] = -1.0
        initial_guess[-1] = -1.0

        # Set up bounds
        bounds = [[1e-3, None] for p in self.bin_centers]
        bounds.append([-10, 20])
        bounds.append([-10, 10])
        bounds.append((-1, 5))
        bounds.append((-10, 10))

        # Minimize
        out = minimize(errfcn, initial_guess, bounds=bounds)
        return out.x

    def _lnlike(self, pars):
        return self.lnprob(pars)

    def mnest_prior(self, cube, ndim, nparams):
        for i in range(self.Nbins):
            cube[i] *= 10

        cube[self.Nbins] = cube[self.Nbins] * 30 - 10
        cube[self.Nbins + 1] = cube[self.Nbins + 1] * 20 - 10
        cube[self.Nbins + 2] = cube[self.Nbins + 2] * 7 - 2
        cube[self.Nbins + 3] = cube[self.Nbins + 3] * 20 - 10
        return