from __future__ import print_function, division, absolute_import

from george import kernels, GP

import numpy as np
import fitters
from scipy.integrate import quad


class HistFitter(fitters.Bayesian_LS):
    def __init__(self, qvals, bin_edges):
        """
        Histogram Inference a la Dan Foreman-Mackey

        Parameters:
        ===========
        - qvals:      numpy array of shape (Nobs, Nsamples)
                      The MCMC samples for the mass-ratio distribution of all companions

        - bin_edges:  numpy array
                      The edges of the histogram bins to use.

        """
        self.qvals = qvals
        self.bin_edges = bin_edges
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_widths = np.diff(self.bin_edges)
        self.Nbins = self.bin_widths.size
        self.Nobs = self.qvals.shape[0]

        # Find which bin each q falls in
        self.bin_idx = np.digitize(self.qvals, self.bin_edges) - 1

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

        # Normalize theta
        # theta /= np.sum(theta * self.bin_widths)

        # Generate the inner summation
        gamma = np.ones_like(self.bin_idx) * np.nan
        good = (self.bin_idx < self.Nbins) & (self.bin_idx >= 0)  # nans in q get put in nonexistent bins
        gamma[good] = self.Nobs * self.censoring_fcn(self.qvals[good]) * theta[self.bin_idx[good]]
        summation = np.nanmean(gamma, axis=1)

        # Calculate the integral
        I = self._integral_fcn(theta)

        # Generate the log-likelihood
        ll = -I + np.nansum(np.log(summation))
        return ll


    def lnprior(self, pars):
        # Pull theta out of pars
        theta = pars[:self.Nbins]

        return 0.0


    def lnprob(self, pars):
        lp = self.lnprior(pars)
        return lp + self.lnlike(pars) if np.isfinite(lp) else -np.inf


    def _integral_fcn(self, theta):
        return np.sum(theta * self.censor_integrals) * self.Nobs


    def censoring_fcn(self, q):
        """
        Censoring function. This should take a mass-ratio (or array of mass-ratios), and return the completeness
        as a number between 0 and 1.
        """
        return 1.0


    def guess_fit(self):
        from scipy.optimize import minimize

        def errfcn(pars):
            ll = self.lnprob(pars)
            return -ll

        initial_guess = np.ones_like(self.bin_centers)
        bounds = [[1e-3, None] for p in initial_guess]
        out = minimize(errfcn, initial_guess, bounds=bounds)
        return out.x


    def mnest_prior(self, cube, ndim, nparams):
        for i in range(self.Nbins):
            cube[i] *= 10

        return


class CensoredHistFitter(HistFitter):
    def censoring_fcn(self, q, alpha=40, beta=0.25):
        # sigmoid censoring function. Change this for the real deal!
        return 1.0 / (1.0 + np.exp(-alpha * (q - beta)))


class SmoothHistFitter(CensoredHistFitter):
    """ A subclass of HistogramFitter that puts a gaussian process smoothing prior on the bin heights
    """

    def __init__(self, *args, **kwargs):
        super(SmoothHistFitter, self).__init__(*args, **kwargs)
        self.smoothing = self.qvals.shape[0] / self.Nbins
        self.n_params = self.Nbins + 4
        self.param_names = [r'$\theta_{}$'.format(i) for i in range(self.Nbins)]
        self.param_names.extend(('lna', 'lntau', 'lnerr', 'mean'))

    def lnprior(self, pars):
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
        from scipy.optimize import minimize

        def errfcn(pars):
            ll = self.lnprob(pars)
            # print(pars, ll)
            return -ll

        initial_guess = np.ones(self.bin_centers.size + 4)
        initial_guess[-4] = 0.0
        initial_guess[-3] = -0.25
        initial_guess[-2] = -1.0
        initial_guess[-1] = -1.0
        bounds = [[1e-3, None] for p in self.bin_centers]
        bounds.append([-10, 20])
        bounds.append([-10, 10])
        bounds.append((-1, 5))
        bounds.append((-10, 10))
        out = minimize(errfcn, initial_guess, bounds=bounds)
        return out.x

    def _lnlike(self, pars):
        return self.lnprob(pars)

    def mnest_prior(self, cube, ndim, nparams):
        for i in range(self.Nbins):
            cube[i] *= 10
        # cube[:self.Nbins] *= 15
        cube[self.Nbins] = cube[self.Nbins] * 30 - 10
        cube[self.Nbins + 1] = cube[self.Nbins + 1] * 20 - 10
        cube[self.Nbins + 2] = cube[self.Nbins + 2] * 7 - 2
        cube[self.Nbins + 3] = cube[self.Nbins + 3] * 20 - 10
        return